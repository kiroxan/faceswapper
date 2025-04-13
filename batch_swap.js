const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
const { promisify } = require('util');
const readdir = promisify(fs.readdir);
const stat = promisify(fs.stat);

/**
 * Submits a batch of face swap tasks
 * @param {Object} config - Configuration for the batch job
 * @param {string} config.mainDir - Directory containing main/target images
 * @param {string} config.refDir - Directory containing reference face images
 * @param {string} config.maskDir - Optional directory containing mask images
 * @param {boolean} config.useMatchingNames - Whether to match files by name
 * @param {string} config.outputDir - Directory for output files
 * @param {string} config.baseUrl - Base URL of the face swap API
 */
async function batchSwap(config) {
  try {
    // Validate and set defaults
    config = {
      mainDir: './batch/main',
      refDir: './batch/ref',
      maskDir: null,
      useMatchingNames: true,
      outputDir: './output/batch',
      baseUrl: 'https://v1yfznzb7lkce6-7860.proxy.runpod.net',
      ...config
    };
    
    // Create output directory
    if (!fs.existsSync(config.outputDir)) {
      fs.mkdirSync(config.outputDir, { recursive: true });
    }
    
    // Get list of files
    const mainFiles = await getImageFiles(config.mainDir);
    const refFiles = await getImageFiles(config.refDir);
    let maskFiles = [];
    
    if (config.maskDir && fs.existsSync(config.maskDir)) {
      maskFiles = await getImageFiles(config.maskDir);
    }
    
    if (mainFiles.length === 0) {
      console.error(`No image files found in main directory: ${config.mainDir}`);
      return;
    }
    
    if (refFiles.length === 0) {
      console.error(`No image files found in reference directory: ${config.refDir}`);
      return;
    }
    
    console.log(`Found ${mainFiles.length} main images and ${refFiles.length} reference images`);
    if (maskFiles.length > 0) {
      console.log(`Found ${maskFiles.length} mask images`);
    }
    
    // Determine task pairs
    const tasks = [];
    
    if (config.useMatchingNames) {
      // Match files by name
      for (const mainFile of mainFiles) {
        const mainBaseName = path.parse(mainFile).name;
        
        // Find matching reference file
        const matchingRef = refFiles.find(file => {
          return path.parse(file).name === mainBaseName;
        });
        
        // Find matching mask file if available
        let matchingMask = null;
        if (maskFiles.length > 0) {
          matchingMask = maskFiles.find(file => {
            return path.parse(file).name === mainBaseName;
          });
        }
        
        if (matchingRef) {
          tasks.push({
            main: path.join(config.mainDir, mainFile),
            ref: path.join(config.refDir, matchingRef),
            mask: matchingMask ? path.join(config.maskDir, matchingMask) : null
          });
        } else {
          console.warn(`No matching reference file found for ${mainFile}`);
        }
      }
    } else {
      // Use single reference for all main images
      const refFile = refFiles[0];
      
      for (const mainFile of mainFiles) {
        const mainBaseName = path.parse(mainFile).name;
        
        // Find matching mask file if available
        let matchingMask = null;
        if (maskFiles.length > 0) {
          matchingMask = maskFiles.find(file => {
            return path.parse(file).name === mainBaseName;
          });
        }
        
        tasks.push({
          main: path.join(config.mainDir, mainFile),
          ref: path.join(config.refDir, refFile),
          mask: matchingMask ? path.join(config.maskDir, matchingMask) : null
        });
      }
    }
    
    console.log(`Created ${tasks.length} swap tasks`);
    
    // Submit each task
    const taskResults = [];
    
    for (let i = 0; i < tasks.length; i++) {
      const task = tasks[i];
      console.log(`\nSubmitting task ${i + 1}/${tasks.length}:`);
      console.log(`  Main: ${task.main}`);
      console.log(`  Ref: ${task.ref}`);
      if (task.mask) console.log(`  Mask: ${task.mask}`);
      
      const taskId = await submitSwapTask(task.main, task.ref, task.mask, config.baseUrl);
      
      if (taskId) {
        const taskInfo = {
          taskId,
          main: task.main,
          ref: task.ref,
          mask: task.mask,
          status: 'queued',
          mainBaseName: path.parse(task.main).name
        };
        
        taskResults.push(taskInfo);
        console.log(`  Task ID: ${taskId}`);
      }
    }
    
    // Save task information to a JSON file
    const batchInfo = {
      batchId: `batch_${Date.now()}`,
      timestamp: new Date().toISOString(),
      config,
      tasks: taskResults
    };
    
    const batchInfoPath = path.join(config.outputDir, 'batch_info.json');
    fs.writeFileSync(batchInfoPath, JSON.stringify(batchInfo, null, 2));
    
    console.log(`\nBatch job submitted with ${taskResults.length} tasks`);
    console.log(`Batch information saved to ${batchInfoPath}`);
    console.log('\nYou can use check_tasks.js to monitor task status:');
    console.log('  node check_tasks.js list');
    console.log('\nOr check and download results for specific tasks:');
    console.log('  node get_result.js <task_id>');
    
    return batchInfo;
    
  } catch (error) {
    console.error('Error in batch processing:', error.message);
    if (error.response) {
      console.error('Response status:', error.response.status);
    }
  }
}

/**
 * Submits a single face swap task
 * @param {string} mainPath - Path to the main/target image
 * @param {string} refPath - Path to the reference face image
 * @param {string} maskPath - Optional path to the mask image
 * @param {string} baseUrl - Base URL of the face swap API
 * @returns {string} The task ID if successful, null otherwise
 */
async function submitSwapTask(mainPath, refPath, maskPath = null, baseUrl) {
  try {
    // Create form data with files
    const form = new FormData();
    form.append('main', fs.createReadStream(mainPath));
    form.append('ref', fs.createReadStream(refPath));
    
    if (maskPath && fs.existsSync(maskPath)) {
      form.append('mask', fs.createReadStream(maskPath));
    }
    
    // Submit the async task
    const response = await axios.post(
      `${baseUrl}/swap/async`,
      form,
      { headers: form.getHeaders() }
    );
    
    if (response.data && response.data.task_id) {
      return response.data.task_id;
    }
    
    return null;
  } catch (error) {
    console.error('Error submitting task:', error.message);
    if (error.response) {
      console.error('Response status:', error.response.status);
    }
    return null;
  }
}

/**
 * Gets a list of image files from a directory
 * @param {string} dir - Directory to scan
 * @returns {Array} List of image filenames
 */
async function getImageFiles(dir) {
  try {
    const files = await readdir(dir);
    
    // Filter for image files
    const imageExts = ['.jpg', '.jpeg', '.png', '.webp'];
    const imageFiles = [];
    
    for (const file of files) {
      const filePath = path.join(dir, file);
      const fileStat = await stat(filePath);
      
      if (fileStat.isFile() && imageExts.includes(path.extname(file).toLowerCase())) {
        imageFiles.push(file);
      }
    }
    
    return imageFiles;
  } catch (error) {
    console.error(`Error reading directory ${dir}:`, error.message);
    return [];
  }
}

/**
 * Downloads results for a batch job
 * @param {string} batchInfoPath - Path to the batch info JSON file
 */
async function downloadBatchResults(batchInfoPath) {
  try {
    // Load batch info
    if (!fs.existsSync(batchInfoPath)) {
      console.error(`Batch info file not found: ${batchInfoPath}`);
      return;
    }
    
    const batchInfo = JSON.parse(fs.readFileSync(batchInfoPath, 'utf8'));
    const tasks = batchInfo.tasks;
    const outputDir = batchInfo.config.outputDir;
    const baseUrl = batchInfo.config.baseUrl;
    
    console.log(`Checking results for ${tasks.length} tasks in batch ${batchInfo.batchId}`);
    
    let completed = 0;
    let failed = 0;
    let pending = 0;
    
    // Check each task
    for (let i = 0; i < tasks.length; i++) {
      const task = tasks[i];
      console.log(`\nChecking task ${i + 1}/${tasks.length}: ${task.taskId}`);
      
      try {
        const response = await axios.get(`${baseUrl}/tasks/${task.taskId}`);
        const taskData = response.data;
        const status = taskData.status;
        
        console.log(`  Status: ${status}`);
        
        if (status === 'completed' && taskData.result_url) {
          // Download the result
          const outputName = `${task.mainBaseName}_swapped.png`;
          const outputPath = path.join(outputDir, outputName);
          
          console.log(`  Downloading to ${outputPath}...`);
          const imageResponse = await axios.get(taskData.result_url, { responseType: 'arraybuffer' });
          fs.writeFileSync(outputPath, imageResponse.data);
          
          // Update task info
          task.status = 'completed';
          task.resultPath = outputPath;
          completed++;
          
          console.log(`  Downloaded successfully`);
        } else if (status === 'failed') {
          console.error(`  Task failed: ${taskData.error || 'Unknown error'}`);
          task.status = 'failed';
          task.error = taskData.error;
          failed++;
        } else {
          console.log(`  Task is still ${status}`);
          task.status = status;
          pending++;
        }
      } catch (error) {
        console.error(`  Error checking task: ${error.message}`);
        task.status = 'error';
        task.error = error.message;
        failed++;
      }
    }
    
    // Update batch info file
    batchInfo.lastChecked = new Date().toISOString();
    fs.writeFileSync(batchInfoPath, JSON.stringify(batchInfo, null, 2));
    
    // Print summary
    console.log('\nSummary:');
    console.log(`  Completed: ${completed}/${tasks.length}`);
    console.log(`  Failed: ${failed}/${tasks.length}`);
    console.log(`  Pending: ${pending}/${tasks.length}`);
    console.log(`\nBatch information updated in ${batchInfoPath}`);
    
  } catch (error) {
    console.error('Error downloading batch results:', error.message);
  }
}

// Parse command line arguments
const args = process.argv.slice(2);
const command = args[0] || 'submit';

if (command === 'submit') {
  // Look for config options
  const config = {
    mainDir: args.length > 1 ? args[1] : './batch/main',
    refDir: args.length > 2 ? args[2] : './batch/ref',
    maskDir: args.length > 3 ? args[3] : null,
    useMatchingNames: true,
    baseUrl: 'https://v1yfznzb7lkce6-7860.proxy.runpod.net'
  };
  
  // Create batch directories if they don't exist
  [config.mainDir, config.refDir].forEach(dir => {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  });
  
  console.log('Batch processing configuration:');
  console.log(`  Main images: ${config.mainDir}`);
  console.log(`  Reference images: ${config.refDir}`);
  console.log(`  Mask images: ${config.maskDir || 'None'}`);
  console.log(`  Match by filename: ${config.useMatchingNames}`);
  console.log('\nStarting batch processing...');
  
  batchSwap(config);
  
} else if (command === 'results') {
  // Download results for a batch
  const batchInfoPath = args[1] || './output/batch/batch_info.json';
  downloadBatchResults(batchInfoPath);
  
} else if (command === 'help' || command === '--help') {
  console.log(`
Usage: node batch_swap.js [command] [options]

Commands:
  submit            Submit a new batch job (default)
  results <path>    Download results for a batch job
  help              Show this help message

For 'submit' command:
  node batch_swap.js submit [mainDir] [refDir] [maskDir]

  mainDir     Directory containing main/target images (default: ./batch/main)
  refDir      Directory containing reference face images (default: ./batch/ref)
  maskDir     Optional directory containing mask images

For 'results' command:
  node batch_swap.js results [batchInfoPath]

  batchInfoPath   Path to the batch info JSON file (default: ./output/batch/batch_info.json)
  `);
} else {
  console.error(`Unknown command: ${command}`);
  console.error('Use "node batch_swap.js help" for usage information');
} 