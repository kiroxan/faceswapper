const axios = require('axios');
const fs = require('fs');
const path = require('path');

/**
 * Checks the status of a face swap task and downloads the result when ready
 * @param {string} taskId - The UUID of the task to check
 * @param {string} outputDir - Directory to save the result image
 * @param {string} baseUrl - Base URL of the face swap API
 * @param {number} maxRetries - Maximum number of status check retries
 * @param {number} retryInterval - Interval between retries in milliseconds
 */
async function getSwapResult(
  taskId, 
  outputDir = './output',
  baseUrl = 'https://v1yfznzb7lkce6-7860.proxy.runpod.net',
  maxRetries = 30,
  retryInterval = 2000
) {
  console.log(`Checking status for task: ${taskId}`);
  
  // Create output directory if it doesn't exist
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  let retries = 0;
  let completed = false;
  
  while (!completed && retries < maxRetries) {
    try {
      const response = await axios.get(`${baseUrl}/tasks/${taskId}`);
      const taskData = response.data;
      const status = taskData.status;
      
      // Determine model used
      let modelInfo = "";
      if (taskData.parameters && taskData.parameters.use_ace === true) {
        modelInfo = " [ACE_Plus Portrait]";
      } else {
        modelInfo = " [Standard InsightFace]";
      }
      
      console.log(`Status: ${status}${modelInfo} (attempt ${retries + 1}/${maxRetries})`);
      
      if (status === 'completed') {
        console.log('Task completed successfully!');
        
        // Get the result URL
        const resultUrl = taskData.result_url;
        console.log(`Result URL: ${resultUrl}`);
        
        // Display model parameters if using ACE_Plus
        if (taskData.parameters && taskData.parameters.use_ace === true) {
          console.log('\nACE_Plus Portrait Parameters:');
          console.log(`  LoRA Strength: ${taskData.parameters.lora_strength || 0.7}`);
          console.log(`  Guidance Scale: ${taskData.parameters.guidance_scale || 7.5}`);
          console.log(`  Steps: ${taskData.parameters.num_inference_steps || 30}`);
          if (taskData.parameters.seed) {
            console.log(`  Seed: ${taskData.parameters.seed}`);
          }
          if (taskData.parameters.prompt) {
            console.log(`  Prompt: "${taskData.parameters.prompt}"`);
          }
          if (taskData.parameters.negative_prompt) {
            console.log(`  Negative Prompt: "${taskData.parameters.negative_prompt}"`);
          }
        }
        
        // Download the result image
        const outputPath = path.join(outputDir, `result_${taskId}.png`);
        console.log(`\nDownloading result to ${outputPath}...`);
        
        const imageResponse = await axios.get(resultUrl, { responseType: 'arraybuffer' });
        fs.writeFileSync(outputPath, imageResponse.data);
        
        console.log(`Result saved to ${outputPath}`);
        completed = true;
        break;
      } else if (status === 'failed') {
        console.error('Task failed:', taskData.error || 'Unknown error');
        completed = true;
        break;
      } else {
        // Display job details while waiting
        if (retries === 0 && taskData.parameters) {
          if (taskData.parameters.use_ace === true) {
            console.log('Using ACE_Plus Portrait Model');
            if (taskData.parameters.prompt) {
              console.log(`Prompt: "${taskData.parameters.prompt}"`);
            }
          } else {
            console.log('Using Standard InsightFace Model');
          }
        }
        
        // Task is still processing, wait and retry
        await new Promise(resolve => setTimeout(resolve, retryInterval));
        retries++;
      }
    } catch (error) {
      console.error('Error checking task status:', error.message);
      if (error.response) {
        console.error('Response status:', error.response.status);
      }
      retries++;
      await new Promise(resolve => setTimeout(resolve, retryInterval));
    }
  }
  
  if (!completed) {
    console.error(`Reached maximum retries (${maxRetries}) without task completion.`);
    console.error('You can try again later with the same task ID.');
  }
}

// Get task ID from command line arguments
const args = process.argv.slice(2);
let taskId;

if (args.length > 0) {
  // Use task ID from command line argument
  taskId = args[0];
} else {
  // Try to read from the last_task_id.txt file
  try {
    if (fs.existsSync('./output/last_task_id.txt')) {
      taskId = fs.readFileSync('./output/last_task_id.txt', 'utf8').trim();
    }
  } catch (error) {
    console.error('Error reading last task ID file:', error.message);
  }
}

if (!taskId) {
  console.error('No task ID provided. Please specify a task ID as a command line argument.');
  console.error('Usage: node get_result.js <task_id>');
  process.exit(1);
}

// Execute the main function
getSwapResult(taskId); 