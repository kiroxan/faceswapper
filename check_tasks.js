const axios = require('axios');
const fs = require('fs');

/**
 * Lists recent face swap tasks and their status
 * @param {string} baseUrl - Base URL of the face swap API
 * @param {number} limit - Maximum number of tasks to list
 * @param {string} status - Optional status filter ('queued', 'processing', 'completed', 'failed')
 */
async function listTasks(
  baseUrl = 'https://v1yfznzb7lkce6-7860.proxy.runpod.net',
  limit = 10,
  status = null
) {
  try {
    // Build the query URL
    let url = `${baseUrl}/tasks?limit=${limit}`;
    if (status) {
      url += `&status=${status}`;
    }
    
    console.log(`Fetching tasks from: ${url}`);
    
    const response = await axios.get(url);
    const tasksData = response.data;
    
    if (!tasksData.tasks || tasksData.tasks.length === 0) {
      console.log('No tasks found.');
      return;
    }
    
    console.log(`\nFound ${tasksData.tasks.length} tasks:`);
    console.log('==================================');
    
    // Display task information
    tasksData.tasks.forEach((task, index) => {
      // Determine model used
      let modelInfo = "";
      if (task.parameters && task.parameters.use_ace === true) {
        modelInfo = " [ACE_Plus Portrait]";
      } else {
        modelInfo = " [Standard InsightFace]";
      }
      
      console.log(`Task #${index + 1}:`);
      console.log(`  ID: ${task.id}`);
      console.log(`  Status: ${task.status}${modelInfo}`);
      console.log(`  Created: ${new Date(task.created).toLocaleString()}`);
      
      if (task.start_time) {
        console.log(`  Started: ${new Date(task.start_time).toLocaleString()}`);
      }
      
      if (task.end_time) {
        console.log(`  Ended: ${new Date(task.end_time).toLocaleString()}`);
      }
      
      // Show ACE_Plus parameters if available
      if (task.parameters && task.parameters.use_ace === true) {
        console.log('  ACE_Plus Parameters:');
        if (task.parameters.prompt) {
          console.log(`    Prompt: "${task.parameters.prompt}"`);
        }
        if (task.parameters.lora_strength) {
          console.log(`    LoRA Strength: ${task.parameters.lora_strength}`);
        }
      }
      
      if (task.status === 'completed' && task.result_url) {
        console.log(`  Result URL: ${task.result_url}`);
        console.log(`  To download: node get_result.js ${task.id}`);
      }
      
      if (task.status === 'failed' && task.error) {
        console.log(`  Error: ${task.error}`);
      }
      
      console.log('----------------------------------');
    });
    
    // Save task IDs to file for reference
    const taskIds = tasksData.tasks.map(task => task.id);
    fs.writeFileSync('./output/recent_task_ids.json', JSON.stringify(taskIds, null, 2));
    console.log('Task IDs saved to ./output/recent_task_ids.json');
    
    // Print summary by status
    const statusCounts = tasksData.tasks.reduce((counts, task) => {
      counts[task.status] = (counts[task.status] || 0) + 1;
      return counts;
    }, {});
    
    // Count by model type
    const modelCounts = tasksData.tasks.reduce((counts, task) => {
      const modelType = task.parameters && task.parameters.use_ace === true ? 'ACE_Plus' : 'Standard';
      counts[modelType] = (counts[modelType] || 0) + 1;
      return counts;
    }, {});
    
    console.log('\nSummary by Status:');
    Object.entries(statusCounts).forEach(([status, count]) => {
      console.log(`  ${status}: ${count}`);
    });
    
    console.log('\nSummary by Model:');
    Object.entries(modelCounts).forEach(([model, count]) => {
      console.log(`  ${model}: ${count}`);
    });
    
  } catch (error) {
    console.error('Error fetching tasks:', error.message);
    if (error.response) {
      console.error('Response status:', error.response.status);
    }
  }
}

/**
 * Get details for a specific task
 * @param {string} taskId - ID of the task to check
 * @param {string} baseUrl - Base URL of the face swap API
 */
async function getTaskDetails(taskId, baseUrl = 'https://v1yfznzb7lkce6-7860.proxy.runpod.net') {
  try {
    console.log(`Fetching details for task: ${taskId}`);
    
    const response = await axios.get(`${baseUrl}/tasks/${taskId}`);
    const taskData = response.data;
    
    // Determine model used
    let modelInfo = "";
    if (taskData.parameters && taskData.parameters.use_ace === true) {
      modelInfo = " [ACE_Plus Portrait]";
    } else {
      modelInfo = " [Standard InsightFace]";
    }
    
    console.log('\nTask Details:');
    console.log('==================================');
    console.log(`ID: ${taskData.id}`);
    console.log(`Status: ${taskData.status}${modelInfo}`);
    console.log(`Created: ${new Date(taskData.created).toLocaleString()}`);
    
    if (taskData.start_time) {
      console.log(`Started: ${new Date(taskData.start_time).toLocaleString()}`);
    }
    
    if (taskData.end_time) {
      console.log(`Ended: ${new Date(taskData.end_time).toLocaleString()}`);
    }
    
    if (taskData.parameters) {
      console.log('\nParameters:');
      
      // First show model type
      console.log(`  Model: ${taskData.parameters.use_ace === true ? 'ACE_Plus Portrait' : 'Standard InsightFace'}`);
      
      // Show other parameters
      if (taskData.parameters.prompt) {
        console.log(`  Prompt: "${taskData.parameters.prompt}"`);
      }
      
      if (taskData.parameters.negative_prompt) {
        console.log(`  Negative Prompt: "${taskData.parameters.negative_prompt}"`);
      }
      
      // Show ACE_Plus specific parameters
      if (taskData.parameters.use_ace === true) {
        console.log('\n  ACE_Plus Specific Parameters:');
        console.log(`    LoRA Strength: ${taskData.parameters.lora_strength || 0.7}`);
        console.log(`    Guidance Scale: ${taskData.parameters.guidance_scale || 7.5}`);
        console.log(`    Steps: ${taskData.parameters.num_inference_steps || 30}`);
        if (taskData.parameters.seed) {
          console.log(`    Seed: ${taskData.parameters.seed}`);
        }
      } else {
        if (taskData.parameters.prompt_strength) {
          console.log(`  Prompt Strength: ${taskData.parameters.prompt_strength}`);
        }
      }
    }
    
    if (taskData.status === 'completed' && taskData.result_url) {
      console.log(`\nResult URL: ${taskData.result_url}`);
      console.log(`To download: node get_result.js ${taskData.id}`);
    }
    
    if (taskData.status === 'failed' && taskData.error) {
      console.log(`\nError: ${taskData.error}`);
    }
    
    // Add task execution time if available
    if (taskData.start_time && taskData.end_time) {
      const startTime = new Date(taskData.start_time);
      const endTime = new Date(taskData.end_time);
      const duration = (endTime - startTime) / 1000; // in seconds
      console.log(`\nExecution Time: ${duration.toFixed(1)} seconds`);
    }
    
  } catch (error) {
    console.error('Error fetching task details:', error.message);
    if (error.response) {
      console.error('Response status:', error.response.status);
    }
  }
}

// Parse command line arguments
const args = process.argv.slice(2);
const options = {
  command: 'list',
  taskId: null,
  limit: 10,
  status: null,
  baseUrl: 'https://v1yfznzb7lkce6-7860.proxy.runpod.net'
};

// Process arguments
for (let i = 0; i < args.length; i++) {
  const arg = args[i];
  
  if (arg === 'list') {
    options.command = 'list';
  } else if (arg === 'get' && i + 1 < args.length) {
    options.command = 'get';
    options.taskId = args[++i];
  } else if (arg === '--limit' && i + 1 < args.length) {
    options.limit = parseInt(args[++i], 10);
  } else if (arg === '--status' && i + 1 < args.length) {
    options.status = args[++i];
  } else if (arg === '--url' && i + 1 < args.length) {
    options.baseUrl = args[++i];
  } else if (arg === '--model' && i + 1 < args.length) {
    // Filter by model type
    options.model = args[++i].toLowerCase();
    if (options.model !== 'ace' && options.model !== 'standard') {
      console.error('Invalid model filter. Use "ace" or "standard".');
      process.exit(1);
    }
  } else if (arg === '--help') {
    console.log(`
Usage: node check_tasks.js [command] [options]

Commands:
  list               List recent tasks (default)
  get <task_id>      Get details for a specific task

Options:
  --limit <number>   Maximum number of tasks to list (default: 10)
  --status <status>  Filter tasks by status (queued, processing, completed, failed)
  --model <type>     Filter tasks by model type (ace, standard)
  --url <url>        Base URL of the face swap API
  --help             Show this help message
    `);
    process.exit(0);
  }
}

// Create output directory if it doesn't exist
if (!fs.existsSync('./output')) {
  fs.mkdirSync('./output', { recursive: true });
}

// Execute the appropriate command
if (options.command === 'list') {
  listTasks(options.baseUrl, options.limit, options.status);
} else if (options.command === 'get' && options.taskId) {
  getTaskDetails(options.taskId, options.baseUrl);
} else {
  console.error('Invalid command or missing required arguments. Use --help for usage information.');
} 