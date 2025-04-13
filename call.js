const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

async function runAsyncSwap(options = {}) {
  // Set default options
  const defaults = {
    mainPath: './test/main.png',
    refPath: './test/ref.png',
    maskPath: null,
    prompt: '',
    negativePrompt: '',
    useAce: true,
    useFft: false,
    loraStrength: 0.7,
    guidanceScale: 7.5,
    steps: 30,
    seed: null,
    apiUrl: 'https://v1yfznzb7lkce6-7860.proxy.runpod.net'
  };
  
  // Merge defaults with provided options
  const config = { ...defaults, ...options };
  
  // If FFT mode is enabled, adjust the parameters to optimal values
  if (config.useFft) {
    // Set optimal FFT parameters if not already set
    if (config.guidanceScale < 8.5) config.guidanceScale = 8.5;
    if (config.steps < 35) config.steps = 35;
    
    // Set FFT-optimized prompts if not provided
    if (!config.prompt) {
      config.prompt = "ultra realistic photograph of person, highly detailed face, clear eyes, 8k, masterpiece";
    }
    
    if (!config.negativePrompt) {
      config.negativePrompt = "blurry, low quality, cartoon, drawing, painting, 3d render, deformed face, bad eyes, bad mouth";
    }
  }
  
  // Create form data
  const form = new FormData();
  form.append('ref', fs.createReadStream(config.refPath));  // Face to use
  form.append('main', fs.createReadStream(config.mainPath)); // Where to place the face
  
  // Add mask if provided
  if (config.maskPath && fs.existsSync(config.maskPath)) {
    form.append('mask', fs.createReadStream(config.maskPath));
  }
  
  // Add other parameters
  if (config.prompt) form.append('prompt', config.prompt);
  if (config.negativePrompt) form.append('negative_prompt', config.negativePrompt);
  
  // Add ACE_Plus specific parameters
  form.append('use_ace', config.useAce.toString());
  form.append('use_fft', config.useFft.toString());
  
  if (config.useAce) {
    form.append('lora_strength', config.loraStrength.toString());
    form.append('guidance_scale', config.guidanceScale.toString());
    form.append('num_inference_steps', config.steps.toString());
    if (config.seed !== null) form.append('seed', config.seed.toString());
  } else {
    form.append('prompt_strength', config.loraStrength.toString());
  }
  
  // Determine endpoint - use specialized endpoint if ACE
  let endpoint;
  if (config.useAce) {
    if (config.useFft) {
      endpoint = `${config.apiUrl}/swap/ace/fft/async`;
    } else {
      endpoint = `${config.apiUrl}/swap/ace/async`;
    }
  } else {
    endpoint = `${config.apiUrl}/swap/async`;
  }
  
  try {
    // Submit the async task
    console.log(`Submitting task to ${endpoint}`);
    console.log(`Using model: ${config.useAce ? 'ACE_Plus portrait' : 'Standard InsightFace'}`);
    
    const response = await axios.post(
      endpoint,
      form,
      {
        headers: form.getHeaders()
      }
    );
    
    const taskData = response.data;
    const taskId = taskData.task_id;
    
    console.log(`Task submitted successfully with ID: ${taskId}`);
    console.log(`Initial status: ${taskData.status}`);
    console.log(`Check task status at: ${config.apiUrl}/tasks/${taskId}`);
    console.log('To download the result once ready, use get_result.js with this task ID');
    
    // Save task ID to file for reference
    if (!fs.existsSync('./output')) {
      fs.mkdirSync('./output', { recursive: true });
    }
    fs.writeFileSync('./output/last_task_id.txt', taskId);
    
    return taskId;
  } catch (error) {
    console.error('Error submitting face swap task:', error.message);
    if (error.response) {
      console.error('Response data:', error.response.data);
      console.error('Response status:', error.response.status);
    }
    return null;
  }
}

// Parse command line arguments
const args = process.argv.slice(2);
const options = {
  useAce: true,
  useFft: true,
  loraStrength: 0.7,
  prompt: '',
  negativePrompt: ''
};

// Process command line arguments
for (let i = 0; i < args.length; i++) {
  const arg = args[i];
  
  if (arg === '--ace' || arg === '--use-ace') {
    options.useAce = true;
  } else if (arg === '--fft') {
    options.useFft = true;
    options.useAce = true; // FFT requires ACE to be enabled
  } else if (arg === '--strength' && i + 1 < args.length) {
    options.loraStrength = parseFloat(args[++i]);
  } else if (arg === '--guidance' && i + 1 < args.length) {
    options.guidanceScale = parseFloat(args[++i]);
  } else if (arg === '--steps' && i + 1 < args.length) {
    options.steps = parseInt(args[++i], 10);
  } else if (arg === '--prompt' && i + 1 < args.length) {
    options.prompt = args[++i];
  } else if (arg === '--negative' && i + 1 < args.length) {
    options.negativePrompt = args[++i];
  } else if (arg === '--seed' && i + 1 < args.length) {
    options.seed = parseInt(args[++i], 10);
  } else if (arg === '--help') {
    console.log(`
Usage: node call.js [options]

Options:
  --ace, --use-ace     Use ACE_Plus portrait model (default: false)
  --fft                Use ACE_Plus FFT model variant (implies --ace)
  --strength VALUE     Set LoRA strength (default: 0.7)
  --guidance VALUE     Set guidance scale (default: 7.5, ACE mode only)
  --steps VALUE        Set inference steps (default: 30, ACE mode only)
  --prompt "TEXT"      Set prompt text
  --negative "TEXT"    Set negative prompt text
  --seed VALUE         Set random seed (ACE mode only)
  --help               Show this help
    `);
    process.exit(0);
  }
}

// Determine model name
let modelName = "Standard InsightFace";
if (options.useAce) {
  modelName = options.useFft ? "ACE_Plus FFT" : "ACE_Plus Portrait";
}

// Run the swap
console.log(`Starting face swap with model: ${modelName}`);
if (options.useAce) {
  console.log('ACE_Plus parameters:');
  console.log(`  LoRA strength: ${options.loraStrength}`);
  console.log(`  Guidance scale: ${options.guidanceScale || 7.5}`);
  console.log(`  Steps: ${options.steps || 30}`);
  if (options.seed !== undefined) console.log(`  Seed: ${options.seed}`);
  
  if (options.useFft) {
    console.log(`\nNOTE: Using the ACE_Plus FFT variant which works best for detailed face enhancements.`);
    console.log(`The FFT model works in enhanced mode even if the LoRA file can't be loaded - using optimized parameters.`);
    console.log(`FFT enhancement uses higher guidance (8.5+) and more steps (35+) even without the model file.`);
    console.log(`If you encounter issues, try using the standard portrait model instead (without --fft flag).`);
  }
}
if (options.prompt) console.log(`Prompt: "${options.prompt}"`);
if (options.negativePrompt) console.log(`Negative prompt: "${options.negativePrompt}"`);

// Show full command used for reference
let commandUsed = "node call.js";
if (options.useAce) commandUsed += " --ace";
if (options.useFft) commandUsed += " --fft";
if (options.loraStrength !== 0.7) commandUsed += ` --strength ${options.loraStrength}`;
if (options.guidanceScale && options.guidanceScale !== 7.5) commandUsed += ` --guidance ${options.guidanceScale}`;
if (options.steps && options.steps !== 30) commandUsed += ` --steps ${options.steps}`;
if (options.seed !== undefined) commandUsed += ` --seed ${options.seed}`;
if (options.prompt) commandUsed += ` --prompt "${options.prompt}"`;
if (options.negativePrompt) commandUsed += ` --negative "${options.negativePrompt}"`;

console.log(`\nCommand used: ${commandUsed}`);

runAsyncSwap(options);