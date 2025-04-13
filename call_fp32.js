#!/usr/bin/env node
/**
 * Float32 wrapper for call.js to avoid CUDA errors
 * 
 * This script sets the appropriate environment variables to ensure
 * PyTorch uses float32 precision, which helps avoid the
 * "LayerNormKernelImpl not implemented for 'Half'" error.
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

// ANSI color codes for prettier output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[2m',
  yellow: '\x1b[33m',
  green: '\x1b[32m',
  blue: '\x1b[34m',
};

console.log(`${colors.green}=== ACE_Plus Face Swap with Float32 Precision ===${colors.reset}`);
console.log(`${colors.blue}This wrapper ensures PyTorch uses float32 precision to avoid CUDA errors${colors.reset}`);

// Create force_float32.py if it doesn't exist
const forceFloat32Path = path.join(__dirname, 'force_float32.py');
if (!fs.existsSync(forceFloat32Path)) {
  console.log(`${colors.yellow}Creating force_float32.py to ensure float32 precision...${colors.reset}`);
  const forceFloat32Content = `
# This file is used to force PyTorch to use float32 precision
# Import this file in your scripts before importing other libraries

import torch
torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
`;
  fs.writeFileSync(forceFloat32Path, forceFloat32Content);
}

// Get the original command line arguments, but remove the node and script name
const args = process.argv.slice(2);

// Set the environment variables
const env = {
  ...process.env,
  PYTORCH_CUDA_ALLOC_CONF: 'max_split_size_mb:32',
  PYTHONPATH: `${__dirname}:${process.env.PYTHONPATH || ''}`,
};

// Force --ace flag to be present
if (!args.includes('--ace') && !args.includes('--use-ace')) {
  args.push('--ace');
}

console.log(`${colors.green}Running with arguments:${colors.reset} ${args.join(' ')}`);

// Spawn the call.js process with the modified environment
const callJsPath = path.join(__dirname, 'call.js');
const child = spawn('node', [callJsPath, ...args], { 
  env, 
  stdio: 'inherit',
  shell: process.platform === 'win32' // Use shell on Windows
});

child.on('close', (code) => {
  if (code === 0) {
    console.log(`${colors.green}Face swap completed successfully.${colors.reset}`);
  } else {
    console.log(`${colors.yellow}Face swap exited with code ${code}.${colors.reset}`);
    console.log(`If you're still encountering errors, try running: python ${path.join(__dirname, 'fix_cuda_errors.py')}`);
  }
}); 