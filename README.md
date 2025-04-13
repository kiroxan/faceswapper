# Face Swap API

A simple API for face swapping using InsightFace and GFPGAN for enhancement.

## Features

- Face detection and swap using InsightFace
- Optional face enhancement with GFPGAN
- Optional mask blending
- FastAPI web server with simple endpoints
- Debug image output at each step
- Asynchronous processing with UUID task tracking
- S3 storage integration for results

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- The following models in the `../models` directory:
  - `inswapper_128.onnx` (InsightFace face swap model)
  - `GFPGANv1.4.pth` (GFPGAN face enhancement model)
  - Buffalo-L model for InsightFace face detection
- AWS account with S3 bucket (for async mode)

## Setup

1. Ensure you have all required models in the `../models` directory

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure AWS credentials (for async mode):
   ```bash
   # Edit the placeholder values in setup_env.sh with your AWS credentials
   source setup_env.sh
   ```

4. Start the server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 7860 --reload
   ```

## Usage

### API Endpoints

#### Synchronous API (Immediate Response)
- **POST /swap**: Upload images and perform face swap (returns the image directly)

#### Asynchronous API (Background Processing)
- **POST /swap/async**: Submit a face swap job for background processing (returns a task ID)
- **GET /tasks/{task_id}**: Check the status of a specific task and get the result URL when ready
- **GET /tasks**: List recent tasks with optional status filtering

#### Debug Endpoints
- **GET /outputs/result.png**: View the latest result from synchronous swap
- **GET /debug/[filename]**: View debug/intermediate images

### Example using cURL

#### Synchronous Request
```bash
curl -X POST "http://localhost:7860/swap" \
  -F "main=@./path/to/target/image.jpg" \
  -F "ref=@./path/to/face/image.jpg" \
  -o result.png
```

#### Asynchronous Request
```bash
# Submit the swap job
TASK_ID=$(curl -X POST "http://localhost:7860/swap/async" \
  -F "main=@./path/to/target/image.jpg" \
  -F "ref=@./path/to/face/image.jpg" | jq -r '.task_id')

# Check task status
curl "http://localhost:7860/tasks/$TASK_ID"

# After task completes, download the result using the provided URL
curl -o result.png "$(curl "http://localhost:7860/tasks/$TASK_ID" | jq -r '.result_url')"
```

### Example using JavaScript

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

// Synchronous face swap (immediate result)
async function runSyncSwap() {
  const form = new FormData();
  form.append('main', fs.createReadStream('./test/target.png')); // Where to place the face
  form.append('ref', fs.createReadStream('./test/face.png'));    // Face to use
  
  // Optional: include mask if available
  if (fs.existsSync('./test/mask.png')) {
    form.append('mask', fs.createReadStream('./test/mask.png'));
  }
  
  try {
    const response = await axios.post(
      'http://localhost:7860/swap',
      form,
      {
        headers: form.getHeaders(),
        responseType: 'arraybuffer'
      }
    );
    
    fs.writeFileSync('./result.png', response.data);
    console.log('Face swap complete. Saved as result.png');
  } catch (error) {
    console.error('Error during face swap:', error);
  }
}

// Asynchronous face swap (background processing)
async function runAsyncSwap() {
  const form = new FormData();
  form.append('main', fs.createReadStream('./test/target.png')); // Where to place the face
  form.append('ref', fs.createReadStream('./test/face.png'));    // Face to use
  
  try {
    // Submit the task
    const submitResponse = await axios.post(
      'http://localhost:7860/swap/async',
      form,
      { headers: form.getHeaders() }
    );
    
    const taskId = submitResponse.data.task_id;
    console.log(`Task submitted with ID: ${taskId}`);
    
    // Poll for completion
    let completed = false;
    while (!completed) {
      console.log('Checking task status...');
      const statusResponse = await axios.get(`http://localhost:7860/tasks/${taskId}`);
      const status = statusResponse.data.status;
      
      console.log(`Task status: ${status}`);
      
      if (status === 'completed') {
        // Download the result
        const resultUrl = statusResponse.data.result_url;
        console.log(`Task completed! Result URL: ${resultUrl}`);
        
        const imageResponse = await axios.get(resultUrl, { responseType: 'arraybuffer' });
        fs.writeFileSync('./async_result.png', imageResponse.data);
        console.log('Result saved as async_result.png');
        completed = true;
      } else if (status === 'failed') {
        console.error('Task failed:', statusResponse.data.error);
        completed = true;
      } else {
        // Wait 2 seconds before checking again
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    }
  } catch (error) {
    console.error('Error during async face swap:', error);
  }
}

// Choose which function to run
runAsyncSwap();
```

## Directory Structure

- `app.py`: FastAPI web application
- `input/`: Directory for uploaded images
- `output/`: Directory for output images
- `debug/`: Directory for debug/intermediate images
- `setup_env.sh`: Script to set AWS environment variables

## S3 Storage Configuration

For the asynchronous API to work, you need to configure AWS S3:

1. Create an S3 bucket in your AWS account
2. Create an IAM user with the following permissions:
   - `s3:PutObject`
   - `s3:GetObject`
   - `s3:ListBucket`
3. Generate an access key and secret for this user
4. Update the `setup_env.sh` script with your credentials:
   ```bash
   export AWS_ACCESS_KEY_ID="your_access_key_id"
   export AWS_SECRET_ACCESS_KEY="your_secret_access_key"
   export AWS_REGION="us-east-1"  # Change to your preferred region
   export S3_BUCKET="your-faceswapper-bucket"  # Change to your bucket name
   ```
5. Source the script before running the server:
   ```bash
   source setup_env.sh
   uvicorn app:app --host 0.0.0.0 --port 7860
   ```

## Notes

- The source image should contain a clear face to use for swapping
- The target image contains the scene where the face will be placed
- If a mask is provided, it should be a grayscale image with white in the areas to blend
- For best results, use images with similar lighting conditions and face angles
- The asynchronous API is recommended for batch processing or integrating with other systems
- Task information is stored in memory and will be lost if the server restarts 