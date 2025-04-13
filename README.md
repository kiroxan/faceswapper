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
- **NEW**: ACE_Plus portrait LoRA model integration for high-quality face refinement

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- The following models in the `../models` directory:
  - `inswapper_128.onnx` (InsightFace face swap model)
  - `GFPGANv1.4.pth` (GFPGAN face enhancement model)
  - Buffalo-L model for InsightFace face detection
  - For ACE_Plus: `ace_plus/comfyui_portrait_lora64.safetensors` (downloaded automatically if missing)
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

4. Download ACE_Plus portrait model (optional):
   ```bash
   python download_ace_model.py
   ```

5. Start the server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 7860 --reload
   ```

## Model Options

### Standard InsightFace Model

The standard face swap uses InsightFace for face detection and swapping. This is a fast and reliable method that works well for most cases.

### ACE_Plus Portrait Model

The ACE_Plus portrait model enhances results by using a Stable Diffusion LoRA model specialized for high-quality portraits. This option:

- Provides more realistic and higher quality results
- Refines facial details using generative AI technology
- Takes longer to process but delivers superior output quality
- Works best with portrait photos

The ACE_Plus model is available on [Hugging Face](https://huggingface.co/ali-vilab/ACE_Plus/tree/main/portrait) and will be downloaded automatically on first use.

## Usage

### API Endpoints

#### Synchronous API (Immediate Response)
- **POST /swap**: Upload images and perform face swap (returns the image directly)
  - Parameter `use_ace`: Set to `true` to use ACE_Plus portrait model

#### Asynchronous API (Background Processing)
- **POST /swap/async**: Submit a face swap job for background processing (returns a task ID)
  - Parameter `use_ace`: Set to `true` to use ACE_Plus portrait model
  - ACE-specific parameters:
    - `lora_strength`: Strength of the ACE_Plus model effect (0-1, default: 0.7)
    - `guidance_scale`: Classifier-free guidance scale (default: 7.5)
    - `num_inference_steps`: Number of denoising steps (default: 30)
    - `seed`: Optional random seed for reproducibility
- **POST /swap/ace/async**: Dedicated endpoint for ACE_Plus portrait model face swap (same parameters as above)
- **GET /tasks/{task_id}**: Check the status of a specific task and get the result URL when ready
- **GET /tasks**: List recent tasks with optional status filtering

#### Debug Endpoints
- **GET /outputs/result.png**: View the latest result from synchronous swap
- **GET /debug/[filename]**: View debug/intermediate images

### Example using cURL

#### Synchronous Request
```bash
# Standard InsightFace model
curl -X POST "http://localhost:7860/swap" \
  -F "main=@./path/to/target/image.jpg" \
  -F "ref=@./path/to/face/image.jpg" \
  -o result.png

# Using ACE_Plus portrait model
curl -X POST "http://localhost:7860/swap" \
  -F "main=@./path/to/target/image.jpg" \
  -F "ref=@./path/to/face/image.jpg" \
  -F "use_ace=true" \
  -F "lora_strength=0.7" \
  -o result_ace.png
```

#### Asynchronous Request
```bash
# Standard InsightFace model
TASK_ID=$(curl -X POST "http://localhost:7860/swap/async" \
  -F "main=@./path/to/target/image.jpg" \
  -F "ref=@./path/to/face/image.jpg" | jq -r '.task_id')

# Using ACE_Plus portrait model
TASK_ID=$(curl -X POST "http://localhost:7860/swap/ace/async" \
  -F "main=@./path/to/target/image.jpg" \
  -F "ref=@./path/to/face/image.jpg" \
  -F "lora_strength=0.7" \
  -F "guidance_scale=7.5" \
  -F "num_inference_steps=30" | jq -r '.task_id')

# Check task status
curl "http://localhost:7860/tasks/$TASK_ID"

# After task completes, download the result using the provided URL
curl -o result.png "$(curl "http://localhost:7860/tasks/$TASK_ID" | jq -r '.result_url')"
```

### Using the JavaScript Client

We provide several JavaScript clients to interact with the API:

1. **call.js**: Submit an async face swap task
   ```bash
   # Standard InsightFace model
   node call.js
   
   # ACE_Plus portrait model
   node call.js --ace --strength 0.7 --guidance 7.5 --steps 30 --prompt "portrait photo with perfect face"
   ```

2. **get_result.js**: Check status and download result
   ```bash
   node get_result.js <task_id>
   ```

3. **check_tasks.js**: List and monitor tasks
   ```bash
   node check_tasks.js list
   node check_tasks.js get <task_id>
   ```

4. **batch_swap.js**: Process multiple images
   ```bash
   node batch_swap.js submit
   ```

## Directory Structure

- `app.py`: FastAPI web application
- `run_faceswap_ace.py`: ACE_Plus portrait model implementation
- `input/`: Directory for uploaded images
- `output/`: Directory for output images
- `debug/`: Directory for debug/intermediate images
- `setup_env.sh`: Script to set AWS environment variables
- `download_ace_model.py`: Script to download the ACE_Plus portrait model

## Comparison of Swap Methods

| Feature | Standard InsightFace | ACE_Plus Portrait |
|---------|---------------------|------------------|
| Speed | Fast (seconds) | Slower (30+ seconds) |
| Quality | Good | Excellent |
| GPU Requirements | Low-Medium | High |
| Best for | Quick swaps, non-portrait images | High-quality portraits |
| Customizable | Limited | Extensive (prompts, guidance, steps) |

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
- The ACE_Plus portrait model works best with frontal face portraits and may take significantly longer to process 