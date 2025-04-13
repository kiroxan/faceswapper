# Face Swap API

A simple API for face swapping using InsightFace and GFPGAN for enhancement.

## Features

- Face detection and swap using InsightFace
- Optional face enhancement with GFPGAN
- Optional mask blending
- FastAPI web server with simple endpoints
- Debug image output at each step

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- The following models in the `../models` directory:
  - `inswapper_128.onnx` (InsightFace face swap model)
  - `GFPGANv1.4.pth` (GFPGAN face enhancement model)
  - Buffalo-L model for InsightFace face detection

## Setup

1. Ensure you have all required models in the `../models` directory

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 7860 --reload
   ```

## Usage

### API Endpoints

- **POST /swap**: Upload images and perform face swap
- **GET /outputs/result.png**: View the latest result
- **GET /swap/download**: Download the latest result
- **GET /debug/[filename]**: View debug/intermediate images

### Example using cURL

```bash
curl -X POST "http://localhost:7860/swap" \
  -F "source=@./path/to/face/image.jpg" \
  -F "target=@./path/to/target/image.jpg" \
  -o result.png
```

### Example using JavaScript

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

async function runSwap() {
  const form = new FormData();
  form.append('source', fs.createReadStream('./test/face.png'));  // Face to use
  form.append('target', fs.createReadStream('./test/target.png')); // Where to place the face
  
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

runSwap();
```

## Directory Structure

- `app.py`: FastAPI web application
- `run_faceswap.py`: Core face swapping implementation
- `input/`: Directory for uploaded images
- `output/`: Directory for output images
- `debug/`: Directory for debug/intermediate images

## Notes

- The source image should contain a clear face to use for swapping
- The target image contains the scene where the face will be placed
- If a mask is provided, it should be a grayscale image with white in the areas to blend
- For best results, use images with similar lighting conditions and face angles 