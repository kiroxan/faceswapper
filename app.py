import os
import cv2
import numpy as np
import uuid
import boto3
import json
import time
from datetime import datetime
from threading import Thread
from typing import Dict, Optional, List
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from gfpgan import GFPGANer
from pathlib import Path

# Import the ACE_Plus face swap function
from run_faceswap_ace import run_faceswap_ace, download_ace_plus_model

# --- Initialize Reusable Components ---
face_detector = FaceAnalysis(name="buffalo_l", root="../models")
face_detector.prepare(ctx_id=0)  # Use GPU if available, else CPU fallback

# Load the ReActor model
inswapper = get_model("../models/inswapper_128.onnx", providers=["CPUExecutionProvider"])

# Initialize GFPGAN (optional face enhancement)
try:
    gfpganer = GFPGANer(
        model_path="../models/GFPGANv1.4.pth",
        upscale=1,
        arch='clean',
        channel_multiplier=2
    )
except Exception as e:
    print(f"[WARN] GFPGAN initialization failed: {e}")
    gfpganer = None

# --- S3 Configuration ---
# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    region_name=os.environ.get('AWS_REGION', 'us-east-1')
)
S3_BUCKET = os.environ.get('S3_BUCKET')

# --- Task Management ---
# In-memory storage for task status
tasks: Dict[str, Dict] = {}

# --- Main Faceswap Function ---
def run_faceswap(main_path, ref_path, output_path, mask_path=None, prompt="", negative_prompt=""):
    """
    Simple face swap function using InsightFace
    
    Args:
        main_path: Path to the target image (where face will be swapped)
        ref_path: Path to the source image (with the face to use)
        output_path: Path to save the output image
        mask_path: Optional path to mask image
        prompt: Optional prompt (not used in this version)
        negative_prompt: Optional negative prompt (not used in this version)
        
    Returns:
        Path to the output image
    """
    print("[INFO] Starting face swap")
    print(f"[INFO] Main image: {main_path}")
    print(f"[INFO] Reference image: {ref_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load images
    main_img = cv2.imread(main_path)
    ref_img = cv2.imread(ref_path)
    
    if main_img is None:
        print(f"[ERROR] Could not load main image from {main_path}")
        return None
        
    if ref_img is None:
        print(f"[ERROR] Could not load reference image from {ref_path}")
        return None
        
    # Save copies of input images for debugging
    debug_dir = "./debug"
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, "input_main.png"), main_img)
    cv2.imwrite(os.path.join(debug_dir, "input_ref.png"), ref_img)
    
    mask_img = None
    if mask_path and os.path.exists(mask_path):
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        print(f"[INFO] Mask loaded from {mask_path}")
        cv2.imwrite(os.path.join(debug_dir, "input_mask.png"), mask_img)

    print("[INFO] Detecting faces...")
    main_faces = face_detector.get(main_img)
    ref_faces = face_detector.get(ref_img)

    if not main_faces:
        print("[ERROR] No faces detected in main image")
        cv2.imwrite(output_path, main_img)  # Save original as fallback
        return output_path
        
    if not ref_faces:
        print("[ERROR] No faces detected in reference image")
        cv2.imwrite(output_path, main_img)  # Save original as fallback
        return output_path

    # Use the first detected face from each image
    target_face = main_faces[0]
    source_face = ref_faces[0]

    # Perform face swap
    print("[INFO] Performing face swap...")
    result = main_img.copy()
    result = inswapper.get(result, target_face, source_face, paste_back=True)
    
    # Save intermediate result for debugging
    cv2.imwrite(os.path.join(debug_dir, "1_face_swapped.png"), result)
    
    # Apply mask blending if provided
    if mask_img is not None:
        try:
            print("[INFO] Blending using mask...")
            # Resize mask to match image dimensions if needed
            mask_resized = cv2.resize(mask_img, (main_img.shape[1], main_img.shape[0]))
            # Clean mask to ensure binary values
            mask_resized = cv2.threshold(mask_resized, 10, 255, cv2.THRESH_BINARY)[1]
            
            # Compute center of the face for seamless cloning
            x1, y1, x2, y2 = map(int, target_face.bbox)
            center_x = max(0, min(main_img.shape[1] - 1, (x1 + x2) // 2))
            center_y = max(0, min(main_img.shape[0] - 1, (y1 + y2) // 2))
            center = (center_x, center_y)
            
            try:
                # Try seamless cloning for better blending
                blended = cv2.seamlessClone(result, main_img, mask_resized, center, cv2.NORMAL_CLONE)
                result = blended
                cv2.imwrite(os.path.join(debug_dir, "2_mask_blended.png"), result)
            except cv2.error as e:
                print(f"[WARN] Seamless cloning failed: {e}, using simple alpha blending")
                # Fallback to simple alpha blending if seamless cloning fails
                normalized_mask = mask_resized.astype(float) / 255.0
                if len(normalized_mask.shape) == 2:
                    normalized_mask = normalized_mask[:, :, np.newaxis]
                result = (result * normalized_mask + main_img * (1 - normalized_mask)).astype(np.uint8)
                cv2.imwrite(os.path.join(debug_dir, "2_alpha_blended.png"), result)
        except Exception as e:
            print(f"[WARN] Mask blending failed: {e}")
    
    # Apply GFPGAN for face enhancement if available
    if gfpganer:
        try:
            print("[INFO] Enhancing with GFPGAN...")
            _, _, enhanced = gfpganer.enhance(
                result, 
                has_aligned=False, 
                only_center_face=False, 
                paste_back=True
            )
            if enhanced is not None:
                result = enhanced
                cv2.imwrite(os.path.join(debug_dir, "3_gfpgan_enhanced.png"), result)
            else:
                print("[WARN] GFPGAN returned None, using unenhanced result")
        except Exception as e:
            print(f"[WARN] GFPGAN enhancement failed: {e}")

    # Save the final result
    print(f"[INFO] Writing result to {output_path}")
    cv2.imwrite(output_path, result)
    return output_path

# --- Async Task Processing Function ---
def process_swap_task(task_id: str, main_path: str, ref_path: str, output_path: str, 
                     mask_path: Optional[str] = None, prompt: str = "", negative_prompt: str = "",
                     use_ace: bool = False, lora_strength: float = 0.7, guidance_scale: float = 7.5,
                     num_inference_steps: int = 30, seed: Optional[int] = None, use_fft: bool = False):
    """Process a face swap task asynchronously and upload result to S3"""
    try:
        # Update task status
        tasks[task_id]['status'] = 'processing'
        
        # Choose which face swap method to use
        if use_ace:
            # Use ACE_Plus portrait enhancement
            model_name = "ACE_Plus FFT" if use_fft else "ACE_Plus Portrait"
            print(f"[INFO] Using {model_name} model for task {task_id}")
            
            # Check if required model exists
            lora_path = "../models/ace_plus_fft.safetensors" if use_fft else "../models/comfyui_portrait_lora64.safetensors"
            if not os.path.exists(lora_path):
                # Try downloading the model
                download_result = download_ace_plus_model(use_fft=use_fft)
                if not download_result:
                    tasks[task_id]['status'] = 'failed'
                    tasks[task_id]['error'] = f"{model_name} model not available"
                    return
            
            # Run ACE_Plus face swap
            result_path = run_faceswap_ace(
                main_path, ref_path, output_path, mask_path,
                prompt, negative_prompt, lora_strength=lora_strength,
                guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                seed=seed, use_fft=use_fft
            )
        else:
            # Use standard face swap
            print(f"[INFO] Using standard InsightFace model for task {task_id}")
            result_path = run_faceswap(main_path, ref_path, output_path, mask_path, prompt, negative_prompt)
        
        if result_path and os.path.exists(result_path):
            # Upload result to S3
            s3_key = f"results/{task_id}/{os.path.basename(result_path)}"
            s3_client.upload_file(result_path, S3_BUCKET, s3_key)
            
            # Generate presigned URL for the result (valid for 1 hour)
            presigned_url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': S3_BUCKET, 'Key': s3_key},
                ExpiresIn=3600
            )
            
            # Update task info
            tasks[task_id]['status'] = 'completed'
            tasks[task_id]['result_url'] = presigned_url
            tasks[task_id]['s3_key'] = s3_key
            tasks[task_id]['end_time'] = datetime.now().isoformat()
        else:
            # Handle failure
            tasks[task_id]['status'] = 'failed'
            tasks[task_id]['error'] = 'Face swap processing failed'
            tasks[task_id]['end_time'] = datetime.now().isoformat()
    
    except Exception as e:
        # Handle exceptions
        print(f"[ERROR] Task {task_id} failed: {str(e)}")
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['error'] = str(e)
        tasks[task_id]['end_time'] = datetime.now().isoformat()

# --- FastAPI Setup ---
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import shutil

app = FastAPI()
UPLOAD_DIR = "./input"
OUTPUT_DIR = "./output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/swap")
async def swap_faces(
    main: UploadFile = File(...),
    ref: UploadFile = File(...),
    mask: UploadFile = File(None),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    prompt_strength: float = Form(0.25),  # Default prompt strength of 0.25 (lower = more preservation)
    use_ace: bool = Form(False)  # Whether to use ACE_Plus portrait model
):
    print(f"[INFO] Received request with prompt: '{prompt}', strength: {prompt_strength}, use_ace: {use_ace}")
    main_path = os.path.join(UPLOAD_DIR, "main.png")
    ref_path = os.path.join(UPLOAD_DIR, "ref.png")
    mask_path = os.path.join(UPLOAD_DIR, "mask.png") if mask else None

    # Save the uploaded files
    for file, dest in [(main, main_path), (ref, ref_path)]:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

    if mask:
        with open(mask_path, "wb") as f:
            shutil.copyfileobj(mask.file, f)

    # Run the appropriate faceswap pipeline
    if use_ace:
        # Check if ACE_Plus model is available or try to download it
        if not os.path.exists("../models/ace_plus/comfyui_portrait_lora64.safetensors"):
            # Try downloading the model
            download_result = download_ace_plus_model()
            if not download_result:
                return {"error": "ACE_Plus model not available. Please download it manually."}
        
        # Run ACE_Plus face swap
        result_path = run_faceswap_ace(
            main_path, ref_path, OUTPUT_DIR + "/result.png", mask_path, 
            prompt, negative_prompt, lora_strength=prompt_strength
        )
    else:
        # Run standard face swap
        result_path = run_faceswap(main_path, ref_path, OUTPUT_DIR + "/result.png", mask_path, prompt, negative_prompt)
    
    if result_path and os.path.exists(result_path):
        return FileResponse(result_path, media_type="image/png")
    else:
        return {"error": "Face swap failed"}

@app.post("/swap/async")
async def async_swap_faces(
    background_tasks: BackgroundTasks,
    main: UploadFile = File(...),
    ref: UploadFile = File(...),
    mask: UploadFile = File(None),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    prompt_strength: float = Form(0.25),
    use_ace: bool = Form(False),  # Whether to use ACE_Plus portrait model
    use_fft: bool = Form(False),  # Whether to use ACE_Plus FFT model instead of portrait
    lora_strength: float = Form(0.7),  # ACE_Plus specific: strength of LoRA model
    guidance_scale: float = Form(7.5),  # ACE_Plus specific: guidance scale
    num_inference_steps: int = Form(30),  # ACE_Plus specific: number of sampling steps
    seed: Optional[int] = Form(None)  # ACE_Plus specific: random seed
):
    """Start an asynchronous face swap task and return a task ID"""
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    
    # Create task-specific directories
    task_input_dir = os.path.join(UPLOAD_DIR, task_id)
    task_output_dir = os.path.join(OUTPUT_DIR, task_id)
    os.makedirs(task_input_dir, exist_ok=True)
    os.makedirs(task_output_dir, exist_ok=True)
    
    # Define file paths
    main_path = os.path.join(task_input_dir, "main.png")
    ref_path = os.path.join(task_input_dir, "ref.png")
    mask_path = os.path.join(task_input_dir, "mask.png") if mask else None
    output_path = os.path.join(task_output_dir, "result.png")
    
    # Save the uploaded files
    for file, dest in [(main, main_path), (ref, ref_path)]:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

    if mask:
        with open(mask_path, "wb") as f:
            shutil.copyfileobj(mask.file, f)
    
    # Initialize task info
    tasks[task_id] = {
        'id': task_id,
        'status': 'queued',
        'created': datetime.now().isoformat(),
        'input_paths': {
            'main': main_path,
            'ref': ref_path,
            'mask': mask_path
        },
        'output_path': output_path,
        'parameters': {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'prompt_strength': prompt_strength,
            'use_ace': use_ace,
            'use_fft': use_fft,
            'lora_strength': lora_strength,
            'guidance_scale': guidance_scale,
            'num_inference_steps': num_inference_steps,
            'seed': seed
        }
    }
    
    # If using ACE_Plus, check if model exists or try to download it
    if use_ace:
        lora_path = "../models/ace_plus_fft.safetensors" if use_fft else "../models/comfyui_portrait_lora64.safetensors"
        if not os.path.exists(lora_path):
            # Start a background task to download the model
            Thread(target=download_ace_plus_model, args=(use_fft,)).start()
            # Note: We'll let the swap process handle the case if download fails
    
    # Start processing in the background
    # Use a Thread for simplicity - for production use consider a proper task queue like Celery
    task_thread = Thread(
        target=process_swap_task,
        args=(
            task_id, main_path, ref_path, output_path, mask_path, 
            prompt, negative_prompt, use_ace, lora_strength, guidance_scale,
            num_inference_steps, seed, use_fft
        )
    )
    task_thread.start()
    
    # Determine model name for response
    model_name = "Standard InsightFace"
    if use_ace:
        model_name = "ACE_Plus FFT" if use_fft else "ACE_Plus Portrait"
    
    return JSONResponse({
        "task_id": task_id,
        "status": "queued",
        "message": "Face swap task has been queued",
        "model": model_name
    })

@app.post("/swap/ace/async")
async def async_ace_portrait_swap(
    background_tasks: BackgroundTasks,
    main: UploadFile = File(...),
    ref: UploadFile = File(...),
    mask: UploadFile = File(None),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    lora_strength: float = Form(0.7),
    guidance_scale: float = Form(7.5),
    num_inference_steps: int = Form(30),
    seed: Optional[int] = Form(None)
):
    """Dedicated endpoint for ACE_Plus portrait model face swap"""
    # This is just a convenience wrapper around the async_swap_faces endpoint
    return await async_swap_faces(
        background_tasks=background_tasks,
        main=main,
        ref=ref,
        mask=mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        prompt_strength=lora_strength,  # Use lora_strength as prompt_strength
        use_ace=True,
        use_fft=False,
        lora_strength=lora_strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed
    )

@app.post("/swap/ace/fft/async")
async def async_ace_fft_swap(
    background_tasks: BackgroundTasks,
    main: UploadFile = File(...),
    ref: UploadFile = File(...),
    mask: UploadFile = File(None),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    lora_strength: float = Form(0.7),
    guidance_scale: float = Form(7.5),
    num_inference_steps: int = Form(30),
    seed: Optional[int] = Form(None)
):
    """Dedicated endpoint for ACE_Plus FFT model face swap"""
    # This is just a convenience wrapper around the async_swap_faces endpoint
    return await async_swap_faces(
        background_tasks=background_tasks,
        main=main,
        ref=ref,
        mask=mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        prompt_strength=lora_strength,  # Use lora_strength as prompt_strength
        use_ace=True,
        use_fft=True,
        lora_strength=lora_strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed
    )

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of an async face swap task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    task_info = tasks[task_id].copy()
    
    # If the task is completed but the URL has expired, generate a new one
    if task_info.get('status') == 'completed' and 's3_key' in task_info:
        # Generate a new presigned URL valid for 1 hour
        try:
            presigned_url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': S3_BUCKET, 'Key': task_info['s3_key']},
                ExpiresIn=3600
            )
            task_info['result_url'] = presigned_url
        except Exception as e:
            print(f"[WARN] Failed to generate presigned URL: {str(e)}")
    
    # Don't return internal file paths
    if 'input_paths' in task_info:
        del task_info['input_paths']
    if 'output_path' in task_info:
        del task_info['output_path']
    
    return JSONResponse(task_info)

@app.get("/tasks")
async def list_tasks(limit: int = 10, status: Optional[str] = None):
    """List recent tasks with optional status filter"""
    filtered_tasks = []
    
    for task_id, task_info in sorted(
        tasks.items(), 
        key=lambda x: x[1].get('created', ''), 
        reverse=True
    ):
        # Apply status filter if provided
        if status and task_info.get('status') != status:
            continue
            
        # Create a copy without internal file paths
        task_copy = task_info.copy()
        if 'input_paths' in task_copy:
            del task_copy['input_paths']
        if 'output_path' in task_copy:
            del task_copy['output_path']
        
        filtered_tasks.append(task_copy)
        
        # Apply limit
        if len(filtered_tasks) >= limit:
            break
    
    return JSONResponse({"tasks": filtered_tasks})
