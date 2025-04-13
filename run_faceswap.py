import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from gfpgan import GFPGANer
from pathlib import Path

# --- Initialize Reusable Components ---
face_detector = FaceAnalysis(name="buffalo_l", root="./models")
face_detector.prepare(ctx_id=0)  # Use GPU if available, else CPU fallback

# Load the ReActor model
inswapper = get_model("./models/inswapper_128.onnx", providers=["CPUExecutionProvider"])

# Initialize GFPGAN (optional face enhancement)
try:
    gfpganer = GFPGANer(
        model_path="./models/GFPGANv1.4.pth",
        upscale=1,
        arch='clean',
        channel_multiplier=2
    )
except Exception as e:
    print(f"[WARN] GFPGAN initialization failed: {e}")
    gfpganer = None


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


# --- FastAPI Setup ---
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import shutil

app = FastAPI()
UPLOAD_DIR = "./input"
OUTPUT_PATH = "./output/result.png"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/swap")
async def swap_faces(
    main: UploadFile = File(...),
    ref: UploadFile = File(...),
    mask: UploadFile = File(None),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    prompt_strength: float = Form(0.25)  # Default prompt strength of 0.25 (lower = more preservation)
):
    print(f"[INFO] Received request with prompt: '{prompt}', strength: {prompt_strength}")
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

    # Run the faceswap pipeline
    result_path = run_faceswap(main_path, ref_path, OUTPUT_PATH, mask_path, prompt, negative_prompt)
    
    if result_path and os.path.exists(result_path):
        return FileResponse(result_path, media_type="image/png")
    else:
        return {"error": "Face swap failed"}
