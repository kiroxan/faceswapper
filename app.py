from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os, shutil
from run_faceswap import run_faceswap

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define directories
UPLOAD_DIR = "./input"
OUTPUT_DIR = "./output"
DEBUG_DIR = "./debug"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "result.png")

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# Mount the output and debug directories as static files
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/debug", StaticFiles(directory=DEBUG_DIR), name="debug")

@app.post("/swap")
async def swap_faces(
    source: UploadFile = File(...),  # Source face image
    target: UploadFile = File(...),  # Target image where face will be swapped
    mask: UploadFile = File(None),   # Optional mask
    prompt: str = Form(""),          # Optional prompt
    negative_prompt: str = Form(""), # Optional negative prompt
    strength: float = Form(0.0)      # Not used in the basic implementation
):
    try:
        # Save uploaded files
        source_path = os.path.join(UPLOAD_DIR, "source.png")
        target_path = os.path.join(UPLOAD_DIR, "target.png")
        mask_path = None
        
        # Save source image
        with open(source_path, "wb") as f:
            shutil.copyfileobj(source.file, f)
        print(f"[INFO] Source file saved to {source_path}")
        
        # Save target image
        with open(target_path, "wb") as f:
            shutil.copyfileobj(target.file, f)
        print(f"[INFO] Target file saved to {target_path}")
        
        # Save mask if provided
        if mask:
            mask_path = os.path.join(UPLOAD_DIR, "mask.png")
            with open(mask_path, "wb") as f:
                shutil.copyfileobj(mask.file, f)
            print(f"[INFO] Mask file saved to {mask_path}")
        
        # Run the simple face swap
        print("[INFO] Running basic face swap...")
        run_faceswap(
            main_path=target_path,  # Target is the main image
            ref_path=source_path,   # Source is the reference face
            output_path=OUTPUT_PATH,
            mask_path=mask_path,
            prompt=prompt,
            negative_prompt=negative_prompt
        )
        
        if not os.path.exists(OUTPUT_PATH):
            return {"error": "Face swap failed to generate output image"}
            
        # Get the list of debug files
        debug_files = []
        if os.path.exists(DEBUG_DIR):
            debug_files = [f"/debug/{f}" for f in os.listdir(DEBUG_DIR) if f.endswith(('.png', '.jpg'))]
        print(f"[INFO] Found {len(debug_files)} debug files")
        
        # Return the file URLs and debug information
        return {
            "status": "success",
            "message": "Face swap completed successfully",
            "file_url": "/outputs/result.png",
            "direct_download_url": "/swap/download",
            "debug_files": debug_files
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] Face swap failed: {str(e)}")
        print(error_trace)
        return {
            "error": f"Face swap failed: {str(e)}",
            "traceback": error_trace,
            "source_info": source.filename if source else "No source file",
            "target_info": target.filename if target else "No target file",
        }

@app.get("/swap/download")
async def download_result():
    """Direct download endpoint for the latest result"""
    return FileResponse(OUTPUT_PATH, media_type="image/png")

@app.get("/")
async def read_root():
    """Root endpoint with basic information"""
    return {
        "app": "Face Swap API",
        "endpoints": {
            "POST /swap": "Upload images and perform face swap",
            "GET /outputs/result.png": "View the latest result",
            "GET /swap/download": "Download the latest result",
            "GET /debug/[filename]": "View debug/intermediate images"
        },
        "parameters": {
            "source": "Source face image (required)",
            "target": "Target image where face will be swapped (required)",
            "mask": "Optional mask for the face region",
            "prompt": "Optional text prompt",
            "negative_prompt": "Optional negative text prompt"
        }
    }
