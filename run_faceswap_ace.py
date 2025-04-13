import os
import cv2
import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from safetensors.torch import load_file
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from gfpgan import GFPGANer
from PIL import Image
from pathlib import Path

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

# Load ACE_Plus LoRA model paths
ACE_LORA_PATH = "../models/ace_plus/comfyui_portrait_lora64.safetensors"
ACE_FFT_LORA_PATH = "../models/ace_plus/ace_plus_fft.safetensors"
DEFAULT_MODEL_PATH = "runwayml/stable-diffusion-v1-5"

# Check safetensors and diffusers versions
try:
    import pkg_resources
    safetensors_version = pkg_resources.get_distribution("safetensors").version
    diffusers_version = pkg_resources.get_distribution("diffusers").version
    print(f"[INFO] Using safetensors version: {safetensors_version}")
    print(f"[INFO] Using diffusers version: {diffusers_version}")
except Exception as e:
    print(f"[WARN] Could not determine package versions: {e}")

# Lazily initialize StableDiffusion pipeline on first use
sd_pipeline = None
# Flag to track if we've already tried and failed to load LoRA
lora_load_failed = False

def get_sd_pipeline(simple_mode=False, use_fft=False):
    """
    Initialize and return the Stable Diffusion Img2Img pipeline
    
    Args:
        simple_mode: If True, skip LoRA loading entirely
        use_fft: If True, use the FFT variant of ACE_Plus
    """
    global sd_pipeline, lora_load_failed
    
    # Return existing pipeline if already initialized
    if sd_pipeline is not None:
        return sd_pipeline
    
    print("[INFO] Loading Stable Diffusion Img2Img pipeline...")
    
    # Select the appropriate LoRA path
    lora_path = ACE_FFT_LORA_PATH if use_fft else ACE_LORA_PATH
    model_variant = "ACE_Plus FFT" if use_fft else "ACE_Plus Portrait"
    
    try:
        # Create Stable Diffusion Img2Img pipeline
        scheduler = DDIMScheduler.from_pretrained(DEFAULT_MODEL_PATH, subfolder="scheduler")
        sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            DEFAULT_MODEL_PATH,
            scheduler=scheduler,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        # Skip LoRA loading if in simple mode or if we've already failed
        if simple_mode or lora_load_failed:
            print("[INFO] Running in simple mode without LoRA")
        else:
            # Check if LoRA file exists
            if not os.path.exists(lora_path):
                print(f"[ERROR] {model_variant} LoRA model not found at {lora_path}")
                print(f"[INFO] Please download it to {lora_path}")
                if use_fft and os.path.exists(ACE_LORA_PATH):
                    print("[INFO] Falling back to standard ACE_Plus Portrait model")
                    lora_path = ACE_LORA_PATH
                    model_variant = "ACE_Plus Portrait (fallback)"
                else:
                    return None
            
            # Attempt to apply the LoRA weights
            try:
                # Create a proper diffusers-format LoRA directory structure
                print(f"[INFO] Setting up {model_variant} LoRA folder structure...")
                import shutil
                import tempfile
                
                # Create a temporary directory with proper structure
                with tempfile.TemporaryDirectory() as temp_dir:
                    lora_temp_path = os.path.join(temp_dir, "pytorch_lora_weights.safetensors")
                    
                    # Copy the LoRA file to the temp directory with the expected name
                    shutil.copy(lora_path, lora_temp_path)
                    
                    # Try to use the diffusers built-in LoRA loading method
                    try:
                        print(f"[INFO] Loading {model_variant} LoRA weights from {temp_dir}...")
                        
                        # For newer diffusers versions
                        sd_pipeline.load_lora_weights(temp_dir)
                        print(f"[INFO] Successfully loaded {model_variant} LoRA weights with diffusers method")
                    except (AttributeError, ImportError, RuntimeError) as e:
                        lora_load_failed = True
                        print(f"[WARN] Could not load LoRA with standard method: {e}")
                        print("[INFO] Will continue without LoRA weights")
                
            except Exception as e:
                lora_load_failed = True
                print(f"[WARN] Failed to set up or load LoRA weights: {e}")
                print("[INFO] Will continue without LoRA weights")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            sd_pipeline = sd_pipeline.to("cuda")
            print("[INFO] Using GPU for inference")
        else:
            print("[WARN] GPU not available, using CPU (this will be very slow)")
        
        # Don't return None - this lets us try SD without LoRA
        return sd_pipeline
        
    except Exception as e:
        print(f"[ERROR] Failed to load StableDiffusion pipeline: {e}")
        return None

# --- ACE_Plus Faceswap Function ---
def run_faceswap_ace(main_path, ref_path, output_path, mask_path=None, 
                     prompt="", negative_prompt="", lora_strength=0.7,
                     guidance_scale=7.5, num_inference_steps=30,
                     seed=None, use_initial_face_swap=True, use_fft=False):
    """
    Face swap function using ACE_Plus portrait LoRA model
    
    Args:
        main_path: Path to the target image (where face will be swapped)
        ref_path: Path to the source image (with the face to use)
        output_path: Path to save the output image
        mask_path: Optional path to mask image
        prompt: Additional text prompt
        negative_prompt: Negative text prompt
        lora_strength: Strength of the LoRA model (0-1)
        guidance_scale: Classifier-free guidance scale
        num_inference_steps: Number of denoising steps
        seed: Random seed for reproducibility
        use_initial_face_swap: Whether to use InsightFace for initial face swap
        use_fft: Whether to use ACE_Plus FFT model instead of portrait model
        
    Returns:
        Path to the output image
    """
    model_name = "ACE_Plus FFT" if use_fft else "ACE_Plus Portrait"
    print(f"[INFO] Starting {model_name} face swap")
    print(f"[INFO] Main image: {main_path}")
    print(f"[INFO] Reference image: {ref_path}")
    print(f"[INFO] LoRA strength: {lora_strength}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create debug directory
    debug_dir = "./debug"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Set flag to track if we need to fall back to initial swap only
    fallback_to_initial_swap = False

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
    cv2.imwrite(os.path.join(debug_dir, "input_main.png"), main_img)
    cv2.imwrite(os.path.join(debug_dir, "input_ref.png"), ref_img)
    
    mask_img = None
    if mask_path and os.path.exists(mask_path):
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        print(f"[INFO] Mask loaded from {mask_path}")
        cv2.imwrite(os.path.join(debug_dir, "input_mask.png"), mask_img)

    # Use InsightFace for first-stage face swap if enabled
    if use_initial_face_swap:
        print("[INFO] Performing initial face swap with InsightFace...")
        
        # Detect faces
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
        print("[INFO] Swapping faces with InsightFace...")
        initial_result = main_img.copy()
        initial_result = inswapper.get(initial_result, target_face, source_face, paste_back=True)
        
        # Save intermediate result for debugging
        cv2.imwrite(os.path.join(debug_dir, "1_initial_face_swapped.png"), initial_result)
        
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
                    blended = cv2.seamlessClone(initial_result, main_img, mask_resized, center, cv2.NORMAL_CLONE)
                    initial_result = blended
                    cv2.imwrite(os.path.join(debug_dir, "2_mask_blended.png"), initial_result)
                except cv2.error as e:
                    print(f"[WARN] Seamless cloning failed: {e}, using simple alpha blending")
                    # Fallback to simple alpha blending if seamless cloning fails
                    normalized_mask = mask_resized.astype(float) / 255.0
                    if len(normalized_mask.shape) == 2:
                        normalized_mask = normalized_mask[:, :, np.newaxis]
                    initial_result = (initial_result * normalized_mask + main_img * (1 - normalized_mask)).astype(np.uint8)
                    cv2.imwrite(os.path.join(debug_dir, "2_alpha_blended.png"), initial_result)
            except Exception as e:
                print(f"[WARN] Mask blending failed: {e}")
        
        # Use the face swapped result as input for ACE_Plus
        input_for_ace = initial_result
    else:
        # Use the original main image as input
        input_for_ace = main_img
    
    # Try to initialize StableDiffusion with ACE_Plus LoRA
    try:
        # First try with LoRA
        pipeline = get_sd_pipeline(simple_mode=False, use_fft=use_fft)
        if pipeline is None:
            print("[INFO] Trying again without LoRA...")
            # Try again in simple mode (no LoRA)
            pipeline = get_sd_pipeline(simple_mode=True, use_fft=use_fft)
            if pipeline is None:
                print("[ERROR] Could not initialize StableDiffusion pipeline")
                fallback_to_initial_swap = True
    except Exception as e:
        print(f"[ERROR] Failed to initialize Stable Diffusion pipeline: {e}")
        fallback_to_initial_swap = True
    
    # Check if we need to fall back to just the initial swap
    if fallback_to_initial_swap:
        print("[INFO] Using only the initial face swap result")
        if use_initial_face_swap:
            # Save final output
            print(f"[INFO] Writing result to {output_path}")
            cv2.imwrite(output_path, input_for_ace)
            return output_path
        else:
            # If no initial swap was done, just use the original image
            print(f"[INFO] Writing original image to {output_path}")
            cv2.imwrite(output_path, main_img)
            return output_path
        
    # Convert BGR to RGB for StableDiffusion
    input_for_ace_rgb = cv2.cvtColor(input_for_ace, cv2.COLOR_BGR2RGB)
    
    # Create PIL image
    input_pil = Image.fromarray(input_for_ace_rgb)
    
    # Set default prompt if not provided or empty
    if not prompt or prompt.strip() == "":
        prompt = "a portrait photo of person, highly detailed face, clear eyes, perfect face"
    
    # Set default negative prompt if not provided
    if not negative_prompt or negative_prompt.strip() == "":
        negative_prompt = "blurry, low quality, disfigured face, bad eyes, bad nose, bad ears, bad mouth, bad teeth"
    
    print(f"[INFO] Using prompt: '{prompt}'")
    print(f"[INFO] Using negative prompt: '{negative_prompt}'")
    
    # Set random seed if provided
    if seed is not None:
        generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    else:
        generator = None
    
    # Run ACE_Plus inference
    print("[INFO] Running ACE_Plus portrait enhancement...")
    try:
        with torch.inference_mode():
            try:
                print("[INFO] Using pipeline parameters: guidance_scale={}, steps={}, strength={}".format(
                    guidance_scale, num_inference_steps, lora_strength
                ))
                
                # Try the standard API call first
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=input_pil,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    strength=lora_strength,
                ).images[0]
                
            except TypeError as type_err:
                # Check if there's a specific parameter issue
                print(f"[WARN] Parameter error in pipeline call: {type_err}")
                print("[INFO] Trying alternative parameter set...")
                
                if "got an unexpected keyword argument 'strength'" in str(type_err):
                    # Try without strength parameter (some versions don't use it)
                    result = pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=input_pil,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images[0]
                else:
                    # Let the error propagate to outer try/catch
                    raise
        
        # Save intermediate result
        result.save(os.path.join(debug_dir, "3_ace_enhanced.png"))
        
        # Convert back to OpenCV format
        result_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[ERROR] ACE_Plus enhancement failed: {e}")
        print("[INFO] Falling back to the initial face swap result")
        
        # Print diffusers and CUDA details for diagnostics
        try:
            print(f"[DEBUG] Diffusers version: {pkg_resources.get_distribution('diffusers').version}")
            print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"[DEBUG] CUDA device: {torch.cuda.get_device_name()}")
                print(f"[DEBUG] CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        except:
            print("[DEBUG] Could not get detailed diagnostics")
            
        # Use the initial face swap result or original image as fallback
        result_cv = input_for_ace
        cv2.imwrite(os.path.join(debug_dir, "3_ace_failed.png"), result_cv)
    
    # Apply GFPGAN for optional face enhancement
    if gfpganer:
        try:
            print("[INFO] Enhancing with GFPGAN...")
            _, _, enhanced = gfpganer.enhance(
                result_cv, 
                has_aligned=False, 
                only_center_face=False, 
                paste_back=True
            )
            if enhanced is not None:
                result_cv = enhanced
                cv2.imwrite(os.path.join(debug_dir, "4_gfpgan_enhanced.png"), result_cv)
            else:
                print("[WARN] GFPGAN returned None, using unenhanced result")
        except Exception as e:
            print(f"[WARN] GFPGAN enhancement failed: {e}")

    # Save the final result
    print(f"[INFO] Writing result to {output_path}")
    cv2.imwrite(output_path, result_cv)
    return output_path

# Helper function to download the ACE_Plus model if needed
def download_ace_plus_model(use_fft=False):
    """
    Download the ACE_Plus LoRA model if it doesn't exist
    
    Args:
        use_fft: If True, download the FFT variant of ACE_Plus
    """
    lora_path = ACE_FFT_LORA_PATH if use_fft else ACE_LORA_PATH
    model_name = "ACE_Plus FFT" if use_fft else "ACE_Plus Portrait"
    
    if os.path.exists(lora_path):
        print(f"[INFO] {model_name} LoRA model already exists at {lora_path}")
        return True
    
    try:
        import requests
        from tqdm import tqdm
        
        print(f"[INFO] {model_name} LoRA model not found. Downloading...")
        os.makedirs(os.path.dirname(lora_path), exist_ok=True)
        
        # URL for the model
        if use_fft:
            # FFT model URL - update this with the correct URL when available
            model_url = "https://huggingface.co/ali-vilab/ACE_Plus/resolve/main/fft/ace_plus_fft.safetensors"
        else:
            # Portrait model URL
            model_url = "https://huggingface.co/ali-vilab/ACE_Plus/resolve/main/portrait/comfyui_portrait_lora64.safetensors"
        
        # Stream the download with progress bar
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(lora_path, 'wb') as f, tqdm(
                desc=f"Downloading {model_name} LoRA",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
        
        print(f"[INFO] Downloaded {model_name} LoRA model to {lora_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to download {model_name} LoRA model: {e}")
        print(f"[INFO] Please download it manually to {lora_path}")
        return False 