import os
import requests
from tqdm import tqdm
import sys

def download_ace_plus_model():
    """
    Download the ACE_Plus portrait LoRA model from Hugging Face
    """
    # Define model paths
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/ace_plus")
    model_path = os.path.join(model_dir, "comfyui_portrait_lora64.safetensors")
    
    # Create directory if it doesn't exist
    if not os.path.exists(model_dir):
        print(f"Creating directory: {model_dir}")
        os.makedirs(model_dir, exist_ok=True)
    
    # Check if model already exists
    if os.path.exists(model_path):
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model already exists at {model_path} ({file_size_mb:.2f} MB)")
        print("Delete this file if you want to download it again.")
        return True
    
    # URL for the model
    model_url = "https://huggingface.co/ali-vilab/ACE_Plus/resolve/main/portrait/comfyui_portrait_lora64.safetensors"
    
    print(f"Downloading ACE_Plus portrait LoRA from {model_url}")
    print(f"Saving to {model_path}")
    print("This file is ~613MB, it may take some time depending on your connection speed.")
    
    try:
        # Stream the download with progress bar
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(model_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                file=sys.stdout
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
        
        print(f"\nDownload complete! Model saved to {model_path}")
        
        # Verify file size
        expected_size_mb = 613  # Approximate size in MB
        actual_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        if actual_size_mb < expected_size_mb * 0.9:  # Allow 10% variation
            print(f"WARNING: Downloaded file size ({actual_size_mb:.2f} MB) is smaller than expected (~{expected_size_mb} MB)")
            print("The download may be incomplete or corrupted.")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nIf the download fails, you can:")
        print("1. Check your internet connection and try again")
        print("2. Download the model manually from https://huggingface.co/ali-vilab/ACE_Plus/tree/main/portrait")
        print(f"3. Save it to {model_path}")
        return False

if __name__ == "__main__":
    download_ace_plus_model() 