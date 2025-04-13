#!/usr/bin/env python3
"""
Fix CUDA errors in the face swapper application.

This script helps diagnose and fix common CUDA errors, including the
"LayerNormKernelImpl not implemented for 'Half'" error that can occur
with some GPU configurations.
"""

import os
import torch
import sys
import subprocess

def print_section(title):
    """Print a section title with formatting."""
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "-"))
    print("=" * 50)

def print_info(message):
    """Print an information message."""
    print(f"[INFO] {message}")

def print_warning(message):
    """Print a warning message."""
    print(f"[WARN] {message}")

def print_error(message):
    """Print an error message."""
    print(f"[ERROR] {message}")

def print_success(message):
    """Print a success message."""
    print(f"[SUCCESS] {message}")

def check_cuda():
    """Check CUDA availability and configuration."""
    print_section("CUDA Configuration Check")
    
    try:
        print_info(f"Torch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print_success("CUDA is available!")
            print_info(f"CUDA version: {torch.version.cuda}")
            print_info(f"GPU device: {torch.cuda.get_device_name(0)}")
            print_info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Check current default dtype
            current_dtype = torch.get_default_dtype()
            print_info(f"Current default dtype: {current_dtype}")
            
            if current_dtype == torch.float16:
                print_warning("Default dtype is float16, which may cause the LayerNormKernelImpl error.")
            elif current_dtype == torch.float32:
                print_success("Default dtype is float32, which should avoid the LayerNormKernelImpl error.")
        else:
            print_error("CUDA is not available. The face swapper will run slowly on CPU.")
            print_info("If you have a compatible NVIDIA GPU, please install the appropriate CUDA drivers.")
            return False
    except Exception as e:
        print_error(f"Error checking CUDA configuration: {e}")
        return False
    
    return True

def fix_layer_norm_error():
    """Fix the LayerNormKernelImpl error by ensuring float32 is used."""
    print_section("Fixing LayerNormKernelImpl Error")
    
    # Create or modify the .env file to force float32
    env_file = '.env'
    env_var = 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32'
    
    try:
        # Check if .env exists
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                lines = f.readlines()
                
            # Check if the var is already there
            if any(env_var in line for line in lines):
                print_info("CUDA allocation configuration is already set.")
            else:
                # Append the var
                with open(env_file, 'a') as f:
                    f.write(f"\n{env_var}\n")
                print_info("Added CUDA allocation configuration to .env file.")
        else:
            # Create the .env file
            with open(env_file, 'w') as f:
                f.write(f"{env_var}\n")
            print_info("Created .env file with CUDA allocation configuration.")
            
        # Create a file to force float32 in the code
        with open('force_float32.py', 'w') as f:
            f.write("""
# This file is used to force PyTorch to use float32 precision
# Import this file in your scripts before importing other libraries

import torch
torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
""")
        print_info("Created force_float32.py file to force float32 precision.")
        
        # Modify run_faceswap_ace.py to import force_float32
        try:
            with open('run_faceswap_ace.py', 'r') as f:
                content = f.read()
                
            if 'import force_float32' not in content:
                # Add import after the first line (assuming it's a shebang or comment)
                lines = content.split('\n')
                lines.insert(1, "try: import force_float32\nexcept ImportError: pass  # Ignore if not found")
                modified_content = '\n'.join(lines)
                
                with open('run_faceswap_ace.py', 'w') as f:
                    f.write(modified_content)
                print_info("Modified run_faceswap_ace.py to import force_float32.py.")
            else:
                print_info("run_faceswap_ace.py already imports force_float32.py.")
        except Exception as e:
            print_warning(f"Could not modify run_faceswap_ace.py: {e}")
            print_info("You might need to manually add 'import force_float32' at the top of run_faceswap_ace.py.")
        
        print_success("Applied fixes for the LayerNormKernelImpl error!")
        print_info("Please restart your application for the changes to take effect.")
        
    except Exception as e:
        print_error(f"Error fixing LayerNormKernelImpl error: {e}")
        return False
    
    return True

def main():
    """Main function to check and fix CUDA errors."""
    print_section("CUDA Error Fixer for Face Swapper")
    print("This script will check your CUDA configuration and fix common errors.")
    
    if not check_cuda():
        print_warning("CUDA check failed. Please make sure you have a compatible GPU and drivers.")
    
    # Ask user if they want to apply the fix
    print("\nDo you want to apply the fix for the 'LayerNormKernelImpl' error? (y/n)")
    response = input("> ").strip().lower()
    
    if response == 'y':
        if fix_layer_norm_error():
            print_section("Summary")
            print_success("Applied fixes for the LayerNormKernelImpl error.")
            print_info("Please restart your application and try running it again.")
            print_info("If you still encounter errors, please report them with details.")
        else:
            print_error("Failed to apply fixes. Please check the error messages above.")
    else:
        print_info("No changes were made.")
    
    print("\nThank you for using the CUDA Error Fixer!")

if __name__ == "__main__":
    main() 