#!/usr/bin/env python3
"""
Test script for the asynchronous face swap API
"""

import requests
import time
import os
import json
import argparse
from urllib.parse import urlparse
from pathlib import Path

def download_file(url, destination):
    """Download a file from a URL to a local destination"""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(destination, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return destination

def main(main_image_path, ref_image_path, mask_image_path=None, server_url='http://localhost:7860'):
    """
    Test the asynchronous face swap API by submitting a task and polling for completion
    
    Args:
        main_image_path: Path to the main/target image where face will be swapped
        ref_image_path: Path to the reference image with the face to use
        mask_image_path: Optional path to a mask image
        server_url: Base URL of the face swap API server
    """
    print(f"Testing async face swap API at {server_url}")
    
    # Check if files exist
    for path in [main_image_path, ref_image_path]:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            return
    
    if mask_image_path and not os.path.exists(mask_image_path):
        print(f"Warning: Mask file not found: {mask_image_path}")
        mask_image_path = None
    
    # Prepare the files for upload
    files = {
        'main': (os.path.basename(main_image_path), open(main_image_path, 'rb')),
        'ref': (os.path.basename(ref_image_path), open(ref_image_path, 'rb')),
    }
    
    if mask_image_path:
        files['mask'] = (os.path.basename(mask_image_path), open(mask_image_path, 'rb'))
    
    # Submit the async task
    print("Submitting async face swap task...")
    response = requests.post(f"{server_url}/swap/async", files=files)
    
    # Close file handlers
    for f in files.values():
        f[1].close()
    
    if response.status_code != 200:
        print(f"Error submitting task: {response.status_code} {response.text}")
        return
    
    task_data = response.json()
    task_id = task_data['task_id']
    print(f"Task submitted successfully with ID: {task_id}")
    print(f"Initial status: {task_data['status']}")
    
    # Poll for completion
    print("\nPolling for task completion...")
    completed = False
    start_time = time.time()
    
    while not completed and (time.time() - start_time) < 300:  # 5-minute timeout
        time.sleep(2)  # Poll every 2 seconds
        
        status_response = requests.get(f"{server_url}/tasks/{task_id}")
        
        if status_response.status_code != 200:
            print(f"Error checking status: {status_response.status_code} {status_response.text}")
            continue
        
        status_data = status_response.json()
        current_status = status_data['status']
        
        # Display progress
        elapsed = time.time() - start_time
        print(f"Status: {current_status} (elapsed: {elapsed:.1f}s)", end="\r")
        
        if current_status == 'completed':
            print(f"\nTask completed in {elapsed:.1f} seconds!")
            
            # Download the result
            result_url = status_data['result_url']
            print(f"Result URL: {result_url}")
            
            # Extract filename from URL or use a default
            url_path = urlparse(result_url).path
            filename = os.path.basename(url_path) or "result.png"
            output_path = f"async_{filename}"
            
            print(f"Downloading result to {output_path}...")
            download_file(result_url, output_path)
            print(f"Result saved to {output_path}")
            
            completed = True
        elif current_status == 'failed':
            print(f"\nTask failed after {elapsed:.1f} seconds!")
            if 'error' in status_data:
                print(f"Error: {status_data['error']}")
            completed = True
    
    if not completed:
        print("\nTimeout waiting for task completion!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the asynchronous face swap API")
    parser.add_argument("main", help="Path to the main/target image")
    parser.add_argument("ref", help="Path to the reference face image")
    parser.add_argument("--mask", help="Optional path to a mask image")
    parser.add_argument("--server", default="http://localhost:7860", help="API server URL")
    
    args = parser.parse_args()
    main(args.main, args.ref, args.mask, args.server) 