import runpod
import torch
import io
import os
import sys
import argparse
import numpy as np
import requests # New dependency
from PIL import Image, ImageOps 
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- Configuration ---
# Use the Network Volume path if set, otherwise fallback
if os.path.exists('/runpod-volume'):
    base_volume_path = '/runpod-volume'
else:
    base_volume_path = '/workspace'

MODEL_DIR = os.path.join(base_volume_path, "models", "sam3")
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Global Model Storage ---
model = None
processor = None

def init_model():
    """Initializes the model using LOCAL weights only."""
    global model, processor
    if model is not None:
        return

    print(f"--- Initializing SAM 3 on {device} ---")
    
    checkpoint_file = os.path.join(MODEL_DIR, "sam3.pt")
    print(f"Loading checkpoint from: {checkpoint_file}")

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_file}")

    try:
        # Load model using the 'checkpoint_path' argument
        model = build_sam3_image_model(checkpoint_path=checkpoint_file)
        model.to(device)
        processor = Sam3Processor(model)
        print("SAM 3 Model loaded successfully from local storage.")

    except Exception as e:
        print(f"CRITICAL: Failed to load SAM 3 Model: {e}")
        raise e

# --- Helper Functions ---

def download_image_from_url(url):
    """Downloads image from URL and fixes rotation."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content))
        
        # CRITICAL: Fix rotation metadata (EXIF)
        image = ImageOps.exif_transpose(image)
        
        return image.convert("RGB")
    except Exception as e:
        raise Exception(f"Failed to download image: {str(e)}")

def upload_mask_to_azure(pil_image, upload_url):
    """Uploads the binary mask to Azure Blob Storage via SAS URL."""
    try:
        # Convert PIL image to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        image_data = buffer.getvalue()

        # Azure Blob Storage typically requires this header for PUT uploads
        headers = {
            'x-ms-blob-type': 'BlockBlob',
            'Content-Type': 'image/png'
        }

        response = requests.put(upload_url, data=image_data, headers=headers, timeout=60)
        response.raise_for_status()
        print("Upload successful.")
        return True
    except Exception as e:
        raise Exception(f"Failed to upload output to Azure: {str(e)}")

# --- Serverless Handler ---

def handler(job):
    job_input = job.get("input", {})
    
    # 1. Input Parsing
    image_url = job_input.get("image_url")
    description = job_input.get("description")
    output_location = job_input.get("output_location")

    # Basic Validation
    if not image_url:
        return {"status": "error", "error_message": "Missing 'image_url' in input."}
    if not description:
        return {"status": "error", "error_message": "Missing 'description' in input."}
    if not output_location:
        return {"status": "error", "error_message": "Missing 'output_location' in input."}

    try:
        # 2. Ensure model is loaded
        init_model()
        
        # 3. Download and Preprocess
        image = download_image_from_url(image_url)
        
        # 4. Inference
        inference_state = processor.set_image(image)
        results = processor.set_text_prompt(state=inference_state, prompt=description)
        
        masks = results.get("masks")
        if masks is None or len(masks) == 0:
            return {"status": "error", "error_message": "No objects found matching description."}

        # 5. Mask Processing (Combine masks into one binary image)
        combined_mask = np.zeros((image.height, image.width), dtype=bool)
        
        for mask_tensor in masks:
            if isinstance(mask_tensor, torch.Tensor):
                mask_np = mask_tensor.cpu().numpy().squeeze()
            else:
                mask_np = mask_tensor
            combined_mask = np.logical_or(combined_mask, mask_np > 0)

        # Convert to Uint8 (0 vs 255)
        mask_uint8 = (combined_mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8, mode='L')

        # 6. Upload Result
        upload_mask_to_azure(mask_pil, output_location)
        
        # 7. Return Success
        return {
            "status": "success",
            "message": "Mask generated and uploaded successfully."
        }

    except Exception as e:
        print(f"Error processing job: {e}")
        return {
            "status": "error", 
            "error_message": str(e)
        }

runpod.serverless.start({"handler": handler})

# # --- Local Testing Block (Updated for URL inputs) ---
# if __name__ == "__main__":
#     # To test this locally, you need real URLs or you need to mock the requests.
#     print("--- Running Local Test Mode ---")
    
#     # Mock job input
#     test_job = {
#         "input": {
#             "image_url": "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg", # Public test image
#             "description": "truck",
#             "output_location": "MOCK_UPLOAD" # This will fail the upload step if not a real SAS URL
#         }
#     }
    
#     # We override the upload function for local testing to save to disk instead
#     def mock_upload(pil_image, url):
#         print(f"Simulating upload to {url}...")
#         pil_image.save("local_test_mask.png")
#         print("Saved locally to local_test_mask.png instead.")
    
#     # Swap the function just for this test run
#     original_upload_func = upload_mask_to_azure
#     upload_mask_to_azure = mock_upload
    
#     # Run
#     init_model()
#     result = handler(test_job)
#     print("Result:", result)
    
# else:
#     runpod.serverless.start({"handler": handler})