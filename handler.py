import runpod
import torch
import base64
import io
import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageOps 
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- Configuration ---

# 1. User-defined Path Logic
if os.path.exists('/runpod-volume'):
    base_volume_path = '/runpod-volume'
else:
    # If testing locally or without volume mount
    base_volume_path = '/workspace'

# The folder containing the files shown in your screenshot
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
    
    # Define the specific file path to the weights
    checkpoint_file = os.path.join(MODEL_DIR, "sam3.pt")
    
    print(f"Loading checkpoint from: {checkpoint_file}")

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_file}")

    try:
        # We use strictly 'checkpoint_path' because we proved it worked in the previous test.
        model = build_sam3_image_model(
            checkpoint_path=checkpoint_file
        )
        
        model.to(device)
        processor = Sam3Processor(model)
        print("SAM 3 Model loaded successfully from local storage.")

    except Exception as e:
        print(f"CRITICAL: Failed to load SAM 3 Model: {e}")
        raise e

# --- Helper Functions ---

def decode_base64_image(base64_string):
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    image_data = base64.b64decode(base64_string)
    
    image = Image.open(io.BytesIO(image_data))
    # EXIF Rotation Fix
    image = ImageOps.exif_transpose(image)
    
    return image.convert("RGB")

def encode_image_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# --- Serverless Handler ---

def handler(job):
    job_input = job.get("input", {})
    
    if "image" not in job_input:
        return {"error": "Missing 'image' in input payload."}
    
    try:
        # Ensure model is loaded
        init_model()
        
        image = decode_base64_image(job_input["image"])
        prompt_text = job_input.get("prompt_text", "object")
        
        # Inference
        inference_state = processor.set_image(image)
        results = processor.set_text_prompt(state=inference_state, prompt=prompt_text)
        
        masks = results.get("masks")
        if masks is None or len(masks) == 0:
            return {"message": "No objects found."}

        # --- Mask Processing ---
        combined_mask = np.zeros((image.height, image.width), dtype=bool)
        
        for mask_tensor in masks:
            if isinstance(mask_tensor, torch.Tensor):
                mask_np = mask_tensor.cpu().numpy().squeeze()
            else:
                mask_np = mask_tensor
            combined_mask = np.logical_or(combined_mask, mask_np > 0)

        mask_uint8 = (combined_mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8, mode='L')

        # Create Visualization Overlay
        overlay = Image.new("RGBA", image.size, (255, 0, 0, 100))
        mask_rgba = Image.fromarray(mask_uint8, mode='L')
        result_image = Image.composite(overlay, image.convert("RGBA"), mask_rgba)
        
        return {
            "output_image": encode_image_to_base64(result_image),
            "output_mask": encode_image_to_base64(mask_pil),
            "found": True
        }

    except Exception as e:
        return {"error": str(e)}
    
    
runpod.serverless.start({"handler": handler})

# # --- Local Testing Block ---
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Test SAM 3 locally")
#     parser.add_argument("--input_image", required=True, help="Path to input image")
#     parser.add_argument("--output_image", default="output_overlay.png", help="Path to save overlay result")
#     parser.add_argument("--output_mask", default="output_mask.png", help="Path to save binary mask")
#     parser.add_argument("--prompt", default="cat", help="Text prompt for segmentation")
    
#     args = parser.parse_args()
    
#     if not os.path.exists(args.input_image):
#         print(f"Error: Input file {args.input_image} not found.")
#         sys.exit(1)
        
#     print(f"--- Running Local Test with prompt: '{args.prompt}' ---")
    
#     with open(args.input_image, "rb") as f:
#         b64_str = base64.b64encode(f.read()).decode("utf-8")
        
#     test_job = {
#         "input": {
#             "image": b64_str,
#             "prompt_text": args.prompt
#         }
#     }
    
#     # Run the init manually to catch errors
#     init_model()
    
#     result = handler(test_job)
    
#     if "error" in result:
#         print(f"Error in handler: {result['error']}")
#     elif "message" in result:
#          print(f"Handler completed with message: {result['message']}")
#     elif result.get("output_image"):
#         overlay_data = base64.b64decode(result["output_image"])
#         with open(args.output_image, "wb") as f:
#             f.write(overlay_data)
#         mask_data = base64.b64decode(result["output_mask"])
#         with open(args.output_mask, "wb") as f:
#             f.write(mask_data)
#         print(f"Success! Saved to {args.output_image} and {args.output_mask}")
#     else:
#         print(f"Unknown result format: {result}")

# else:
#     runpod.serverless.start({"handler": handler})