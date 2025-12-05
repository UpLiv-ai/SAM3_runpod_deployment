import runpod
import torch
import base64
import io
import os
import sys
import argparse
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- Configuration ---
# Use the Network Volume path if set, otherwise fallback (mostly for local dev without volumes)
CHECKPOINT_DIR = os.environ.get("SAM3_CHECKPOINT_DIR", "/runpod-volume/sam3/checkpoints")
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Global Model Storage ---
model = None
processor = None

def init_model():
    """Initializes the model once during the worker cold start."""
    global model, processor
    if model is not None:
        return

    print(f"Initializing SAM 3 on {device} using weights from {CHECKPOINT_DIR}...")
    try:
        # NOTE: You might need to adjust how build_sam3_image_model finds checkpoints 
        # depending on the exact library version. Usually it looks in standard cache 
        # or you might need to symlink your volume to the cache folder.
        model = build_sam3_image_model() 
        model.to(device)
        processor = Sam3Processor(model)
        print("SAM 3 Model loaded successfully.")
    except Exception as e:
        print(f"CRITICAL: Failed to load SAM 3 Model: {e}")
        raise e

# --- Helper Functions ---

def decode_base64_image(base64_string):
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

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
        # Ensure model is loaded (idempotent)
        init_model()
        
        image = decode_base64_image(job_input["image"])
        prompt_text = job_input.get("prompt_text", "object") # Default prompt
        
        # Inference
        inference_state = processor.set_image(image)
        results = processor.set_text_prompt(state=inference_state, prompt=prompt_text)
        
        masks = results.get("masks")
        if masks is None or len(masks) == 0:
            return {"message": "No objects found."}

        # --- Mask Processing ---
        # SAM 3 returns a list of masks (one per object found).
        # We combine them into a single binary mask for simplicity in this workflow.
        # This creates a "Foreground vs Background" mask.
        
        combined_mask = np.zeros((image.height, image.width), dtype=bool)
        
        for mask_tensor in masks:
            if isinstance(mask_tensor, torch.Tensor):
                mask_np = mask_tensor.cpu().numpy().squeeze()
            else:
                mask_np = mask_tensor
            
            # Logical OR to combine masks
            combined_mask = np.logical_or(combined_mask, mask_np > 0)

        # 1. Create Binary Mask Image (White object, Black background)
        # This is what SAM 3D needs.
        mask_uint8 = (combined_mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8, mode='L')

        # 2. Create Visualization Overlay (Red tint on object)
        # This is for human verification.
        overlay = Image.new("RGBA", image.size, (255, 0, 0, 100))
        mask_rgba = Image.fromarray(mask_uint8, mode='L')
        result_image = Image.composite(overlay, image.convert("RGBA"), mask_rgba)
        
        return {
            "output_image": encode_image_to_base64(result_image), # Visual overlay
            "output_mask": encode_image_to_base64(mask_pil),      # Raw binary mask
            "found": True
        }

    except Exception as e:
        return {"error": str(e)}

# --- Local Testing Block ---
if __name__ == "__main__":
    # This block only runs when you execute `python handler.py` explicitly
    parser = argparse.ArgumentParser(description="Test SAM 3 locally")
    parser.add_argument("--input_image", required=True, help="Path to input image")
    parser.add_argument("--output_image", default="output_overlay.png", help="Path to save overlay result")
    parser.add_argument("--output_mask", default="output_mask.png", help="Path to save binary mask")
    parser.add_argument("--prompt", default="cat", help="Text prompt for segmentation")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_image):
        print(f"Error: Input file {args.input_image} not found.")
        sys.exit(1)
        
    print(f"--- Running Local Test with prompt: '{args.prompt}' ---")
    
    # 1. Read file and encode to base64 (simulating API payload)
    with open(args.input_image, "rb") as f:
        b64_str = base64.b64encode(f.read()).decode("utf-8")
        
    # 2. Construct simulated job
    test_job = {
        "input": {
            "image": b64_str,
            "prompt_text": args.prompt
        }
    }
    
    # 3. Initialize model manually
    init_model()
    
    # 4. Run handler
    result = handler(test_job)
    
    # 5. Process result
    if "error" in result:
        print(f"Error in handler: {result['error']}")
    elif result.get("output_image"):
        # Save Overlay
        overlay_data = base64.b64decode(result["output_image"])
        with open(args.output_image, "wb") as f:
            f.write(overlay_data)
            
        # Save Mask
        mask_data = base64.b64decode(result["output_mask"])
        with open(args.output_mask, "wb") as f:
            f.write(mask_data)
            
        print(f"Success!")
        print(f"1. Overlay saved to: {args.output_image}")
        print(f"2. Binary Mask saved to: {args.output_mask}")
    else:
        print("Unknown result format.")

else:
    # If imported by RunPod, start the worker
    runpod.serverless.start({"handler": handler})