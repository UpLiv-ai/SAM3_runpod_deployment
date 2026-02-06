import runpod
import torch
import base64
import io
import os
import glob
import numpy as np
from PIL import Image, ImageOps
from transformers import Sam3Processor, Sam3Model
from huggingface_hub import snapshot_download

# --- Configuration ---
MODEL_PATH = "/workspace/models/sam3"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = None
processor = None

def init_model():
    global model, processor
    if model is not None:
        return
    
    try:
    
        model = Sam3Model.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(device)
        
        processor = Sam3Processor.from_pretrained(MODEL_PATH)
        print("✅ SAM 3 Model loaded successfully.")
        
    except Exception as e:
        print(f"❌ CRITICAL: Failed to load model: {e}")
        raise e

def decode_base64_image(base64_string):
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    image = ImageOps.exif_transpose(image) # Fix rotation
    return image.convert("RGB")

def encode_image_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def process_single_image(image_data_b64, prompt_text, threshold, mask_threshold):
    """Helper function to process one image independently."""
    try:
        image = decode_base64_image(image_data_b64)
        
        inputs = processor(
            images=image, 
            text=prompt_text, 
            return_tensors="pt"
        ).to(device)
        
        # Cast to float16 if needed
        for key in inputs:
            if inputs[key].dtype == torch.float32:
                inputs[key] = inputs[key].to(model.dtype)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        masks = results['masks']
        scores = results['scores']
        
        if len(masks) == 0:
             return {"found": False, "message": "No objects found"}

        # Select Best Mask
        best_idx = scores.argmax().item()
        best_mask_tensor = masks[best_idx]
        best_score = scores[best_idx].item()
        
        # Create Output
        mask_np = best_mask_tensor.cpu().numpy().astype(bool)
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8, mode='L')
        
        # Overlay
        overlay = Image.new("RGBA", image.size, (255, 0, 0, 100))
        result_image = Image.composite(overlay, image.convert("RGBA"), mask_pil)
        
        return {
            "found": True,
            "output_image": encode_image_to_base64(result_image),
            "output_mask": encode_image_to_base64(mask_pil),
            "score": float(best_score)
        }
    except Exception as e:
        return {"found": False, "error": str(e)}

def handler(job):
    job_input = job.get("input", {})
    
    # Support "images" (batch) or fallback to "image" (single)
    images_input = job_input.get("images")
    if not images_input and "image" in job_input:
        # Normalize single input to list format
        images_input = {"single_image": job_input["image"]}
    
    if not images_input:
        return {"error": "Missing 'images' dict or 'image' string in input."}

    try:
        init_model()
        
        prompt_text = job_input.get("prompt_text", "polished black grand piano")
        threshold = job_input.get("threshold", 0.3) 
        mask_threshold = job_input.get("mask_threshold", 0.5)
        
        batch_results = {}
        
        # Iterate over the dictionary of images { "filename": "b64", ... }
        for name, b64_data in images_input.items():
            print(f"Processing {name}...")
            result = process_single_image(b64_data, prompt_text, threshold, mask_threshold)
            batch_results[name] = result
            
        return {"results": batch_results}

    except Exception as e:
        return {"error": str(e)}

# --- Local Test Block for Batch Processing ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch Test SAM 3")
    parser.add_argument("--input_dir", required=True, help="Folder containing images")
    parser.add_argument("--prompt", default="polished black grand piano")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Directory {args.input_dir} not found.")
        import sys; sys.exit(1)
        
    print(f"--- Batch Testing with prompt: '{args.prompt}' ---")
    
    # 1. Gather all images
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
    
    if not image_files:
        print("No images found in directory.")
        import sys; sys.exit(1)
        
    # 2. Build Payload
    images_payload = {}
    for img_path in image_files:
        filename = os.path.basename(img_path)
        with open(img_path, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode("utf-8")
        images_payload[filename] = b64_str
        
    test_job = { 
        "input": { 
            "images": images_payload, 
            "prompt_text": args.prompt 
        } 
    }
    
    # 3. Run Handler
    try:
        init_model()
        response = handler(test_job)
        
        results = response.get("results", {})
        
        # 4. Save Outputs
        for filename, res in results.items():
            if res.get("found"):
                base_name = os.path.splitext(filename)[0]
                
                # Save Overlay
                overlay_path = os.path.join(args.input_dir, f"{base_name}_overlay.png")
                with open(overlay_path, "wb") as f:
                    f.write(base64.b64decode(res["output_image"]))
                
                # Save Mask
                mask_path = os.path.join(args.input_dir, f"{base_name}_mask.png")
                with open(mask_path, "wb") as f:
                    f.write(base64.b64decode(res["output_mask"]))
                    
                print(f"✅ {filename}: Score {res['score']:.4f} -> Saved overlay/mask")
            else:
                msg = res.get("message") or res.get("error")
                print(f"❌ {filename}: {msg}")
                
    except Exception as e:
        print(f"❌ Global Error: {e}")