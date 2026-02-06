import runpod
import torch
import base64
import io
import os
import numpy as np
import requests
from PIL import Image, ImageOps
from transformers import Sam3Processor, Sam3Model

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

def upload_mask_to_azure(pil_image, upload_url):
    """Uploads the binary mask to Azure Blob Storage via SAS URL."""
    try:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        image_data = buffer.getvalue()

        headers = {
            'x-ms-blob-type': 'BlockBlob',
            'Content-Type': 'image/png'
        }
        response = requests.put(upload_url, data=image_data, headers=headers, timeout=60)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Failed to upload: {str(e)}")
        return False

def decode_base64_image(base64_string):
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    image = ImageOps.exif_transpose(image) 
    return image.convert("RGB")

def process_mask_only(image_data_b64, prompt_text, bbox, threshold, mask_threshold):
    """Performs inference and returns a PIL mask."""
    image = decode_base64_image(image_data_b64)
    input_boxes = [[bbox]] if bbox else None

    inputs = processor(
        images=image, 
        text=prompt_text,
        input_boxes=input_boxes, 
        return_tensors="pt"
    ).to(device)
    
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
    if len(masks) == 0:
         return None

    # Combine Masks (Union)
    all_masks_np = masks.cpu().numpy().astype(bool)
    combined_mask_np = np.any(all_masks_np, axis=0)
    
    mask_uint8 = (combined_mask_np * 255).astype(np.uint8)
    return Image.fromarray(mask_uint8, mode='L')

def handler(job):
    job_input = job.get("input", {})
    images_input = job_input.get("images", {})
    output_locations = job_input.get("output_locations", {})
    
    if not images_input or not output_locations:
        return {"status": "error", "message": "Missing 'images' or 'output_locations'."}

    try:
        init_model()
        
        prompt_text = job_input.get("prompt_text", "object")
        threshold = job_input.get("threshold", 0.3) 
        mask_threshold = job_input.get("mask_threshold", 0.5)
        bboxes_input = job_input.get("bboxes", {})
        
        generated_masks = {}
        
        # 1. GENERATION PHASE
        for name, b64_data in images_input.items():
            bbox = bboxes_input.get(name)
            mask_pil = process_mask_only(b64_data, prompt_text, bbox, threshold, mask_threshold)
            
            if mask_pil:
                generated_masks[name] = mask_pil
            else:
                print(f"⚠️ No mask generated for {name}")

        # --- MEMORY CLEANUP ---
        # Freeing up VRAM before starting network tasks
        if device == "cuda":
            torch.cuda.empty_cache()

        # 2. BATCH UPLOAD PHASE
        failed_uploads = []
        successful_count = 0

        for name, mask_pil in generated_masks.items():
            azure_url = output_locations.get(name)
            if not azure_url:
                failed_uploads.append(f"{name} (No URL provided)")
                continue
                
            if upload_mask_to_azure(mask_pil, azure_url):
                successful_count += 1
            else:
                failed_uploads.append(name)

        # 3. RESPONSE LOGIC
        if len(failed_uploads) == 0 and successful_count > 0:
            return {
                "status": "success",
                "message": f"All {successful_count} masks uploaded successfully."
            }
        elif successful_count > 0:
            return {
                "status": "partial_success",
                "message": f"Uploaded {successful_count} masks, but failed on: {', '.join(failed_uploads)}"
            }
        else:
            return {
                "status": "error",
                "message": "No masks were successfully uploaded.",
                "details": failed_uploads
            }

    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})