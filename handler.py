import runpod
import torch
import io
import os
import numpy as np
import requests
from PIL import Image, ImageOps
from transformers import Sam3Processor, Sam3Model

# --- Configuration --- #
MODEL_PATH = "/runpod-volume/models/sam3"
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
        print(f"Failed to download image from {url}: {str(e)}")
        return None

def process_mask_only(pil_image, prompt_text, bbox, threshold, mask_threshold):
    """Performs inference and returns a PIL mask."""
    
    input_boxes = [[bbox]] if bbox else None

    inputs = processor(
        images=pil_image, 
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
    
    # 1. PARSE LIST INPUTS
    image_urls = job_input.get("image_urls", [])
    output_locations = job_input.get("output_locations", [])
    bboxes_input = job_input.get("bboxes", [])
    
    # Validation
    if not isinstance(image_urls, list) or not isinstance(output_locations, list):
         return {"status": "error", "message": "'image_urls' and 'output_locations' must be lists."}
         
    if len(image_urls) == 0:
        return {"status": "error", "message": "No images provided."}

    if len(image_urls) != len(output_locations):
        return {"status": "error", "message": f"Mismatch: Received {len(image_urls)} images but {len(output_locations)} output locations."}

    try:
        init_model()
        
        prompt_text = job_input.get("prompt_text", "object")
        threshold = job_input.get("threshold", 0.3) 
        mask_threshold = job_input.get("mask_threshold", 0.5)
        
        generated_results = [] # Stores (index, mask_pil)
        failed_items = []
        
        # 2. GENERATION PHASE
        for i, url in enumerate(image_urls):
            # Get corresponding bbox if it exists in the list, else None
            bbox = bboxes_input[i] if (isinstance(bboxes_input, list) and i < len(bboxes_input)) else None
            
            pil_image = download_image_from_url(url)
            
            if pil_image:
                mask_pil = process_mask_only(pil_image, prompt_text, bbox, threshold, mask_threshold)
                
                if mask_pil:
                    generated_results.append((i, mask_pil))
                else:
                    print(f"⚠️ No mask generated for image index {i}")
            else:
                failed_items.append(f"Index {i} (Download Failed)")
                print(f"⚠️ Skipping index {i} due to download failure.")

        # --- MEMORY CLEANUP ---
        if device == "cuda":
            torch.cuda.empty_cache()

        # 3. BATCH UPLOAD PHASE
        successful_count = 0

        for i, mask_pil in generated_results:
            upload_url = output_locations[i]
            
            if upload_mask_to_azure(mask_pil, upload_url):
                successful_count += 1
            else:
                failed_items.append(f"Index {i} (Upload Failed)")

        # 4. RESPONSE LOGIC
        if len(failed_items) == 0 and successful_count > 0:
            return {
                "status": "success",
                "message": f"All {successful_count} masks uploaded successfully."
            }
        elif successful_count > 0:
            return {
                "status": "partial_success",
                "message": f"Uploaded {successful_count} masks, but failed on: {', '.join(failed_items)}"
            }
        else:
            return {
                "status": "error",
                "message": "No masks were successfully uploaded.",
                "details": failed_items
            }

    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})