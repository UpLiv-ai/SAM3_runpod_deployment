import requests
import base64
import os
import json

# --- Configuration ---
API_KEY = "REPLACE_WITH_KEY"  # Replace with your actual key
ENDPOINT_ID = "288ejvv8hk3qfb"
URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"

# Input Settings
INPUT_IMAGE_PATH = "test_image_1.jpg"  # Make sure this file exists locally
PROMPT = "stained wood dresser and shelves"

def encode_image(image_path):
    """Encodes a local image file to a base64 string."""
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found.")
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def save_b64_image(b64_str, output_path):
    """Decodes a base64 string and saves it as an image file."""
    try:
        image_data = base64.b64decode(b64_str)
        with open(output_path, "wb") as f:
            f.write(image_data)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Failed to save {output_path}: {e}")

def main():
    print(f"--- Sending request to Endpoint: {ENDPOINT_ID} ---")
    
    # 1. Prepare Image
    b64_image = encode_image(INPUT_IMAGE_PATH)
    if not b64_image:
        return

    # 2. Construct Payload
    # NOTE: We use 'prompt_text' to match the variable name in your handler.py
    payload = {
        "input": {
            "image": b64_image,
            "prompt_text": PROMPT
        }
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # 3. Send Request (using runsync for immediate response)
    try:
        response = requests.post(URL, headers=headers, json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        if response is not None:
             print(response.text)
        return

    # 4. Handle Response
    status = result.get("status")
    print(f"Job Status: {status}")

    if status == "COMPLETED":
        output = result.get("output", {})
        
        # Check if the handler returned an error inside the JSON
        if "error" in output:
            print(f"Handler Error: {output['error']}")
        elif "message" in output:
            print(f"Handler Message: {output['message']}")
        else:
            # Extract and Save Images
            if "output_image" in output:
                save_b64_image(output["output_image"], "api_result_overlay.png")
            
            if "output_mask" in output:
                save_b64_image(output["output_mask"], "api_result_mask.png")
                
            print("\nSuccess! Check api_result_overlay.png and api_result_mask.png")
            
    else:
        print("Job did not complete successfully.")
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()