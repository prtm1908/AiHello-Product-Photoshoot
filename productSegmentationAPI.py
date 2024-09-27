from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from PIL import Image, ImageOps
from ultralytics import SAM
from skimage import morphology
import base64
import io

app = FastAPI()

# Load SAM model
model = SAM("sam2_l.pt")

class ImageRequest(BaseModel):
    image: str

def add_white_border(image, border_size=50):
    return ImageOps.expand(image, border=border_size, fill='white')

def touches_border(mask):
    return np.any(mask[0,:]) or np.any(mask[-1,:]) or np.any(mask[:,0]) or np.any(mask[:,-1])

def process_image(image, border_size=50):
    # Add white border
    bordered_image = add_white_border(image, border_size)

    # Process the bordered image
    results = model(bordered_image, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Get all masks
    masks = results[0].masks.data

    # Select all masks that don't touch the border
    selected_masks = []
    for mask in masks:
        mask_np = mask.cpu().numpy()
        if not touches_border(mask_np):
            selected_masks.append(mask_np)

    # If no suitable masks found, return None
    if not selected_masks:
        return None

    # Combine all selected masks
    combined_mask = np.any(selected_masks, axis=0)

    # Apply morphological operations to refine the mask
    refined_mask = morphology.closing(combined_mask, morphology.disk(5))
    refined_mask = morphology.remove_small_holes(refined_mask, area_threshold=500)

    # Remove the border from the refined mask
    original_size = image.size[::-1]  # PIL uses (width, height)
    refined_mask_original = refined_mask[border_size:-border_size, border_size:-border_size]

    # Resize the mask if necessary
    if refined_mask_original.shape != original_size:
        refined_mask_original = Image.fromarray(refined_mask_original)
        refined_mask_original = refined_mask_original.resize(image.size, Image.NEAREST)
        refined_mask_original = np.array(refined_mask_original)

    # Create masked image with transparent background
    original_array = np.array(image.convert('RGBA'))
    masked_image = original_array.copy()
    masked_image[~refined_mask_original] = [0, 0, 0, 0]  # Set non-mask area to transparent

    return Image.fromarray(masked_image)

@app.post("/segment")
async def segment_image(request: ImageRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))

        # Process the image
        result_image = process_image(image)

        if result_image is None:
            raise HTTPException(status_code=400, detail="No suitable masks found")

        # Convert the result image to base64
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {"segmented_image": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)