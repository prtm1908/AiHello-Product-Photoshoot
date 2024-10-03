from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import DiffusionPipeline
from PIL import Image, ImageOps
from io import BytesIO
from transparent_background import Remover
import torch
import base64
import numpy as np
from ultralytics import SAM
from skimage import morphology
import io
import uuid
import os

app = FastAPI()

# Load models
model_id = "yahoo-inc/photo-background-generation"
pipeline = DiffusionPipeline.from_pretrained(model_id, custom_pipeline=model_id)
pipeline = pipeline.to('cuda')
remover = Remover(mode='base')

# Load SAM model
sam_model = SAM("sam2_l.pt")

class ImageRequest(BaseModel):
    image: str

class BackgroundImageRequest(ImageRequest):
    prompt: str
    seed: int = 13
    cond_scale: float = 1.0

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

def add_white_border(image, border_size=50):
    return ImageOps.expand(image, border=border_size, fill='white')

def touches_border(mask):
    return np.any(mask[0,:]) or np.any(mask[-1,:]) or np.any(mask[:,0]) or np.any(mask[:,-1])

def process_image(image, border_size=50):
    bordered_image = add_white_border(image, border_size)
    results = sam_model(bordered_image, device='cuda' if torch.cuda.is_available() else 'cpu')
    masks = results[0].masks.data
    selected_masks = [mask.cpu().numpy() for mask in masks if not touches_border(mask.cpu().numpy())]
    
    if not selected_masks:
        return None
    
    combined_mask = np.any(selected_masks, axis=0)
    refined_mask = morphology.closing(combined_mask, morphology.disk(5))
    refined_mask = morphology.remove_small_holes(refined_mask, area_threshold=500)
    
    original_size = image.size[::-1]
    refined_mask_original = refined_mask[border_size:-border_size, border_size:-border_size]
    
    if refined_mask_original.shape != original_size:
        refined_mask_original = Image.fromarray(refined_mask_original)
        refined_mask_original = refined_mask_original.resize(image.size, Image.NEAREST)
        refined_mask_original = np.array(refined_mask_original)
    
    original_array = np.array(image.convert('RGBA'))
    masked_image = original_array.copy()
    masked_image[~refined_mask_original] = [0, 0, 0, 0]
    
    return Image.fromarray(masked_image)

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding, fill=(255, 255, 255, 0) if img.mode == 'RGBA' else (255, 255, 255))

@app.post("/generate_background")
async def generate_background(request: BackgroundImageRequest):
    try:
        image_data = base64.b64decode(request.image)
        img = Image.open(BytesIO(image_data))
        img = resize_with_padding(img, (512, 512))

        # Convert image to RGB if it's not already
        if img.mode != 'RGB':
            img = img.convert('RGB')

        fg_mask = remover.process(img, type='map')
        mask = ImageOps.invert(fg_mask)

        generator = torch.Generator(device='cuda').manual_seed(request.seed)
        
        with torch.autocast("cuda"):
            result_image = pipeline(
                prompt=request.prompt,
                image=img,
                mask_image=mask,
                control_image=mask,
                num_images_per_prompt=1,
                generator=generator,
                num_inference_steps=20,
                guess_mode=False,
                controlnet_conditioning_scale=request.cond_scale
            ).images[0]
        
        buffered = BytesIO()
        result_image.save(buffered, format="PNG")

        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {"image": img_str}
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/segment")
async def segment_image(request: ImageRequest):
    try:
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))
        result_image = process_image(image)
        
        if result_image is None:
            raise HTTPException(status_code=400, detail="No suitable masks found")
        
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {"segmented_image": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/segment_and_save")
async def segment_and_save_image(request: ImageRequest):
    try:
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))
        result_image = process_image(image)
        
        if result_image is None:
            raise HTTPException(status_code=400, detail="No suitable masks found")
        
        filename = f"{uuid.uuid4()}.png"
        os.makedirs('output', exist_ok=True)
        file_path = os.path.join('output', filename)
        result_image.save(file_path, format="PNG")
        
        return {"filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)