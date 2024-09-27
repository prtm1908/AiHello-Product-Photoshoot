from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import DiffusionPipeline
from PIL import Image, ImageOps
from io import BytesIO
from transparent_background import Remover
import torch
import base64

app = FastAPI()

# Load models
model_id = "yahoo-inc/photo-background-generation"
pipeline = DiffusionPipeline.from_pretrained(model_id, custom_pipeline=model_id)
pipeline = pipeline.to('cuda')
remover = Remover(mode='base')

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

class ImageRequest(BaseModel):
    image: str
    prompt: str
    seed: int = 13
    cond_scale: float = 1.0

@app.post("/generate_background")
async def generate_background(request: ImageRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        img = Image.open(BytesIO(image_data))
        
        # Resize image
        img = resize_with_padding(img, (512, 512))
        
        # Get foreground mask
        fg_mask = remover.process(img, type='map')
        mask = ImageOps.invert(fg_mask)
        
        # Set up generator
        generator = torch.Generator(device='cuda').manual_seed(request.seed)
        
        # Generate new background
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
        
        # Convert result to base64
        buffered = BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {"image": img_str}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)