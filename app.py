import asyncio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
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
import random
from celery import Celery
from celery.signals import worker_init
from celery.result import AsyncResult
import uvicorn
import multiprocessing

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Celery configuration
celery = Celery('tasks', broker='redis://host.docker.internal:6379/0', backend='redis://host.docker.internal:6379/0')
celery.conf.update(task_track_started=True)

# Global variables to hold the models
pipeline = None
remover = None
sam_model = None

@worker_init.connect
def init_worker(**kwargs):
    global pipeline, remover, sam_model
    
    print("Initializing models...")
    model_id = "yahoo-inc/photo-background-generation"
    pipeline = DiffusionPipeline.from_pretrained(model_id, custom_pipeline=model_id)
    pipeline = pipeline.to('cuda')
    remover = Remover(mode='base')
    sam_model = SAM("sam2_l.pt")
    print("Models initialized successfully.")

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

@celery.task(bind=True)
def generate_background_task(self, image_data, prompt, seed, cond_scale):
    try:
        self.update_state(state='PROGRESS', meta={'progress': 0})
        
        # Load and process image
        img = Image.open(BytesIO(base64.b64decode(image_data)))
        img = resize_with_padding(img, (2048, 2048))
        self.update_state(state='PROGRESS', meta={'progress': 20})
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Generate mask
        fg_mask = remover.process(img, type='map')
        mask = ImageOps.invert(fg_mask)
        self.update_state(state='PROGRESS', meta={'progress': 40})
        
        # Generate background
        generator = torch.Generator(device='cuda').manual_seed(seed)
        self.update_state(state='PROGRESS', meta={'progress': 60})
        
        with torch.autocast("cuda"):
            result_image = pipeline(
                prompt=prompt,
                image=img,
                mask_image=mask,
                control_image=mask,
                num_images_per_prompt=1,
                generator=generator,
                num_inference_steps=20,
                guess_mode=False,
                controlnet_conditioning_scale=cond_scale
            ).images[0]
        self.update_state(state='PROGRESS', meta={'progress': 80})
        
        # Encode result
        buffered = BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        self.update_state(state='PROGRESS', meta={'progress': 100})
        return {"image": img_str, "seed": seed}
    
    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@celery.task(bind=True)
def segment_image_task(self, image_data):
    try:
        self.update_state(state='PROGRESS', meta={'progress': 0})
        
        # Load image
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        self.update_state(state='PROGRESS', meta={'progress': 20})
        
        # Process image
        result_image = process_image(image)
        self.update_state(state='PROGRESS', meta={'progress': 60})
        
        if result_image is None:
            raise ValueError("No suitable masks found")
        
        # Encode result
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        self.update_state(state='PROGRESS', meta={'progress': 100})
        return {"segmented_image": img_str}
    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@celery.task(bind=True)
def segment_and_save_image_task(self, image_data):
    try:
        self.update_state(state='PROGRESS', meta={'progress': 0})
        
        # Load image
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        self.update_state(state='PROGRESS', meta={'progress': 20})
        
        # Process image
        result_image = process_image(image)
        self.update_state(state='PROGRESS', meta={'progress': 60})
        
        if result_image is None:
            raise ValueError("No suitable masks found")
        
        # Save image
        filename = f"{uuid.uuid4()}.png"
        os.makedirs('output', exist_ok=True)
        file_path = os.path.join('output', filename)
        result_image.save(file_path, format="PNG")
        
        self.update_state(state='PROGRESS', meta={'progress': 100})
        return {"filename": filename}
    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@app.post("/generate_background")
async def generate_background(request: BackgroundImageRequest):
    try:
        task = generate_background_task.delay(request.image, request.prompt, request.seed, request.cond_scale)
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/segment")
async def segment_image(request: ImageRequest):
    try:
        task = segment_image_task.delay(request.image)
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/segment_and_save")
async def segment_and_save_image(request: ImageRequest):
    try:
        task = segment_and_save_image_task.delay(request.image)
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery)
    if task_result.state == 'PENDING':
        response = {
            'state': task_result.state,
            'status': 'Task is pending...'
        }
    elif task_result.state != 'FAILURE':
        response = {
            'state': task_result.state,
            'status': task_result.info.get('status', '')
        }
        if task_result.state == 'SUCCESS':
            response['result'] = task_result.result
    else:
        response = {
            'state': task_result.state,
            'status': str(task_result.info)
        }
    return response

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        del self.active_connections[client_id]

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_json()
            if 'task_id' in data:
                task_id = data['task_id']
                while True:
                    task_result = AsyncResult(task_id, app=celery)
                    if task_result.state == 'PENDING':
                        await websocket.send_json({
                            'task_id': task_id,
                            'state': task_result.state,
                            'status': 'Task is pending...'
                        })
                    elif task_result.state != 'FAILURE':
                        await websocket.send_json({
                            'task_id': task_id,
                            'state': task_result.state,
                            'status': task_result.info.get('status', ''),
                            'progress': task_result.info.get('progress', 0)
                        })
                        if task_result.state == 'SUCCESS':
                            await websocket.send_json({
                                'task_id': task_id,
                                'state': task_result.state,
                                'result': task_result.result
                            })
                            break
                    else:
                        await websocket.send_json({
                            'task_id': task_id,
                            'state': task_result.state,
                            'status': str(task_result.info)
                        })
                        break
                    await asyncio.sleep(1)  # Check every second
    except WebSocketDisconnect:
        manager.disconnect(client_id)

# Background task to update clients about task progress
async def update_task_status():
    while True:
        for client_id, websocket in manager.active_connections.items():
            task_id = await websocket.receive_text()
            task_result = AsyncResult(task_id, app=celery)
            await manager.send_personal_message(f"Task {task_id} status: {task_result.state}", client_id)
        await asyncio.sleep(5)  # Update every 5 seconds

def run_celery():
    os.environ.setdefault('FORKED_BY_MULTIPROCESSING', '1')
    argv = [
        'worker',
        '--loglevel=info',
        '-P', 'solo'  # Use solo pool to avoid subprocess issues
    ]
    celery.worker_main(argv)

def run_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=5103)

if __name__ == "__main__":
    # Create and start the Celery worker process
    celery_process = multiprocessing.Process(target=run_celery)
    celery_process.start()

    # Create and start the Uvicorn server process
    uvicorn_process = multiprocessing.Process(target=run_uvicorn)
    uvicorn_process.start()

    try:
        # Wait for both processes to complete (which they won't, unless stopped)
        celery_process.join()
        uvicorn_process.join()
    except KeyboardInterrupt:
        print("Shutting down...")
        celery_process.terminate()
        uvicorn_process.terminate()
        celery_process.join()
        uvicorn_process.join()
