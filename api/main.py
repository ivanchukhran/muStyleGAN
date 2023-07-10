import io
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from generate import generate

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/landscape")
async def generate_single_image():
    image = generate(1)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    return StreamingResponse(image_bytes, media_type="image/png")


@app.get("/landscape/{num_images}")
async def generate_multiple_images(num_images: int):
    images = generate(num_images)
    image_bytes = io.BytesIO()
    images.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    return StreamingResponse(image_bytes, media_type="image/png")
