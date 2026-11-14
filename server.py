"""FastAPI backend: loads the conditional model once, serves /api/generate."""

import base64
import io
import threading
import time
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

import config
from diffusion import device
from model import UNet
from sample_cond import cond_ddim_sample

BASE = Path(__file__).parent

# Load the model once at startup
model = UNet(num_classes=10).to(device)
model.load_state_dict(
    torch.load(BASE / "checkpoints" / "cond_ema.pt", map_location=device)
)
model.eval()

lock = threading.Lock()  # one generation at a time (CPU)

app = FastAPI(title="Diffusion MNIST API")


class GenRequest(BaseModel):
    digit: int = Field(7, ge=0, le=9)
    count: int = Field(8, ge=1, le=16)
    w: float = Field(1.2, ge=0.0, le=3.0)
    steps: int = Field(20, ge=5, le=100)
    seed: int = Field(0, ge=0, le=2**31 - 1)


def tensor_to_png_b64(img, size=192):
    """img: (1, H, W) tensor in [-1, 1] -> base64 PNG string."""
    arr = ((img.squeeze(0) + 1) / 2 * 255).round().clamp(0, 255).byte().numpy()
    pil = Image.fromarray(arr).resize((size, size), Image.BICUBIC)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/api/info")
def info():
    return {"img_size": config.IMG_SIZE, "device": str(device)}


@app.post("/api/generate")
def generate(req: GenRequest):
    with lock:
        start = time.perf_counter()
        torch.manual_seed(req.seed)
        labels = torch.full((req.count,), req.digit, device=device, dtype=torch.long)
        imgs = cond_ddim_sample(model, labels, steps=req.steps, w=req.w).cpu()
        elapsed = time.perf_counter() - start
    return {
        "images": [tensor_to_png_b64(im) for im in imgs],
        "seconds": round(elapsed, 2),
        "params": req.model_dump(),
    }


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


# Figures and the denoising GIF are served from /results
app.mount("/results", StaticFiles(directory=str(BASE / "results")), name="results")
