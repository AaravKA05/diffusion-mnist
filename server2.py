"""FastAPI backend v3 for DiffusionLab.

Run:
    python -m uvicorn server2:app --host 127.0.0.1 --port 8000

APIs:
    GET  /api/info
    POST /api/generate
    POST /api/compare
    POST /api/evaluate
"""

import base64
import io
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

import config
from diffusion import device
from evaluate import Classifier
from model import UNet
from sample_cond import cond_ddim_sample

BASE = Path(__file__).parent

# --------------------------------------------------
# Load models once at startup
# --------------------------------------------------

model = UNet(num_classes=10).to(device)

model.load_state_dict(
    torch.load(BASE / "checkpoints" / "cond_ema.pt", map_location=device)
)

model.eval()

clf = Classifier().to(device)

clf.load_state_dict(
    torch.load(BASE / "checkpoints" / "classifier.pt", map_location=device)
)

clf.eval()

lock = threading.Lock()

# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(title="DiffusionLab API v3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://diffusion-mnist.vercel.app",
        "https://diffusion-mnist-ah3vhn8js-aaravka05.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Request models
# --------------------------------------------------


class GenRequest(BaseModel):
    digit: int = Field(7, ge=0, le=9)
    count: int = Field(8, ge=1, le=16)
    w: float = Field(1.2, ge=0.0, le=3.0)
    steps: int = Field(20, ge=5, le=100)
    seed: int = Field(0, ge=0, le=2**31 - 1)


class CompareRequest(BaseModel):
    digit: int = Field(7, ge=0, le=9)
    count: int = Field(6, ge=1, le=8)
    ws: list[float] = Field(
        default_factory=lambda: [0.0, 1.2, 2.0],
        min_length=2,
        max_length=4,
    )
    steps: int = Field(20, ge=5, le=100)
    seed: int = Field(0, ge=0, le=2**31 - 1)


class EvalRequest(BaseModel):
    digit: int = Field(7, ge=0, le=9)
    count: int = Field(32, ge=2, le=64)
    w: float = Field(1.2, ge=0.0, le=3.0)
    steps: int = Field(20, ge=5, le=100)
    seed: int = Field(42, ge=0, le=2**31 - 1)


# --------------------------------------------------
# Image utilities
# --------------------------------------------------


def tensor_to_png_b64(img, size=192):
    """Convert an image tensor in [-1, 1] into a PNG base64 string."""

    arr = ((img.squeeze(0) + 1) / 2 * 255).round().clamp(0, 255).byte().cpu().numpy()

    pil = Image.fromarray(arr).resize((size, size), Image.BICUBIC)

    buf = io.BytesIO()
    pil.save(buf, format="PNG")

    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------
# Core generation function
# --------------------------------------------------


def run(digit, count, w, steps, seed, size):
    """Generate images and classify every generated image."""

    torch.manual_seed(seed)

    labels = torch.full((count,), digit, device=device, dtype=torch.long)

    with torch.no_grad():
        imgs = cond_ddim_sample(model, labels, steps=steps, w=w)

        probs = F.softmax(clf(imgs), dim=1)

    conf, pred = probs.max(1)

    imgs = imgs.cpu()
    conf = conf.cpu()
    pred = pred.cpu()

    items = [
        {
            "image": tensor_to_png_b64(im, size),
            "pred": int(p),
            "conf": round(float(c), 3),
        }
        for im, p, c in zip(imgs, pred, conf)
    ]

    matches = sum(1 for item in items if item["pred"] == digit)

    return items, matches


# --------------------------------------------------
# Health and model information
# --------------------------------------------------


@app.get("/api/info")
def info():
    return {
        "status": "online",
        "model": "Class-Conditional MNIST Diffusion",
        "img_size": config.IMG_SIZE,
        "device": str(device),
        "classes": 10,
        "evaluation_available": True,
    }


# --------------------------------------------------
# Image generation
# --------------------------------------------------


@app.post("/api/generate")
def generate(req: GenRequest):

    with lock:
        start = time.perf_counter()

        items, matches = run(req.digit, req.count, req.w, req.steps, req.seed, 192)

        elapsed = time.perf_counter() - start

    return {
        "items": items,
        "matches": matches,
        "count": req.count,
        "seconds": round(elapsed, 2),
        "params": req.model_dump(),
    }


# --------------------------------------------------
# Guidance comparison
# --------------------------------------------------


@app.post("/api/compare")
def compare(req: CompareRequest):

    ws = [min(max(float(w), 0.0), 3.0) for w in req.ws]

    with lock:
        start = time.perf_counter()

        results = []

        for w in ws:

            items, matches = run(req.digit, req.count, w, req.steps, req.seed, 128)

            results.append(
                {
                    "w": w,
                    "matches": matches,
                    "items": items,
                }
            )

        elapsed = time.perf_counter() - start

    return {
        "digit": req.digit,
        "count": req.count,
        "seconds": round(elapsed, 2),
        "results": results,
    }


# --------------------------------------------------
# Model evaluation
# --------------------------------------------------


@app.post("/api/evaluate")
def evaluate_model(req: EvalRequest):

    with lock:

        start = time.perf_counter()

        torch.manual_seed(req.seed)

        labels = torch.full((req.count,), req.digit, device=device, dtype=torch.long)

        with torch.no_grad():

            imgs = cond_ddim_sample(model, labels, steps=req.steps, w=req.w)

            probs = F.softmax(clf(imgs), dim=1)

            conf, pred = probs.max(1)

        matches = int((pred == req.digit).sum().item())

        class_consistency = (matches / req.count) * 100

        flat = imgs.detach().flatten(1)

        diversity = float(torch.pdist(flat, p=1).mean().item() / flat.shape[1])

        pixel_std = float(imgs.detach().std(dim=0).mean().item())

        mean_confidence = float(conf.mean().item())

        items = [
            {
                "image": tensor_to_png_b64(img.cpu(), 128),
                "pred": int(p),
                "conf": round(float(c), 3),
            }
            for img, p, c in zip(imgs, pred, conf)
        ]

        elapsed = time.perf_counter() - start

    return {
        "digit": req.digit,
        "count": req.count,
        "matches": matches,
        "class_consistency": round(class_consistency, 2),
        "mean_confidence": round(mean_confidence, 4),
        "diversity_mean_pairwise_l1": round(diversity, 4),
        "mean_pixel_std": round(pixel_std, 4),
        "seconds": round(elapsed, 2),
        "images_per_second": round(req.count / elapsed, 2) if elapsed > 0 else 0,
        "items": items,
        "params": req.model_dump(),
    }


# --------------------------------------------------
# Original demo
# --------------------------------------------------


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index2.html")


# --------------------------------------------------
# Results directory
# --------------------------------------------------


app.mount("/results", StaticFiles(directory=str(BASE / "results")), name="results")
