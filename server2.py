"""FastAPI backend v3 for DiffusionLab.

Run:
    python -m uvicorn server2:app --host 127.0.0.1 --port 8000

APIs:
    GET  /api/info
    POST /api/generate
    POST /api/compare
    POST /api/evaluate
    POST /api/visualize
    POST /api/recognition   (when does the digit become recognisable?)
    POST /api/sketch        (sketch to digit, SDEdit)
"""

import base64
import io
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

import config
from diffusion import device, T
from evaluate import Classifier
from lab_features import decode_sketch, recognition_timeline, sketch_to_digit
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


class VisualizeRequest(BaseModel):
    digit: int = Field(7, ge=0, le=9)
    w: float = Field(1.2, ge=0.0, le=3.0)
    steps: int = Field(20, ge=5, le=100)
    seed: int = Field(42, ge=0, le=2**31 - 1)
    snapshot_count: int = Field(8, ge=4, le=12)


class InterpolateRequest(BaseModel):
    digit: int = Field(7, ge=0, le=9)
    w: float = Field(1.2, ge=0.0, le=3.0)
    steps: int = Field(30, ge=5, le=100)
    seed_a: int = Field(42, ge=0, le=2**31 - 1)
    seed_b: int = Field(123, ge=0, le=2**31 - 1)
    frames: int = Field(9, ge=3, le=15)


class ExperimentRequest(BaseModel):
    experiment_type: str = Field("guidance", pattern="^(guidance|steps|seeds)$")
    digit: int = Field(7, ge=0, le=9)
    count: int = Field(4, ge=1, le=8)
    values: list[float] = Field(min_length=2, max_length=4)
    w: float = Field(1.2, ge=0.0, le=3.0)
    steps: int = Field(30, ge=5, le=100)
    seed: int = Field(42, ge=0, le=2**31 - 1)


class RepresentationRequest(BaseModel):
    digit: int = Field(7, ge=0, le=9)
    timestep: int = Field(200, ge=0, le=399)
    steps: int = Field(50, ge=5, le=100)
    w: float = Field(1.2, ge=0.0, le=3.0)
    seed: int = Field(42, ge=0, le=2**31 - 1)


class RepresentationEvolutionRequest(BaseModel):
    digit: int = Field(7, ge=0, le=9)
    timesteps: list[int] = Field(
        default=[350, 250, 150, 50],
        min_length=2,
        max_length=6,
    )
    steps: int = Field(50, ge=5, le=100)
    w: float = Field(1.2, ge=0.0, le=3.0)
    seed: int = Field(42, ge=0, le=2**31 - 1)
    channel: int = Field(0, ge=0, le=255)


class RecognitionRequest(BaseModel):
    digit: int = Field(7, ge=0, le=9)
    ws: list[float] = Field(
        default_factory=lambda: [0.0, 1.2, 3.0],
        min_length=1,
        max_length=4,
    )
    steps: int = Field(30, ge=5, le=100)
    count: int = Field(8, ge=1, le=16)
    seed: int = Field(42, ge=0, le=2**31 - 1)


class SketchRequest(BaseModel):
    image: str = Field(..., min_length=20, max_length=3_000_000)
    digit: int | None = Field(None, ge=0, le=9)
    strength: float = Field(0.5, ge=0.1, le=1.0)
    w: float = Field(1.2, ge=0.0, le=3.0)
    steps: int = Field(50, ge=5, le=100)
    count: int = Field(4, ge=1, le=8)
    seed: int = Field(42, ge=0, le=2**31 - 1)


# --------------------------------------------------
# Image utilities
# --------------------------------------------------


def tensor_to_png_b64(img, size=192):
    """Convert an image tensor in [-1, 1] into PNG base64."""

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
        "visualization_available": True,
        "recognition_available": True,
        "sketch_available": True,
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
# Experiment Lab
# --------------------------------------------------


@app.post("/api/experiment")
def run_experiment(req: ExperimentRequest):

    if req.experiment_type == "guidance":
        values = [min(max(float(v), 0.0), 3.0) for v in req.values]

    elif req.experiment_type == "steps":
        values = [min(max(int(v), 5), 100) for v in req.values]

    else:
        values = [min(max(int(v), 0), 2**31 - 1) for v in req.values]

    with lock:

        total_start = time.perf_counter()
        results = []

        for value in values:

            if req.experiment_type == "guidance":
                w = value
                steps = req.steps
                seed = req.seed
                parameter = "guidance"

            elif req.experiment_type == "steps":
                w = req.w
                steps = int(value)
                seed = req.seed
                parameter = "steps"

            else:
                w = req.w
                steps = req.steps
                seed = int(value)
                parameter = "seed"

            start = time.perf_counter()

            items, matches = run(req.digit, req.count, w, steps, seed, 128)

            elapsed = time.perf_counter() - start

            avg_confidence = (
                sum(item["conf"] for item in items) / len(items) if items else 0
            )

            results.append(
                {
                    "parameter": parameter,
                    "value": value,
                    "matches": matches,
                    "match_rate": round(matches / len(items) * 100, 2) if items else 0,
                    "avg_confidence": round(avg_confidence, 3),
                    "seconds": round(elapsed, 2),
                    "items": items,
                }
            )

        total_elapsed = time.perf_counter() - total_start

    return {
        "experiment_type": req.experiment_type,
        "digit": req.digit,
        "count": req.count,
        "total_seconds": round(total_elapsed, 2),
        "results": results,
    }


# --------------------------------------------------
# Diffusion visualization
# --------------------------------------------------


@app.post("/api/interpolate")
def interpolate(req: InterpolateRequest):
    with lock:
        start = time.perf_counter()

        labels = torch.full((1,), req.digit, device=device, dtype=torch.long)

        generator_a = torch.Generator(device=device).manual_seed(req.seed_a)
        generator_b = torch.Generator(device=device).manual_seed(req.seed_b)

        shape = (1, 1, config.IMG_SIZE, config.IMG_SIZE)

        noise_a = torch.randn(shape, generator=generator_a, device=device)
        noise_b = torch.randn(shape, generator=generator_b, device=device)

        alphas = torch.linspace(0, 1, req.frames).tolist()
        results = []

        for alpha in alphas:
            noise = (1 - alpha) * noise_a + alpha * noise_b

            images = cond_ddim_sample(
                model,
                labels,
                steps=req.steps,
                w=req.w,
                initial_noise=noise,
            )

            image = images[0].detach().cpu()
            image = ((image + 1) / 2).clamp(0, 1)
            image = (image.squeeze().numpy() * 255).astype("uint8")

            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="PNG")

            results.append(
                {
                    "alpha": round(alpha, 3),
                    "image": base64.b64encode(buffer.getvalue()).decode("utf-8"),
                }
            )

        return {
            "digit": req.digit,
            "frames": results,
            "count": len(results),
            "inference_time": round(time.perf_counter() - start, 3),
        }


@app.post("/api/visualize")
def visualize(req: VisualizeRequest):

    with lock:

        start = time.perf_counter()

        torch.manual_seed(req.seed)

        labels = torch.full((1,), req.digit, device=device, dtype=torch.long)

        with torch.no_grad():

            imgs, snapshots = cond_ddim_sample(
                model,
                labels,
                steps=req.steps,
                w=req.w,
                return_intermediates=True,
                snapshot_count=req.snapshot_count,
            )

        frames = [
            {
                "timestep": frame["timestep"],
                "image": tensor_to_png_b64(frame["image"][0], 192),
            }
            for frame in snapshots
        ]

        elapsed = time.perf_counter() - start

    return {
        "digit": req.digit,
        "steps": req.steps,
        "guidance": req.w,
        "seed": req.seed,
        "seconds": round(elapsed, 2),
        "snapshots": frames,
    }


@app.post("/api/representation")
def get_representation(req: RepresentationRequest):

    with lock:
        start = time.perf_counter()

        torch.manual_seed(req.seed)

        labels = torch.tensor([req.digit], device=device, dtype=torch.long)

        ts = list(reversed(torch.linspace(0, T - 1, req.steps).long().tolist()))

        actual_timestep = min(ts, key=lambda value: abs(value - req.timestep))

        with torch.no_grad():
            initial_noise = torch.randn(
                1, 1, config.IMG_SIZE, config.IMG_SIZE, device=device
            )

            _, features = cond_ddim_sample(
                model,
                labels,
                steps=req.steps,
                w=req.w,
                initial_noise=initial_noise,
                capture_timestep=actual_timestep,
            )

        if features is None:
            raise HTTPException(
                status_code=500, detail="Bottleneck feature capture failed."
            )

        feature_map = features[0]
        channels, height, width = feature_map.shape

        channel_maps = torch.nn.functional.interpolate(
            feature_map.unsqueeze(1),
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        channel_maps = channel_maps.cpu()

        channel_stats = {
            "mean": float(feature_map.mean().item()),
            "std": float(feature_map.std().item()),
            "min": float(feature_map.min().item()),
            "max": float(feature_map.max().item()),
        }

        heatmaps = []

        for index in range(channels):
            activation = channel_maps[index]
            low = activation.min()
            high = activation.max()

            normalized = (activation - low) / (high - low + 1e-8)

            heatmaps.append(
                {
                    "channel": index,
                    "image": tensor_to_png_b64(
                        normalized.unsqueeze(0) * 2 - 1,
                        size=192,
                    ),
                }
            )

        return {
            "digit": req.digit,
            "requested_timestep": req.timestep,
            "actual_timestep": actual_timestep,
            "steps": req.steps,
            "channels": channels,
            "feature_shape": list(feature_map.shape),
            "statistics": channel_stats,
            "heatmaps": heatmaps,
            "seconds": round(time.perf_counter() - start, 3),
        }


@app.post("/api/representation/evolution")
def get_representation_evolution(req: RepresentationEvolutionRequest):
    with lock:
        start = time.perf_counter()

        if any(t < 0 or t >= T for t in req.timesteps):
            raise HTTPException(
                status_code=400, detail=f"Timesteps must be between 0 and {T - 1}."
            )

        torch.manual_seed(req.seed)

        labels = torch.tensor([req.digit], device=device, dtype=torch.long)

        ts = list(reversed(torch.linspace(0, T - 1, req.steps).long().tolist()))

        actual_timesteps = sorted(
            set(
                min(ts, key=lambda value: abs(value - requested))
                for requested in req.timesteps
            ),
            reverse=True,
        )

        with torch.no_grad():
            initial_noise = torch.randn(
                1,
                1,
                config.IMG_SIZE,
                config.IMG_SIZE,
                device=device,
            )

            _, captured = cond_ddim_sample(
                model,
                labels,
                steps=req.steps,
                w=req.w,
                initial_noise=initial_noise,
                capture_timesteps=actual_timesteps,
            )

        if not captured or any(t not in captured for t in actual_timesteps):
            raise HTTPException(
                status_code=500,
                detail="One or more timestep features could not be captured.",
            )

        feature_maps = [captured[t][0, req.channel] for t in actual_timesteps]

        resized_maps = [
            torch.nn.functional.interpolate(
                feature.unsqueeze(0).unsqueeze(0),
                size=(64, 64),
                mode="bilinear",
                align_corners=False,
            )[0, 0].cpu()
            for feature in feature_maps
        ]

        global_min = min(float(f.min()) for f in resized_maps)
        global_max = max(float(f.max()) for f in resized_maps)

        evolution = []

        for timestep, activation in zip(actual_timesteps, resized_maps):
            normalized = (activation - global_min) / (global_max - global_min + 1e-8)

            evolution.append(
                {
                    "timestep": timestep,
                    "image": tensor_to_png_b64(
                        normalized.unsqueeze(0) * 2 - 1,
                        size=192,
                    ),
                    "mean": float(activation.mean()),
                    "std": float(activation.std()),
                }
            )

        return {
            "digit": req.digit,
            "channel": req.channel,
            "requested_timesteps": req.timesteps,
            "actual_timesteps": actual_timesteps,
            "steps": req.steps,
            "seed": req.seed,
            "guidance": req.w,
            "feature_shape": list(captured[actual_timesteps[0]][0].shape),
            "activation_range": {
                "min": global_min,
                "max": global_max,
            },
            "evolution": evolution,
            "seconds": round(time.perf_counter() - start, 3),
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
        "images_per_second": (round(req.count / elapsed, 2) if elapsed > 0 else 0),
        "items": items,
        "params": req.model_dump(),
    }


# --------------------------------------------------
# Recognition timeline
# --------------------------------------------------


@app.post("/api/recognition")
def recognition(req: RecognitionRequest):

    for w in req.ws:
        if not 0.0 <= w <= 3.0:
            raise HTTPException(
                status_code=422, detail="Each w must be between 0 and 3."
            )

    with lock:

        start = time.perf_counter()

        series = []

        for w in req.ws:
            result = recognition_timeline(
                model, clf, req.digit, w, req.steps, req.count, req.seed
            )

            final_matches = sum(1 for p in result["final_pred"] if p == req.digit)

            series.append(
                {
                    "w": w,
                    "points": result["points"],
                    "lock_in_step": result["lock_in_step"],
                    "final_matches": final_matches,
                    "previews": [
                        {
                            "step": pv["step"],
                            "timestep": pv["timestep"],
                            "image": tensor_to_png_b64(pv["image"], 128),
                            "pred": pv["pred"],
                            "conf": pv["conf"],
                        }
                        for pv in result["previews"]
                    ],
                }
            )

        elapsed = time.perf_counter() - start

    return {
        "digit": req.digit,
        "steps": len(series[0]["points"]),
        "count": req.count,
        "seed": req.seed,
        "seconds": round(elapsed, 2),
        "series": series,
    }


# --------------------------------------------------
# Sketch to digit (SDEdit)
# --------------------------------------------------


@app.post("/api/sketch")
def sketch(req: SketchRequest):

    try:
        sketch_tensor = decode_sketch(req.image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    with lock:

        start = time.perf_counter()

        result = sketch_to_digit(
            model,
            clf,
            sketch_tensor,
            req.digit,
            req.strength,
            req.w,
            req.steps,
            req.count,
            req.seed,
        )

        elapsed = time.perf_counter() - start

    items = [
        {"image": tensor_to_png_b64(im, 192), "pred": p, "conf": c}
        for im, p, c in zip(result["images"], result["pred"], result["conf"])
    ]

    return {
        "input": tensor_to_png_b64(sketch_tensor[0].cpu(), 192),
        "noised": tensor_to_png_b64(result["noised"], 192),
        "sketch_pred": result["sketch_pred"],
        "sketch_conf": result["sketch_conf"],
        "items": items,
        "matches": (
            sum(1 for item in items if item["pred"] == req.digit)
            if req.digit is not None
            else None
        ),
        "t_start": result["t_start"],
        "steps_used": result["steps_used"],
        "seconds": round(elapsed, 2),
        "params": {k: v for k, v in req.model_dump().items() if k != "image"},
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
