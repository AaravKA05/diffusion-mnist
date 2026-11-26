"""Two analysis tools built on the trained conditional model (no retraining).

1. recognition_timeline: at every DDIM step, take the model's current guess of
   the clean image (x0-hat) and ask the CNN classifier what digit it sees.
   This shows *when* during sampling the digit becomes recognisable.

2. sketch_to_digit (SDEdit, Meng et al. 2021): take a rough user sketch, add
   noise up to an intermediate timestep, then run the reverse process from
   there. Low strength keeps the sketch's shape; high strength lets the model
   redraw it more freely.
"""

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from diffusion import T, alpha_bar, device, q_sample

# --------------------------------------------------
# Shared helpers
# --------------------------------------------------


def guided_eps(model, x, tt, labels, w):
    """Classifier-free guided noise prediction.

    labels=None means "no label" (the null class the model saw during
    label dropout), so the model draws whatever digit it likes.
    """
    null = torch.full((x.size(0),), model.num_classes, device=device, dtype=torch.long)

    if labels is None:
        return model(x, tt, null)

    eps_c = model(x, tt, labels)

    if w == 1.0:
        return eps_c

    eps_u = model(x, tt, null)
    return eps_u + w * (eps_c - eps_u)


def ddim_timesteps(t_start, steps):
    """Evenly spaced timesteps from t_start down to 0 (no duplicates)."""
    ts = torch.linspace(0, t_start, steps).round().long().tolist()
    return sorted(set(ts), reverse=True)


def ddim_step(x, eps, t, t_prev):
    """One deterministic DDIM update. Returns (x_prev, x0_hat)."""
    ab = alpha_bar[t]
    ab_prev = alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

    x0 = ((x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)).clamp(-1, 1)
    x_prev = torch.sqrt(ab_prev) * x0 + torch.sqrt(1 - ab_prev) * eps
    return x_prev, x0


# --------------------------------------------------
# 1. Recognition timeline
# --------------------------------------------------


@torch.no_grad()
def recognition_timeline(model, clf, digit, w, steps, count, seed, preview_count=6):
    """Run DDIM and classify x0-hat after every step.

    Returns a dict with, for each step:
      - timestep t
      - target_prob: mean classifier probability of the requested digit
      - match_rate: share of samples whose top prediction is the requested digit
    plus a few x0-hat previews of the first sample and the final images.
    """
    model.eval()
    clf.eval()

    torch.manual_seed(seed)
    x = torch.randn(count, 1, config.IMG_SIZE, config.IMG_SIZE, device=device)
    labels = torch.full((count,), digit, device=device, dtype=torch.long)

    ts = ddim_timesteps(T - 1, steps)
    preview_idx = set(
        torch.linspace(0, len(ts) - 1, min(preview_count, len(ts)))
        .round()
        .long()
        .tolist()
    )

    points = []
    previews = []

    for i, t in enumerate(ts):
        t_prev = ts[i + 1] if i + 1 < len(ts) else -1
        tt = torch.full((count,), t, device=device, dtype=torch.long)

        eps = guided_eps(model, x, tt, labels, w)
        x, x0 = ddim_step(x, eps, t, t_prev)

        probs = F.softmax(clf(x0), dim=1)
        pred = probs.argmax(1)

        points.append(
            {
                "step": i + 1,
                "timestep": int(t),
                "target_prob": round(float(probs[:, digit].mean()), 4),
                "match_rate": round(float((pred == digit).float().mean()), 4),
            }
        )

        if i in preview_idx:
            previews.append(
                {
                    "step": i + 1,
                    "timestep": int(t),
                    "image": x0[0].detach().cpu(),
                    "pred": int(pred[0]),
                    "conf": round(float(probs[0].max()), 3),
                }
            )

    final = x.clamp(-1, 1)
    final_probs = F.softmax(clf(final), dim=1)
    final_conf, final_pred = final_probs.max(1)

    return {
        "points": points,
        "lock_in_step": lock_in_step(points),
        "previews": previews,
        "final_images": final.detach().cpu(),
        "final_pred": final_pred.cpu().tolist(),
        "final_conf": [round(float(c), 3) for c in final_conf.cpu()],
    }


def lock_in_step(points, threshold=0.9):
    """First step after which match_rate stays >= threshold until the end.

    Returns None if the samples never settle on the requested digit.
    """
    lock = None
    for p in reversed(points):
        if p["match_rate"] >= threshold:
            lock = p["step"]
        else:
            break
    return lock


# --------------------------------------------------
# 2. Sketch to digit (SDEdit)
# --------------------------------------------------


def decode_sketch(image_b64):
    """Turn a canvas PNG (white strokes on black) into an MNIST-style tensor.

    Steps follow how MNIST itself was made: crop the strokes, fit them into a
    20x20 box, centre by centre of mass in a 28x28 image, then resize to
    IMG_SIZE like data.py does. Output is (1, 1, IMG_SIZE, IMG_SIZE) in [-1, 1].
    Raises ValueError if the canvas is empty.
    """
    if "," in image_b64 and image_b64.strip().startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]

    try:
        raw = base64.b64decode(image_b64, validate=False)
        img = Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise ValueError("Could not read the sketch image.") from exc

    # Transparent canvas pixels count as black background
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(background, img.convert("RGBA"))

    arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    # If the user drew dark strokes on a light background, flip it
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    mask = arr > 0.1
    if not mask.any():
        raise ValueError("The sketch is empty. Draw a digit first.")

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    crop = arr[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    small = Image.fromarray((crop * 255).astype(np.uint8)).resize(
        (new_w, new_h), Image.LANCZOS
    )
    small = np.asarray(small, dtype=np.float32) / 255.0

    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = small

    # Shift so the centre of mass sits in the middle, like MNIST
    total = canvas.sum()
    if total > 0:
        ys, xs = np.indices(canvas.shape)
        cy, cx = (ys * canvas).sum() / total, (xs * canvas).sum() / total
        dy, dx = int(round(13.5 - cy)), int(round(13.5 - cx))
        canvas = np.roll(canvas, (dy, dx), axis=(0, 1))

    pil = Image.fromarray((canvas.clip(0, 1) * 255).astype(np.uint8)).resize(
        (config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR
    )
    x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0)
    x = x * 2 - 1
    return x.view(1, 1, config.IMG_SIZE, config.IMG_SIZE).to(device)


@torch.no_grad()
def sketch_to_digit(model, clf, sketch, digit, strength, w, steps, count, seed):
    """SDEdit: noise the sketch to t_start, then denoise back to t=0.

    digit=None lets the model decide (unconditional). strength in (0, 1]
    sets t_start = strength * (T - 1); the number of DDIM steps actually used
    is scaled the same way so a weak edit stays quick.
    """
    model.eval()
    clf.eval()

    torch.manual_seed(seed)

    t_start = max(1, int(round(strength * (T - 1))))
    used_steps = max(2, int(round(steps * strength)))
    ts = ddim_timesteps(t_start, used_steps)

    x0 = sketch.expand(count, -1, -1, -1)
    noise = torch.randn_like(x0)
    t_tensor = torch.full((count,), t_start, device=device, dtype=torch.long)
    x = q_sample(x0, t_tensor, noise)
    noised = x[0].clamp(-1, 1).detach().cpu()

    labels = (
        None
        if digit is None
        else torch.full((count,), digit, device=device, dtype=torch.long)
    )

    for i, t in enumerate(ts):
        t_prev = ts[i + 1] if i + 1 < len(ts) else -1
        tt = torch.full((count,), t, device=device, dtype=torch.long)
        eps = guided_eps(model, x, tt, labels, w)
        x, _ = ddim_step(x, eps, t, t_prev)

    out = x.clamp(-1, 1)
    probs = F.softmax(clf(out), dim=1)
    conf, pred = probs.max(1)

    sketch_probs = F.softmax(clf(sketch), dim=1)[0]

    return {
        "images": out.detach().cpu(),
        "pred": pred.cpu().tolist(),
        "conf": [round(float(c), 3) for c in conf.cpu()],
        "noised": noised,
        "sketch_pred": int(sketch_probs.argmax()),
        "sketch_conf": round(float(sketch_probs.max()), 3),
        "t_start": t_start,
        "steps_used": len(ts),
    }
