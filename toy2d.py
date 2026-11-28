"""Diffusion from scratch in 2D: the whole DDPM recipe on toy point clouds.

Same algorithm as the MNIST model, with a tiny MLP instead of a U-Net and
2-D points instead of images, so every step can be plotted.

    python toy2d.py                          # 3 shapes x 4 schedules, ~15 min on a CPU
    python toy2d.py --datasets swiss_roll    # one shape, ~5 min
    python toy2d.py --steps 8000             # quicker, blurrier samples

At 8000 steps the linear schedule still gives a blurry swiss roll (mean
sample-to-data distance 0.059); 30000 steps brings it to about 0.027.

Outputs
    results/toy2d_reverse.png       noise -> swiss roll in 100 reverse steps
    results/toy2d_schedules.png     4 schedules on the swiss roll
    results/toy2d_shapes.png        3 shapes, linear vs cosine
    results/toy2d_ddim.png          DDPM 100 steps vs DDIM 20 and 10 steps
    results/toy2d_metrics.json      sample quality for every run
    frontend/public/models/toy2d/<shape>.json   weights for the 2D Playground tab
"""
import argparse
import base64
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

T = 100
N_POINTS = 2000
HIDDEN = 128
SEED = 0
BETA_START = 1e-4
DDIM_STEPS = (20, 10)
X0_CLIP = 3.0   # all three shapes fit inside [-2, 2]

# Linear schedules differ only in beta_T. Cosine follows Nichol & Dhariwal
# (2021): alpha_bar_t = cos^2(((t/T + s) / (1 + s)) * pi/2), betas capped.
SCHEDULES = {
    "gentle": ("linear", 0.02),      # alpha_bar_T = 0.36: never reaches noise
    "linear": ("linear", 0.2),       # reaches noise at the last steps
    "aggressive": ("linear", 0.5),   # data gone by step ~50
    "cosine": ("cosine", 0.008),     # signal falls off evenly, reaches noise
}
DATASETS = ("swiss_roll", "moons", "gaussians8")

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
WEB_DIR = os.path.join(ROOT, "frontend", "public", "models", "toy2d")


# ---------------------------------------------------------------- data

def swiss_roll(n, rng):
    """2-D swiss roll scaled to roughly [-1.4, 1.4]."""
    u = 1.5 * np.pi * (1 + 2 * rng.random(n))
    pts = np.stack([u * np.cos(u), u * np.sin(u)], axis=1)
    pts += 0.25 * rng.standard_normal(pts.shape)
    return (pts / 10.0).astype(np.float32)


def moons(n, rng):
    """Two interleaving half circles, centred, roughly [-1.5, 1.5] x [-0.8, 0.8]."""
    k = n // 2
    a = np.pi * rng.random(k)
    b = np.pi * rng.random(n - k)
    upper = np.stack([np.cos(a), np.sin(a)], axis=1)
    lower = np.stack([1 - np.cos(b), 0.5 - np.sin(b)], axis=1)
    pts = np.concatenate([upper, lower]) - np.array([0.5, 0.25])
    pts = pts * 1.2 + 0.06 * rng.standard_normal((n, 2))
    return pts[rng.permutation(n)].astype(np.float32)


GAUSS_RADIUS = 1.6
GAUSS_STD = 0.1
GAUSS_CENTRES = np.array([[GAUSS_RADIUS * math.cos(k * math.pi / 4),
                           GAUSS_RADIUS * math.sin(k * math.pi / 4)] for k in range(8)],
                         dtype=np.float32)


def gaussians8(n, rng):
    """8 small Gaussians on a circle: a mode-coverage test."""
    idx = rng.integers(0, 8, n)
    return (GAUSS_CENTRES[idx] + GAUSS_STD * rng.standard_normal((n, 2))).astype(np.float32)


MAKERS = {"swiss_roll": swiss_roll, "moons": moons, "gaussians8": gaussians8}
LABELS = {"swiss_roll": "Swiss roll", "moons": "Two moons", "gaussians8": "8 Gaussians"}


# ---------------------------------------------------------------- schedule

def make_schedule(name_or_beta_end):
    """Accepts a schedule name, or a float beta_T for a linear schedule."""
    if isinstance(name_or_beta_end, str):
        kind, value = SCHEDULES[name_or_beta_end]
    else:
        kind, value = "linear", name_or_beta_end
    if kind == "linear":
        betas = torch.linspace(BETA_START, value, T, dtype=torch.float64)
    else:
        s = value
        steps = torch.arange(T + 1, dtype=torch.float64) / T
        f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        ab = f / f[0]
        betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas.float(), alphas.float(), alpha_bar.float()


# ---------------------------------------------------------------- model

class MLP(nn.Module):
    """eps_theta(x_t, t): input (x, y, t/T), 3 -> 128 -> 128 -> 2."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 2),
        )

    def forward(self, x, t):
        tt = (t.float() / T).unsqueeze(1)
        return self.net(torch.cat([x, tt], dim=1))


def train(data, schedule, steps):
    torch.manual_seed(SEED)
    _, _, alpha_bar = make_schedule(schedule)
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    x0 = torch.from_numpy(data)
    losses = []
    for _ in range(steps):
        t = torch.randint(0, T, (len(x0),))
        eps = torch.randn_like(x0)
        ab = alpha_bar[t].unsqueeze(1)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps        # forward, one jump
        loss = ((model(xt, t) - eps) ** 2).mean()           # predict the noise
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    return model, losses


# ---------------------------------------------------------------- sampling
# Frame k holds x_k: frame T is pure noise, frame 0 the sample. The model
# index for frame k is k - 1, whose noise level is alpha_bar[k - 1].

@torch.no_grad()
def sample(model, schedule, n, seed, keep=()):
    """DDPM ancestral sampling, all T steps."""
    betas, alphas, alpha_bar = make_schedule(schedule)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g)
    trace = {T: x.clone()}
    for t in reversed(range(T)):
        eps = model(x, torch.full((n,), t))
        x = (x - betas[t] / (1 - alpha_bar[t]).sqrt() * eps) / alphas[t].sqrt()
        if t > 0:
            x = x + betas[t].sqrt() * torch.randn(n, 2, generator=g)
        if t in keep:
            trace[t] = x.clone()
    return x, trace


def ddim_frames(steps):
    """Frames visited by DDIM with `steps` model calls, from T down to 0."""
    return [round(T - i * T / steps) for i in range(steps + 1)]


@torch.no_grad()
def sample_ddim(model, schedule, n, seed, steps):
    """Deterministic DDIM (eta = 0) from the same starting noise as DDPM."""
    _, _, alpha_bar = make_schedule(schedule)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g)
    frames = ddim_frames(steps)
    for k, k_next in zip(frames[:-1], frames[1:]):
        ab = alpha_bar[k - 1]
        ab_next = alpha_bar[k_next - 1] if k_next > 0 else torch.tensor(1.0)
        eps = model(x, torch.full((n,), k - 1))
        # At t = T the cosine schedule keeps sqrt(alpha_bar) ~ 5e-4 of the
        # signal, so x0 = (x - ...) / sqrt(alpha_bar) multiplies any error in
        # eps by ~2000. Clip x0 to the data range, as most DDIM code does.
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-X0_CLIP, X0_CLIP)
        x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
    return x


# ---------------------------------------------------------------- metrics

def nn_dist(a, b):
    """Mean distance from each point in a to its nearest point in b."""
    return torch.cdist(a, b).min(dim=1).values.mean().item()


def quality(samples, data, dataset):
    d = torch.from_numpy(data)
    q = {"sample_to_data": round(nn_dist(samples, d), 4),   # on the shape?
         "data_to_sample": round(nn_dist(d, samples), 4)}   # whole shape covered?
    if dataset == "gaussians8":
        dist = torch.cdist(samples, torch.from_numpy(GAUSS_CENTRES))
        nearest = dist.min(dim=1)
        close = nearest.values < 3 * GAUSS_STD
        counts = torch.bincount(nearest.indices[close], minlength=8)
        q["modes_covered"] = int((counts >= 0.02 * len(samples)).sum())
        q["within_3std"] = round(close.float().mean().item(), 3)
    return q


# ---------------------------------------------------------------- export

def b64_f32(t):
    return base64.b64encode(t.detach().numpy().astype("<f4").tobytes()).decode()


def export_model(model):
    layers = [m for m in model.net if isinstance(m, nn.Linear)]
    return [{"in": l.in_features, "out": l.out_features,
             "W": b64_f32(l.weight), "b": b64_f32(l.bias)} for l in layers]


def scatter(ax, pts, title, color="#1f4e8c"):
    p = pts.numpy() if torch.is_tensor(pts) else pts
    ax.scatter(p[:, 0], p[:, 1], s=2, alpha=0.6, color=color, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.6, 2.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def score_line(m):
    s = f"sample->data {m['sample_to_data']:.3f}"
    if "modes_covered" in m:
        s += f", modes {m['modes_covered']}/8"
    return s


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(WEB_DIR, exist_ok=True)

    metrics = {"steps": args.steps, "T": T, "n_points": N_POINTS}
    runs = {}   # (dataset, schedule) -> (model, samples by sampler)
    datas = {}
    for dataset in args.datasets:
        data = MAKERS[dataset](N_POINTS, np.random.default_rng(SEED))
        datas[dataset] = data
        ref = MAKERS[dataset](N_POINTS, np.random.default_rng(99))
        metrics[dataset] = {"reference_real_vs_real": quality(torch.from_numpy(ref), data, dataset)}
        web = {}
        for name in SCHEDULES:
            t0 = time.time()
            model, losses = train(data, name, args.steps)
            secs = time.time() - t0
            betas, _, ab = make_schedule(name)
            ddpm, _ = sample(model, name, N_POINTS, seed=1)
            out = {"ddpm": ddpm}
            m = {
                "alpha_bar_T": float(f"{ab[-1].item():.3g}"),
                "final_loss_avg_last_500": round(float(np.mean(losses[-500:])), 4),
                "train_seconds": round(secs, 1),
                "ddpm_100": quality(ddpm, data, dataset),
            }
            for s in DDIM_STEPS:
                x = sample_ddim(model, name, N_POINTS, seed=1, steps=s)
                out[f"ddim_{s}"] = x
                m[f"ddim_{s}"] = quality(x, data, dataset)
            metrics[dataset][name] = m
            runs[(dataset, name)] = (model, out)
            print(dataset, name, json.dumps(m), flush=True)
            web[name] = {"betas": b64_f32(betas), "layers": export_model(model), **m}

        with open(os.path.join(WEB_DIR, f"{dataset}.json"), "w") as f:
            json.dump({"dataset": dataset, "T": T, "hidden": HIDDEN, "activation": "relu",
                       "trainSteps": args.steps, "ddimSteps": list(DDIM_STEPS), "x0Clip": X0_CLIP,
                       "data": b64_f32(torch.from_numpy(data)),
                       "reference": metrics[dataset]["reference_real_vs_real"],
                       "schedules": web}, f, separators=(",", ":"))

    with open(os.path.join(RESULTS, "toy2d_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if "swiss_roll" in args.datasets:
        plot_reverse(runs[("swiss_roll", "linear")][0])
        plot_schedules(datas["swiss_roll"], metrics["swiss_roll"], runs)
    if set(args.datasets) == set(DATASETS):
        plot_shapes(datas, metrics, runs)
        plot_ddim(metrics, runs)


def plot_reverse(model):
    keep = (75, 50, 25, 10, 0)
    _, trace = sample(model, "linear", N_POINTS, seed=1, keep=keep)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.2))
    for ax, t in zip(axes.flat, (T,) + keep):
        label = "t = 100 (pure noise)" if t == T else ("t = 0 (samples)" if t == 0 else f"t = {t}")
        scatter(ax, trace[t], label, "#ff5a4a" if t == 0 else "#1f4e8c")
    fig.suptitle(f"Reverse diffusion: {N_POINTS} points, noise to swiss roll in {T} steps",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "toy2d_reverse.png"), dpi=130)
    plt.close(fig)


def plot_schedules(data, m, runs):
    fig, axes = plt.subplots(1, 6, figsize=(20, 3.6))
    scatter(axes[0], data, "Training data", "#444444")
    for ax, name in zip(axes[1:5], SCHEDULES):
        scatter(ax, runs[("swiss_roll", name)][1]["ddpm"],
                f"{name}\n{score_line(m[name]['ddpm_100'])}")
    ax = axes[5]
    for name in SCHEDULES:
        _, _, ab = make_schedule(name)
        ax.plot(range(1, T + 1), ab.sqrt().numpy(), label=name)
    ax.set_title("Signal kept: sqrt(alpha_bar_t)", fontsize=10)
    ax.set_xlabel("t")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "toy2d_schedules.png"), dpi=130)
    plt.close(fig)


def plot_shapes(datas, metrics, runs):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for row, dataset in zip(axes, DATASETS):
        scatter(row[0], datas[dataset], f"{LABELS[dataset]}: data", "#444444")
        for ax, name in zip(row[1:], ("linear", "cosine")):
            scatter(ax, runs[(dataset, name)][1]["ddpm"],
                    f"{name}, DDPM 100\n{score_line(metrics[dataset][name]['ddpm_100'])}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "toy2d_shapes.png"), dpi=130)
    plt.close(fig)


def plot_ddim(metrics, runs):
    cols = [("ddpm", "ddpm_100", "DDPM, 100 steps")] + [
        (f"ddim_{s}", f"ddim_{s}", f"DDIM, {s} steps") for s in DDIM_STEPS]
    fig, axes = plt.subplots(len(DATASETS), len(cols), figsize=(10, 10))
    for row, dataset in zip(axes, DATASETS):
        for ax, (key, mkey, label) in zip(row, cols):
            scatter(ax, runs[(dataset, "cosine")][1][key],
                    f"{LABELS[dataset]}, {label}\n{score_line(metrics[dataset]['cosine'][mkey])}")
    fig.suptitle("Cosine schedule: same model and starting noise, fewer steps", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "toy2d_ddim.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
