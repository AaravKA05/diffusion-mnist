"""Diffusion from scratch in 2D: the whole DDPM recipe on a swiss roll.

Same algorithm as the MNIST model, with a tiny MLP instead of a U-Net and
2-D points instead of images, so every step can be plotted.

    python toy2d.py                # 3 noise schedules, ~1-2 min each on a CPU
    python toy2d.py --steps 8000   # quicker, blurrier samples

At 8000 steps the linear schedule still gives a blurry roll (mean
sample-to-data distance 0.059); 30000 steps brings it to about 0.027.

Outputs
    results/toy2d_reverse.png     noise -> swiss roll in 100 reverse steps
    results/toy2d_schedules.png   gentle vs linear vs aggressive schedule
    results/toy2d_metrics.json    sample quality per schedule
    frontend/public/models/toy2d.json   weights for the 2D Playground tab
"""
import argparse
import base64
import json
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

# beta_1 -> beta_T, all linear. Only the end value changes.
SCHEDULES = {
    "gentle": 0.02,      # alpha_bar_T = 0.36: never reaches pure noise
    "linear": 0.2,       # alpha_bar_T = 2e-5: reaches noise at the last steps
    "aggressive": 0.5,   # data gone by step ~50: half the steps are wasted
}
BETA_START = 1e-4

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
WEB_OUT = os.path.join(ROOT, "frontend", "public", "models", "toy2d.json")


def swiss_roll(n, rng):
    """2-D swiss roll scaled to roughly [-1, 1]."""
    u = 1.5 * np.pi * (1 + 2 * rng.random(n))
    pts = np.stack([u * np.cos(u), u * np.sin(u)], axis=1)
    pts += 0.25 * rng.standard_normal(pts.shape)
    return (pts / 10.0).astype(np.float32)


def make_schedule(beta_end):
    betas = torch.linspace(BETA_START, beta_end, T)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


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


def train(data, beta_end, steps):
    torch.manual_seed(SEED)
    _, _, alpha_bar = make_schedule(beta_end)
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


@torch.no_grad()
def sample(model, beta_end, n, seed, keep=()):
    betas, alphas, alpha_bar = make_schedule(beta_end)
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


def nn_dist(a, b):
    """Mean distance from each point in a to its nearest point in b."""
    return torch.cdist(a, b).min(dim=1).values.mean().item()


def quality(samples, data):
    d = torch.from_numpy(data)
    precision = nn_dist(samples, d)   # are samples on the roll?
    coverage = nn_dist(d, samples)    # is the whole roll covered?
    return {"sample_to_data": round(precision, 4),
            "data_to_sample": round(coverage, 4)}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(SEED)
    data = swiss_roll(N_POINTS, rng)

    models, metrics, web = {}, {}, {}
    for name, beta_end in SCHEDULES.items():
        t0 = time.time()
        model, losses = train(data, beta_end, args.steps)
        secs = time.time() - t0
        x, _ = sample(model, beta_end, N_POINTS, seed=1)
        _, _, ab = make_schedule(beta_end)
        metrics[name] = {
            "beta_end": beta_end,
            "alpha_bar_T": float(f"{ab[-1].item():.3g}"),
            "final_loss_avg_last_500": round(float(np.mean(losses[-500:])), 4),
            "train_seconds": round(secs, 1),
            **quality(x, data),
        }
        print(name, metrics[name])
        models[name] = (model, beta_end, x)
        web[name] = {"betaStart": BETA_START, "betaEnd": beta_end,
                     "layers": export_model(model), **metrics[name]}

    # Reference: real data vs a second real draw (best possible score).
    metrics["reference_real_vs_real"] = quality(
        torch.from_numpy(swiss_roll(N_POINTS, np.random.default_rng(99))), data)
    print("reference", metrics["reference_real_vs_real"])

    # 1. Reverse process for the main schedule, like the slide.
    model, beta_end, _ = models["linear"]
    keep = (75, 50, 25, 10, 0)
    _, trace = sample(model, beta_end, N_POINTS, seed=1, keep=keep)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.2))
    for ax, t in zip(axes.flat, (T,) + keep):
        label = "t = 100 (pure noise)" if t == T else f"t = {t}"
        if t == 0:
            label = "t = 0 (samples)"
        scatter(ax, trace[t], label, "#ff5a4a" if t == 0 else "#1f4e8c")
    fig.suptitle(f"Reverse diffusion: {N_POINTS} points, noise to swiss roll "
                 f"in {T} steps", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "toy2d_reverse.png"), dpi=130)
    plt.close(fig)

    # 2. Schedules side by side, plus how much signal each keeps.
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    scatter(axes[0], data, "Training data", "#444444")
    for ax, name in zip(axes[1:4], SCHEDULES):
        m = metrics[name]
        scatter(ax, models[name][2],
                f"{name} (beta_T={m['beta_end']})\n"
                f"sample->data {m['sample_to_data']:.3f}")
    ax = axes[4]
    for name, beta_end in SCHEDULES.items():
        _, _, ab = make_schedule(beta_end)
        ax.plot(range(1, T + 1), ab.sqrt().numpy(), label=name)
    ax.set_title("Signal kept: sqrt(alpha_bar_t)", fontsize=10)
    ax.set_xlabel("t")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "toy2d_schedules.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(RESULTS, "toy2d_metrics.json"), "w") as f:
        json.dump({"steps": args.steps, "T": T, **metrics}, f, indent=2)

    os.makedirs(os.path.dirname(WEB_OUT), exist_ok=True)
    with open(WEB_OUT, "w") as f:
        json.dump({
            "T": T,
            "hidden": HIDDEN,
            "activation": "relu",
            "trainSteps": args.steps,
            "data": b64_f32(torch.from_numpy(data)),
            "schedules": web,
        }, f, separators=(",", ":"))
    print("wrote", WEB_OUT, os.path.getsize(WEB_OUT) // 1024, "KB")


if __name__ == "__main__":
    main()
