"""Denoising GIF: static -> digits, using the conditional model (DDIM)."""

import argparse

import torch
from PIL import Image
from torchvision.utils import make_grid

import config
from diffusion import T, alpha_bar, device
from model import UNet


def to_pil(x, scale):
    grid = make_grid((x.cpu().clamp(-1, 1) + 1) / 2, nrow=10, padding=1)
    arr = (grid * 255).byte().permute(1, 2, 0).numpy()
    img = Image.fromarray(arr)
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


@torch.no_grad()
def trajectory(model, labels, steps, w, scale):
    model.eval()
    n = labels.size(0)
    null = torch.full_like(labels, model.num_classes)
    x = torch.randn(n, 1, config.IMG_SIZE, config.IMG_SIZE, device=device)
    ts = list(reversed(torch.linspace(0, T - 1, steps).long().tolist()))
    frames = [to_pil(x, scale)]
    for i, t in enumerate(ts):
        t_prev = ts[i + 1] if i + 1 < len(ts) else -1
        tt = torch.full((n,), t, device=device, dtype=torch.long)
        eps = model(x, tt, labels)
        if w != 1.0:
            eps_u = model(x, tt, null)
            eps = eps_u + w * (eps - eps_u)
        ab = alpha_bar[t]
        ab_prev = alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        x0 = ((x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)).clamp(-1, 1)
        x = torch.sqrt(ab_prev) * x0 + torch.sqrt(1 - ab_prev) * eps
        frames.append(to_pil(x, scale))
    return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = UNet(num_classes=10).to(device)
    model.load_state_dict(torch.load("checkpoints/cond_ema.pt", map_location=device))

    labels = torch.arange(10, device=device).repeat(2)  # two rows of 0..9
    frames = trajectory(model, labels, args.steps, args.w, args.scale)

    # hold the final frame for ~2 seconds
    frames.extend([frames[-1]] * 30)
    frames[0].save(
        "results/denoising.gif",
        save_all=True,
        append_images=frames[1:],
        duration=60,
        loop=0,
    )
    print("saved results/denoising.gif with", len(frames), "frames")
