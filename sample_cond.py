"""Class-conditional sampling with classifier-free guidance (DDIM)."""
import argparse

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

import config
from diffusion import T, alpha_bar, device
from model import UNet


@torch.no_grad()
def cond_ddim_sample(model, labels, steps=50, w=1.0):
    """DDIM sampling for the given labels with guidance strength w."""
    model.eval()
    n = labels.size(0)
    null = torch.full_like(labels, model.num_classes)   # "no label"
    x = torch.randn(n, 1, config.IMG_SIZE, config.IMG_SIZE, device=device)
    ts = list(reversed(torch.linspace(0, T - 1, steps).long().tolist()))
    for i, t in enumerate(ts):
        t_prev = ts[i + 1] if i + 1 < len(ts) else -1
        tt = torch.full((n,), t, device=device, dtype=torch.long)
        eps_c = model(x, tt, labels)
        if w == 1.0:
            eps = eps_c
        else:
            eps_u = model(x, tt, null)
            eps = eps_u + w * (eps_c - eps_u)
        ab = alpha_bar[t]
        ab_prev = alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        x0 = ((x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)).clamp(-1, 1)
        x = torch.sqrt(ab_prev) * x0 + torch.sqrt(1 - ab_prev) * eps
    return x.clamp(-1, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/cond_ema.pt")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--w", type=float, default=1.0, help="guidance strength")
    args = parser.parse_args()

    torch.manual_seed(0)
    model = UNet(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    # 10 rows (digits 0-9) x 8 columns
    labels = torch.arange(10, device=device).repeat_interleave(8)
    imgs = cond_ddim_sample(model, labels, steps=args.steps, w=args.w)

    grid = make_grid((imgs.cpu() + 1) / 2, nrow=8)
    plt.figure(figsize=(5, 6.5))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.title(f"conditional, DDIM {args.steps} steps, w={args.w}")
    plt.savefig(f"results/cond_samples_w{args.w}.png", bbox_inches="tight")
    plt.show()