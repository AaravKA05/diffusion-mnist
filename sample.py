"""Step 6: load a checkpoint and generate sample grids (DDPM or DDIM)."""
import argparse
import time

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from diffusion import T, ddim_sample, device, sample
from model import UNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/ema.pt")
    parser.add_argument("--tag", default="ema")
    parser.add_argument("--ddim", type=int, default=0,
                        help="number of DDIM steps (0 = normal DDPM sampling)")
    args = parser.parse_args()

    torch.manual_seed(0)                          # same starting noise every run

    model = UNet().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    start = time.time()
    if args.ddim > 0:
        imgs = ddim_sample(model, n=64, steps=args.ddim)
        tag = f"{args.tag}_ddim{args.ddim}"
        trace = None
    else:
        steps = [T - 1, 300, 200, 100, 50, 0]
        imgs, trace = sample(model, n=64, trace_at=steps)
        tag = args.tag
    print(f"sampling took {time.time() - start:.1f} s")
    print("sample min/max:", imgs.min().item(), imgs.max().item())

    grid = make_grid((imgs.cpu() + 1) / 2, nrow=8)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.title(tag)
    plt.savefig(f"results/samples_{tag}.png", bbox_inches="tight")

    if trace is not None:                         # reverse strip only for DDPM
        fig, axes = plt.subplots(4, len(steps), figsize=(10, 7))
        for col, ti in enumerate(steps):
            for row in range(4):
                axes[row, col].imshow(
                    ((trace[ti][row, 0].cpu() + 1) / 2).clamp(0, 1), cmap="gray"
                )
                axes[row, col].axis("off")
            axes[0, col].set_title(f"t = {ti}")
        plt.savefig(f"results/reverse_strip_{tag}.png", bbox_inches="tight")
    plt.show()