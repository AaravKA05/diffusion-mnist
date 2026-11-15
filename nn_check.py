"""Memorisation check: is each generated digit a copy of a training image?"""

import argparse
import json
import os

import torch
from torchvision.utils import save_image

from data import get_loader
from diffusion import device
from model import UNet
from sample_cond import cond_ddim_sample

ap = argparse.ArgumentParser()
ap.add_argument("--per_digit", type=int, default=10)
ap.add_argument("--w", type=float, default=1.2)
ap.add_argument("--steps", type=int, default=50)
args = ap.parse_args()


def collect(train, limit=None):
    xs, n = [], 0
    for x, _ in get_loader(train=train):
        xs.append(x)
        n += len(x)
        if limit and n >= limit:
            break
    x = torch.cat(xs)
    return x[:limit] if limit else x


def nearest(a, b):
    dist = torch.cdist(a.flatten(1), b.flatten(1))
    return dist.min(1)


model = UNet(num_classes=10).to(device)
model.load_state_dict(torch.load("checkpoints/cond_ema.pt", map_location=device))
model.eval()

torch.manual_seed(0)
gen = []
for digit in range(10):
    labels = torch.full((args.per_digit,), digit, device=device, dtype=torch.long)
    gen.append(cond_ddim_sample(model, labels, steps=args.steps, w=args.w).cpu())
gen = torch.cat(gen)

train = collect(True)
test = collect(False, limit=len(gen))
gd, gi = nearest(gen, train)  # generated -> nearest training image
td, _ = nearest(test, train)  # real test images -> nearest training image (reference)

res = {
    "n_generated": len(gen),
    "w": args.w,
    "generated_median_dist": round(gd.median().item(), 3),
    "test_median_dist": round(td.median().item(), 3),
    "generated_min_dist": round(gd.min().item(), 3),
    "test_min_dist": round(td.min().item(), 3),
    "generated_closer_than_every_test_image": int((gd < td.min()).sum().item()),
}
os.makedirs("results", exist_ok=True)
with open("results/nn_check.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))

sel = torch.arange(16) * len(gen) // 16
pairs = torch.cat([gen[sel], train[gi[sel]]])
save_image(((pairs + 1) / 2).clamp(0, 1), "results/nn_check.png", nrow=16, padding=1)
print(
    "saved results/nn_check.png (top row: generated, bottom row: nearest training image)"
)
