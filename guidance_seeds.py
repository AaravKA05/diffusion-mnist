"""Repeat the guidance comparison over several seeds (does guidance really help?)."""

import argparse
import json
import os
import statistics
import time

import torch

from diffusion import device
from evaluate import Classifier
from model import UNet
from sample_cond import cond_ddim_sample

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
ap.add_argument("--ws", type=float, nargs="+", default=[1.0, 1.2])
ap.add_argument("--per_label", type=int, default=100)
ap.add_argument("--steps", type=int, default=50)
args = ap.parse_args()

model = UNet(num_classes=10).to(device)
model.load_state_dict(torch.load("checkpoints/cond_ema.pt", map_location=device))
model.eval()
clf = Classifier().to(device)
clf.load_state_dict(torch.load("checkpoints/classifier.pt", map_location=device))
clf.eval()

os.makedirs("results", exist_ok=True)
results = {}


def save():
    with open("results/guidance_seeds.json", "w") as f:
        json.dump(
            {"per_label": args.per_label, "steps": args.steps, "accuracy": results},
            f,
            indent=2,
        )


for seed in args.seeds:
    for w in args.ws:
        t0 = time.time()
        correct = total = 0
        for digit in range(10):
            # same seed per (seed, digit): every w starts from the same noise
            torch.manual_seed(seed * 100 + digit)
            labels = torch.full(
                (args.per_label,), digit, device=device, dtype=torch.long
            )
            imgs = cond_ddim_sample(model, labels, steps=args.steps, w=w)
            with torch.no_grad():
                pred = clf(imgs).argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
        acc = correct / total
        results[f"seed{seed}_w{w}"] = acc
        save()
        print(
            f"seed {seed}  w={w}: {acc * 100:.1f}%  ({time.time() - t0:.0f} s)",
            flush=True,
        )

print("\n=== Summary ===")
base = args.ws[0]
for w in args.ws:
    vals = [results[f"seed{s}_w{w}"] for s in args.seeds]
    sd = statistics.stdev(vals) * 100 if len(vals) > 1 else 0.0
    print(
        f"w={w}: mean {statistics.mean(vals) * 100:.1f}%  std {sd:.1f}  runs {[round(v * 100, 1) for v in vals]}"
    )
for w in args.ws[1:]:
    diffs = [
        (results[f"seed{s}_w{w}"] - results[f"seed{s}_w{base}"]) * 100
        for s in args.seeds
    ]
    print(
        f"paired difference (w={w} minus w={base}) per seed: {[round(d, 1) for d in diffs]}  mean {statistics.mean(diffs):.1f} points"
    )
