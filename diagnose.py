"""Diagnose two open questions:
1. why digit 1 is under-represented in unconditional samples
2. why label accuracy at w=3.0 is below chance
"""

import argparse
import json

import torch
import torch.nn.functional as F

import config
from data import get_loader
from diffusion import device, ddim_sample
from evaluate import Classifier
from model import UNet
from sample_cond import cond_ddim_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_uncond", type=int, default=500)
    parser.add_argument("--n_per_label", type=int, default=50)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--ws", type=float, nargs="+", default=[1.2, 3.0])
    args = parser.parse_args()

    torch.manual_seed(0)
    out = {"img_size": config.IMG_SIZE, "args": vars(args)}

    clf = Classifier().to(device)
    clf.load_state_dict(torch.load("checkpoints/classifier.pt", map_location=device))
    clf.eval()

    @torch.no_grad()
    def probs_of(x):
        return F.softmax(clf(x.to(device)), dim=1).cpu()

    # ---- Part 1: classifier on real MNIST test images, per class ----
    correct = torch.zeros(10)
    total = torch.zeros(10)
    ink = torch.zeros(10)
    with torch.no_grad():
        for x, y in get_loader(train=False):
            pred = probs_of(x).argmax(1)
            for d in range(10):
                m = y == d
                total[d] += m.sum()
                correct[d] += (pred[m] == d).sum()
                ink[d] += ((x[m] + 1) / 2).mean(dim=(1, 2, 3)).sum()
    print("\n=== Part 1: classifier on REAL test images ===")
    print("digit  accuracy  mean_ink")
    real = []
    for d in range(10):
        acc = (correct[d] / total[d]).item()
        mi = (ink[d] / total[d]).item()
        real.append({"digit": d, "acc": round(acc, 4), "ink": round(mi, 4)})
        print(f"{d:>5}  {acc:8.4f}  {mi:8.4f}")
    out["real_per_class"] = real

    # ---- Part 2: unconditional samples ----
    print("\n=== Part 2: unconditional samples (EMA, DDIM) ===")
    model_u = UNet().to(device)
    model_u.load_state_dict(torch.load("checkpoints/ema.pt", map_location=device))
    model_u.eval()

    ps, inks = [], []
    done = 0
    while done < args.n_uncond:
        b = min(100, args.n_uncond - done)
        with torch.no_grad():
            x = ddim_sample(model_u, b, args.steps).clamp(-1, 1).cpu()
        ps.append(probs_of(x))
        inks.append(((x + 1) / 2).mean(dim=(1, 2, 3)))
        done += b
        print(f"  generated {done}/{args.n_uncond}", flush=True)
    p = torch.cat(ps)
    ink_s = torch.cat(inks)
    pred = p.argmax(1)
    conf = p.max(1).values
    hard = torch.bincount(pred, minlength=10)
    soft = p.sum(0)
    print("class  hard_count  soft_count  mean_conf  mean_ink")
    unc = []
    for d in range(10):
        m = pred == d
        mc = conf[m].mean().item() if m.any() else float("nan")
        mi = ink_s[m].mean().item() if m.any() else float("nan")
        unc.append({"digit": d, "hard": int(hard[d]), "soft": round(soft[d].item(), 1)})
        print(
            f"{d:>5}  {int(hard[d]):10d}  {soft[d].item():10.1f}  {mc:9.3f}  {mi:8.3f}"
        )
    near1 = ((p[:, 1] > 0.2) & (pred != 1)).sum().item()
    print(f"samples with p(1) > 0.2 but predicted another class: {near1}")
    print(
        f"samples with confidence > 0.9: {(conf > 0.9).float().mean().item() * 100:.1f}%"
    )
    out["unconditional"] = unc
    out["near_one_not_predicted_one"] = near1

    # ---- Part 3: conditional model, per label and confusion matrix ----
    model_c = UNet(num_classes=10).to(device)
    model_c.load_state_dict(torch.load("checkpoints/cond_ema.pt", map_location=device))
    model_c.eval()

    out["conditional"] = {}
    for w in args.ws:
        print(f"\n=== Part 3: conditional, w = {w} ===")
        conf_mat = torch.zeros(10, 10, dtype=torch.long)
        sat_all, conf_all = [], []
        for d in range(10):
            labels = torch.full((args.n_per_label,), d, device=device, dtype=torch.long)
            x = cond_ddim_sample(model_c, labels, steps=args.steps, w=w).cpu()
            pr = probs_of(x)
            pd_ = pr.argmax(1)
            for k in pd_.tolist():
                conf_mat[d, k] += 1
            sat_all.append((x.abs() > 0.99).float().mean().item())
            conf_all.append(pr.max(1).values.mean().item())
            print(f"  label {d} done", flush=True)
        per_label = (conf_mat.diag().float() / conf_mat.sum(1).float()).tolist()
        overall = conf_mat.diag().sum().item() / conf_mat.sum().item()
        print("rows = requested label, columns = predicted class")
        for d in range(10):
            print(
                f"  {d}: "
                + " ".join(f"{v:3d}" for v in conf_mat[d].tolist())
                + f"   acc {per_label[d]:.2f}"
            )
        print(f"overall accuracy: {overall * 100:.1f}%")
        print(f"mean classifier confidence: {sum(conf_all) / 10:.3f}")
        print(f"fraction of pixels at +-1 (saturated): {sum(sat_all) / 10:.3f}")
        print("most predicted class overall:", int(conf_mat.sum(0).argmax()))
        out["conditional"][str(w)] = {
            "overall": round(overall, 4),
            "per_label": [round(v, 3) for v in per_label],
            "confusion": conf_mat.tolist(),
        }

    with open("results/diagnose.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved results/diagnose.json")


if __name__ == "__main__":
    main()
