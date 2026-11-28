"""Linear vs cosine noise schedule on the conditional MNIST model.

    python compare_schedules.py eval --schedule linear    # checkpoints/cond_ema.pt
    python compare_schedules.py eval --schedule cosine    # checkpoints/cosine_cond_ema.pt
    python compare_schedules.py report                    # table + plots from both

`eval` scores one model with the classifier at several DDIM step counts and
guidance strengths. Every (seed, digit) starts from the same noise for both
schedules, so the comparison is paired.

    python compare_schedules.py uncond --schedule cosine  # needs checkpoints/cosine_ema.pt

`uncond` counts digit classes in 500 unconditional samples (DDIM 50) and
writes results/schedule_uncond_<schedule>.json.

`report` reads results/schedule_eval_<schedule>.json for both schedules and
writes results/schedule_compare.md and results/schedule_compare.png.
"""
import argparse
import json
import os
import statistics
import sys
import time


def parse():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("eval")
    ev.add_argument("--schedule", choices=["linear", "cosine"], required=True)
    ev.add_argument("--steps", type=int, nargs="+", default=[10, 20, 50])
    ev.add_argument("--ws", type=float, nargs="+", default=[1.0, 1.2])
    ev.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ev.add_argument("--per_label", type=int, default=100)
    ev.add_argument("--ckpt", help="override the conditional checkpoint path")
    un = sub.add_parser("uncond")
    un.add_argument("--schedule", choices=["linear", "cosine"], required=True)
    un.add_argument("--n", type=int, default=500)
    sub.add_parser("report")
    return ap.parse_args()


def evaluate(args):
    # The schedule is read when diffusion.py is imported, so set it first.
    os.environ["DIFFUSION_SCHEDULE"] = args.schedule
    import torch
    import torch.nn.functional as F
    from torchvision.utils import save_image

    import config
    from diffusion import alpha_bar, ddim_sample, device
    from evaluate import Classifier
    from model import UNet
    from sample_cond import cond_ddim_sample

    assert config.SCHEDULE == args.schedule
    ckpt = args.ckpt or config.ckpt("cond_ema.pt")
    print(f"schedule {args.schedule}, checkpoint {ckpt}, device {device}", flush=True)

    model = UNet(num_classes=10).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    clf = Classifier().to(device)
    clf.load_state_dict(torch.load("checkpoints/classifier.pt", map_location=device))
    clf.eval()

    out = {
        "schedule": args.schedule,
        "checkpoint": ckpt,
        "alpha_bar_T": alpha_bar[-1].item(),
        "per_label": args.per_label,
        "seeds": args.seeds,
        "device": str(device),
        "runs": [],
    }
    path = f"results/schedule_eval_{args.schedule}.json"
    os.makedirs("results", exist_ok=True)

    grid_imgs = []  # 8 samples per digit at DDIM 20, highest w, first seed
    for steps in args.steps:
        for w in args.ws:
            for seed in args.seeds:
                t0 = time.time()
                per_digit, conf = [], []
                for digit in range(10):
                    # same noise for every schedule, step count and w
                    g = torch.Generator().manual_seed(seed * 100 + digit)
                    noise = torch.randn(args.per_label, 1, config.IMG_SIZE, config.IMG_SIZE,
                                        generator=g)
                    labels = torch.full((args.per_label,), digit, device=device, dtype=torch.long)
                    imgs = cond_ddim_sample(model, labels, steps=steps, w=w, initial_noise=noise)
                    with torch.no_grad():
                        p = F.softmax(clf(imgs), dim=1)
                    per_digit.append((p.argmax(1) == digit).float().mean().item())
                    conf.append(p[:, digit].mean().item())
                    if seed == args.seeds[0] and steps == 20 and w == args.ws[-1]:
                        grid_imgs.append(imgs[:8].cpu())
                run = {
                    "steps": steps, "w": w, "seed": seed,
                    "accuracy": sum(per_digit) / 10,
                    "per_digit": [round(a, 4) for a in per_digit],
                    "mean_confidence": sum(conf) / 10,
                    "seconds": round(time.time() - t0, 1),
                }
                out["runs"].append(run)
                print(f"steps {steps:>3}  w {w}  seed {seed}: "
                      f"accuracy {run['accuracy'] * 100:.1f}%  ({run['seconds']} s)", flush=True)
                with open(path, "w") as f:
                    json.dump(out, f, indent=2)

    if grid_imgs:
        grid = torch.cat(grid_imgs)  # 10 digits x 8 samples
        save_image((grid + 1) / 2, f"results/schedule_samples_{args.schedule}.png", nrow=8)

    print("wrote", path)


def count_uncond(args):
    os.environ["DIFFUSION_SCHEDULE"] = args.schedule
    import torch

    import config
    from diffusion import ddim_sample, device
    from evaluate import Classifier
    from model import UNet

    ckpt = config.ckpt("ema.pt")
    model = UNet().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    clf = Classifier().to(device)
    clf.load_state_dict(torch.load("checkpoints/classifier.pt", map_location=device))
    clf.eval()
    torch.manual_seed(0)
    counts = torch.zeros(10, dtype=torch.long)
    left = args.n
    while left > 0:
        n = min(100, left)
        imgs = ddim_sample(model, n=n, steps=50)
        with torch.no_grad():
            counts += torch.bincount(clf(imgs).argmax(1).cpu(), minlength=10)
        left -= n
    out = {"schedule": args.schedule, "checkpoint": ckpt, "n": args.n, "steps": 50,
           "class_counts": counts.tolist()}
    path = f"results/schedule_uncond_{args.schedule}.json"
    os.makedirs("results", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("class counts (digits 0-9):", counts.tolist(), f"-> digit 1: {counts[1]} of {args.n}")
    print("wrote", path)


def summarise(res):
    """(steps, w) -> list of accuracies over seeds, in seed order."""
    table = {}
    for r in res["runs"]:
        table.setdefault((r["steps"], r["w"]), {})[r["seed"]] = r
    return table


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = {}
    for s in ("linear", "cosine"):
        p = f"results/schedule_eval_{s}.json"
        if not os.path.exists(p):
            sys.exit(f"missing {p}: run `python compare_schedules.py eval --schedule {s}` first")
        with open(p) as f:
            res[s] = json.load(f)
    for key in ("per_label", "seeds"):
        if res["linear"][key] != res["cosine"][key]:
            print(f"warning: {key} differs (linear {res['linear'][key]}, "
                  f"cosine {res['cosine'][key]}); only shared settings are compared")
    lin, cos = summarise(res["linear"]), summarise(res["cosine"])
    keys = sorted(set(lin) & set(cos))

    lines = [
        "# Linear vs cosine noise schedule (conditional model)",
        "",
        f"Classifier label accuracy, {res['linear']['per_label']} samples per digit per seed, "
        "same starting noise for both schedules. Mean over seeds; the last column is the "
        "per-seed difference (cosine minus linear, in points).",
        "",
        f"alpha_bar_T: linear {res['linear']['alpha_bar_T']:.3g}, "
        f"cosine {res['cosine']['alpha_bar_T']:.3g}",
        "",
        "| DDIM steps | w | Linear | Cosine | Cosine - linear, per seed |",
        "|---|---|---|---|---|",
    ]
    for k in keys:
        seeds = sorted(set(lin[k]) & set(cos[k]))
        la = [lin[k][s]["accuracy"] * 100 for s in seeds]
        ca = [cos[k][s]["accuracy"] * 100 for s in seeds]
        diffs = ", ".join(f"{c - l:+.1f}" for l, c in zip(la, ca))
        lines.append(f"| {k[0]} | {k[1]} | {statistics.mean(la):.1f}% | "
                     f"{statistics.mean(ca):.1f}% | {diffs} |")

    lines += ["", "Per-digit accuracy at DDIM 20 steps, highest w, mean over seeds:", "",
              "| Digit | " + " | ".join(str(d) for d in range(10)) + " |",
              "|---" * 11 + "|"]
    k20 = max((k for k in keys if k[0] == 20), key=lambda k: k[1], default=None)
    if k20:
        for name, tab in (("Linear", lin), ("Cosine", cos)):
            per = [statistics.mean(r["per_digit"][d] for r in tab[k20].values()) * 100
                   for d in range(10)]
            lines.append(f"| {name} | " + " | ".join(f"{v:.0f}" for v in per) + " |")

    for s in ("linear", "cosine"):
        p = f"results/schedule_uncond_{s}.json"
        if os.path.exists(p):
            with open(p) as f:
                u = json.load(f)
            lines += ["", f"Unconditional {s}, {u['n']} samples, DDIM {u['steps']}: class counts "
                      f"{u['class_counts']} (digit 1: {u['class_counts'][1]})"]

    md = "\n".join(lines) + "\n"
    with open("results/schedule_compare.md", "w") as f:
        f.write(md)
    print(md)

    ws = sorted({k[1] for k in keys})
    fig, axes = plt.subplots(1, len(ws), figsize=(5 * len(ws), 3.8), squeeze=False)
    for ax, w in zip(axes[0], ws):
        for name, tab, color in (("linear", lin, "#1f4e8c"), ("cosine", cos, "#ff5a4a")):
            steps = sorted(k[0] for k in keys if k[1] == w)
            means = [statistics.mean(r["accuracy"] for r in tab[(s, w)].values()) * 100
                     for s in steps]
            ax.plot(steps, means, "o-", color=color, label=name)
            for s in steps:
                pts = [r["accuracy"] * 100 for r in tab[(s, w)].values()]
                ax.scatter([s] * len(pts), pts, color=color, s=8, alpha=0.4)
        ax.set_title(f"Guidance w = {w}")
        ax.set_xlabel("DDIM steps")
        ax.set_ylabel("Label accuracy (%)")
        ax.set_xticks(steps)
        ax.legend()
    fig.tight_layout()
    fig.savefig("results/schedule_compare.png", dpi=130)
    print("wrote results/schedule_compare.md and results/schedule_compare.png")


if __name__ == "__main__":
    a = parse()
    if a.cmd == "eval":
        evaluate(a)
    elif a.cmd == "uncond":
        count_uncond(a)
    else:
        report()
