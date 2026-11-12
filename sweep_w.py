"""Fine sweep of guidance strength w, with a plot."""

import matplotlib.pyplot as plt
import torch

from diffusion import device
from evaluate import get_classifier
from model import UNet
from sample_cond import cond_ddim_sample

if __name__ == "__main__":
    clf = get_classifier()
    model = UNet(num_classes=10).to(device)
    model.load_state_dict(torch.load("checkpoints/cond_ema.pt", map_location=device))
    labels = torch.arange(10, device=device).repeat_interleave(100)

    ws = [0.0, 0.5, 0.8, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
    accs = []
    for w in ws:
        torch.manual_seed(0)  # same starting noise for every w
        imgs = cond_ddim_sample(model, labels, steps=50, w=w)
        with torch.no_grad():
            acc = (clf(imgs).argmax(1) == labels).float().mean().item()
        accs.append(acc)
        print(f"w={w}: label accuracy = {acc:.3f}")

    plt.plot(ws, accs, marker="o")
    plt.xlabel("guidance w")
    plt.ylabel("label accuracy")
    plt.title("Label accuracy vs guidance strength")
    plt.grid(alpha=0.3)
    plt.savefig("results/w_sweep.png", bbox_inches="tight")
    plt.show()
