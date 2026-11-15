"""Small VAE baseline on MNIST, scored by the same classifier as the diffusion model."""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from config import IMG_SIZE
from data import get_loader
from diffusion import device
from evaluate import Classifier

LATENT = 32
S = IMG_SIZE // 8  # spatial size after three stride-2 convolutions (4 for 32x32)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.mu = nn.Linear(128 * S * S, LATENT)
        self.logvar = nn.Linear(128 * S * S, LATENT)
        self.fc = nn.Linear(LATENT, 128 * S * S)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def decode(self, z):
        return self.dec(self.fc(z).view(-1, 128, S, S))

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar


def train(epochs):
    model = VAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = get_loader(train=True)
    for ep in range(epochs):
        t0 = time.time()
        total, n = 0.0, 0
        for x, _ in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            rec = F.mse_loss(recon, x, reduction="sum") / x.size(0)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            loss = rec + kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        print(
            f"epoch {ep + 1}/{epochs}  loss {total / n:.2f}  ({time.time() - t0:.0f} s)"
        )
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/vae.pt")
    return model


@torch.no_grad()
def evaluate(model):
    model.eval()
    torch.manual_seed(0)
    clf = Classifier().to(device)
    clf.load_state_dict(torch.load("checkpoints/classifier.pt", map_location=device))
    clf.eval()

    t0 = time.time()
    imgs = model.decode(torch.randn(64, LATENT, device=device))
    t64 = time.time() - t0
    os.makedirs("results", exist_ok=True)
    save_image(
        ((imgs.cpu() + 1) / 2).clamp(0, 1), "results/vae_samples.png", nrow=8, padding=1
    )

    x = model.decode(torch.randn(1000, LATENT, device=device))
    probs = F.softmax(clf(x), dim=1)
    conf, pred = probs.max(1)
    res = {
        "n_samples": 1000,
        "share_conf_above_0.9": round((conf > 0.9).float().mean().item(), 4),
        "mean_confidence": round(conf.mean().item(), 4),
        "class_counts": torch.bincount(pred, minlength=10).tolist(),
        "seconds_for_64_images": round(t64, 3),
    }
    with open("results/vae_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument(
        "--load", action="store_true", help="skip training, load checkpoints/vae.pt"
    )
    args = ap.parse_args()
    if args.load:
        m = VAE().to(device)
        m.load_state_dict(torch.load("checkpoints/vae.pt", map_location=device))
    else:
        m = train(args.epochs)
    evaluate(m)
