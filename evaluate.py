"""Evaluation: train a small MNIST classifier, then score generated digits."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from data import get_loader
from diffusion import ddim_sample, device
from model import UNet
from sample_cond import cond_ddim_sample


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * (config.IMG_SIZE // 4) ** 2, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


def get_classifier(epochs=2):
    clf = Classifier().to(device)
    path = "checkpoints/classifier.pt"
    if os.path.exists(path):
        clf.load_state_dict(torch.load(path, map_location=device))
        return clf.eval()
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for ep in range(epochs):
        for x, y in get_loader(train=True):
            loss = F.cross_entropy(clf(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"classifier epoch {ep}  loss {loss.item():.4f}")
    clf.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in get_loader(train=False):
            pred = clf(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.numel()
    print(f"classifier test accuracy: {correct / total:.4f}")
    torch.save(clf.state_dict(), path)
    return clf


if __name__ == "__main__":
    torch.manual_seed(0)
    clf = get_classifier()

    # 1) unconditional model: confidence + class balance
    model = UNet().to(device)
    model.load_state_dict(torch.load(config.ckpt("ema.pt"), map_location=device))
    imgs = ddim_sample(model, n=1000, steps=50)
    with torch.no_grad():
        probs = F.softmax(clf(imgs), dim=1)
    conf, pred = probs.max(1)
    print(
        "\n[unconditional] fraction with confidence > 0.9:",
        (conf > 0.9).float().mean().item(),
    )
    print(
        "[unconditional] predicted class counts:",
        torch.bincount(pred, minlength=10).tolist(),
    )

    # 2) conditional model: does the label give the right digit?
    if os.path.exists(config.ckpt("cond_ema.pt")):
        cmodel = UNet(num_classes=10).to(device)
        cmodel.load_state_dict(
            torch.load(config.ckpt("cond_ema.pt"), map_location=device)
        )
        labels = torch.arange(10, device=device).repeat_interleave(100)
        for w in (0.0, 1.0, 1.5, 2.0, 3.0):
            imgs = cond_ddim_sample(cmodel, labels, steps=50, w=w)
            with torch.no_grad():
                pred = clf(imgs).argmax(1)
            acc = (pred == labels).float().mean().item()
            print(f"[conditional] w={w}: label accuracy = {acc:.3f}")
