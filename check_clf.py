"""Test accuracy of the saved classifier on real MNIST (current IMG_SIZE)."""

import torch

from data import get_loader
from diffusion import device
from evaluate import Classifier

clf = Classifier().to(device)
clf.load_state_dict(torch.load("checkpoints/classifier.pt", map_location=device))
clf.eval()

correct = total = 0
with torch.no_grad():
    for x, y in get_loader(train=False):
        pred = clf(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += y.numel()
print(f"classifier test accuracy: {correct / total:.4f}")
