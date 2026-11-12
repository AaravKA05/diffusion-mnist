"""Step 5: training loop (checkpoints, loss curve, overfit test, conditional)."""
import argparse
import copy
import json

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config
from data import get_loader
from diffusion import T, device, q_sample
from model import UNet


@torch.no_grad()
def ema_update(ema_model, model, decay=config.EMA_DECAY):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.mul_(decay).add_(p.detach(), alpha=1 - decay)


def training_step(model, opt, x, ema=None, y=None):
    """One DDPM training step on a batch x in [-1, 1]."""
    t = torch.randint(0, T, (x.size(0),), device=device)
    noise = torch.randn_like(x)
    xt = q_sample(x, t, noise)
    if y is not None:
        # randomly replace some labels with the null label
        drop = torch.rand(y.shape, device=y.device) < config.P_UNCOND
        y = torch.where(drop, torch.full_like(y, model.num_classes), y)
    loss = F.mse_loss(model(xt, t, y), noise)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if ema is not None:
        ema_update(ema, model)
    return loss.item()


def overfit_test(steps=300):
    """Train on ONE batch of 16 images. Loss must fall sharply."""
    x, _ = next(iter(get_loader()))
    x = x[:16].to(device)
    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.LR)
    for step in range(steps):
        loss = training_step(model, opt, x)
        if step % 50 == 0 or step == steps - 1:
            print(f"step {step:4d}  loss {loss:.4f}")


def train(epochs, cond=False):
    prefix = "cond_" if cond else ""
    loader = get_loader()
    model = UNet(num_classes=10 if cond else 0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.LR)
    ema = copy.deepcopy(model)
    history = []

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in tqdm(loader, desc=f"epoch {epoch}"):
            y = y.to(device) if cond else None
            total += training_step(model, opt, x.to(device), ema, y)
        avg = total / len(loader)
        history.append(avg)
        print(f"epoch {epoch}  loss {avg:.4f}")

        torch.save(model.state_dict(), f"checkpoints/{prefix}ckpt.pt")
        torch.save(ema.state_dict(), f"checkpoints/{prefix}ema.pt")
        with open(f"results/{prefix}loss_history.json", "w") as f:
            json.dump(history, f)

    plt.plot(history)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training loss" + (" (conditional)" if cond else ""))
    plt.savefig(f"results/{prefix}loss_curve.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--cond", action="store_true",
                        help="train the class-conditional model")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    args = parser.parse_args()

    if args.overfit:
        overfit_test()
    else:
        train(args.epochs, cond=args.cond)