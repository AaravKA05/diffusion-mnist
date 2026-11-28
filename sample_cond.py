"""Class-conditional sampling with classifier-free guidance (DDIM)."""

import argparse

import matplotlib.pyplot as plt
import torch
from torchgen import model
from torchvision.utils import make_grid

import config
from diffusion import T, alpha_bar, device
from model import UNet


@torch.no_grad()
def cond_ddim_sample(
    model,
    labels,
    steps=50,
    w=1.0,
    return_intermediates=False,
    snapshot_count=8,
    initial_noise=None,
    capture_timestep=None,
    capture_timesteps=None,
):
    """DDIM sampling with optional intermediate denoising snapshots."""

    model.eval()

    n = labels.size(0)
    null = torch.full_like(labels, model.num_classes)

    if initial_noise is None:
        x = torch.randn(n, 1, config.IMG_SIZE, config.IMG_SIZE, device=device)
    else:
        x = initial_noise.to(device).clone()

        expected_shape = (n, 1, config.IMG_SIZE, config.IMG_SIZE)

        if tuple(x.shape) != expected_shape:
            raise ValueError(f"initial_noise must have shape {expected_shape}")

    ts = list(reversed(torch.linspace(0, T - 1, steps).long().tolist()))

    snapshots = []
    captured_features = None
    captured_timestep_features = {}
    capture_targets = set(capture_timesteps or [])
    if return_intermediates:
        capture_indices = set(
            torch.linspace(0, len(ts) - 1, min(snapshot_count, len(ts)))
            .round()
            .long()
            .tolist()
        )
        snapshots.append({"timestep": ts[0], "image": x.clamp(-1, 1).detach().cpu()})

    for i, t in enumerate(ts):

        t_prev = ts[i + 1] if i + 1 < len(ts) else -1

        tt = torch.full((n,), t, device=device, dtype=torch.long)

        model.capture_features = capture_timestep == t or t in capture_targets
        eps_c = model(x, tt, labels)

        if capture_timestep == t:
            captured_features = model.feature_maps["bottleneck"].detach().cpu()
        if t in capture_targets:
            captured_timestep_features[t] = (
                model.feature_maps["bottleneck"].detach().cpu()
            )
        model.capture_features = False

        if w == 1.0:
            eps = eps_c
        else:
            eps_u = model(x, tt, null)
            eps = eps_u + w * (eps_c - eps_u)

        ab = alpha_bar[t]

        ab_prev = alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

        x0 = ((x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)).clamp(-1, 1)

        x = torch.sqrt(ab_prev) * x0 + torch.sqrt(1 - ab_prev) * eps

        if return_intermediates and i in capture_indices:
            snapshots.append(
                {"timestep": t_prev, "image": x.clamp(-1, 1).detach().cpu()}
            )

    model.capture_features = False

    final_images = x.clamp(-1, 1)

    if return_intermediates:
        if capture_timesteps is not None:
            return final_images, snapshots, captured_timestep_features

        if capture_timestep is not None:
            return final_images, snapshots, captured_features

        return final_images, snapshots

    if capture_timesteps is not None:
        return final_images, captured_timestep_features

    if capture_timestep is not None:
        return final_images, captured_features

    return final_images


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", default=config.ckpt("cond_ema.pt"))

    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--w", type=float, default=1.0)

    args = parser.parse_args()

    torch.manual_seed(0)

    model = UNet(num_classes=10).to(device)

    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    labels = torch.arange(10, device=device).repeat_interleave(8)

    imgs = cond_ddim_sample(model, labels, steps=args.steps, w=args.w)

    grid = make_grid((imgs.cpu() + 1) / 2, nrow=8)

    plt.figure(figsize=(5, 6.5))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")

    plt.title(f"conditional, DDIM {args.steps} steps, w={args.w}")

    plt.savefig(f"results/cond_samples_w{args.w}.png", bbox_inches="tight")

    plt.show()
