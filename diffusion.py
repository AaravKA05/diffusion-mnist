"""Step 2 and 6: noise schedule, q_sample (forward process), sampling."""
import torch

import config

T = config.T
device = config.DEVICE

betas = torch.linspace(config.BETA_START, config.BETA_END, T).to(device)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)
sqrt_ab = torch.sqrt(alpha_bar)
sqrt_1m_ab = torch.sqrt(1.0 - alpha_bar)


def extract(a, t, x_shape):
    """Pick a[t] for each sample in the batch and reshape to (B, 1, 1, 1)."""
    out = a.gather(0, t)
    return out.view(-1, *([1] * (len(x_shape) - 1)))


def q_sample(x0, t, noise):
    """Forward process: jump straight from x0 to x_t."""
    return (
        extract(sqrt_ab, t, x0.shape) * x0
        + extract(sqrt_1m_ab, t, x0.shape) * noise
    )


@torch.no_grad()
def sample(model, n=16, trace_at=()):
    """Reverse process: pure noise -> images. Returns (images, trace)."""
    model.eval()
    x = torch.randn(n, 1, config.IMG_SIZE, config.IMG_SIZE, device=device)
    trace = {}
    for t in reversed(range(T)):
        tt = torch.full((n,), t, device=device, dtype=torch.long)
        eps = model(x, tt)
        a, ab, b = alphas[t], alpha_bar[t], betas[t]
        mean = (x - b / torch.sqrt(1 - ab) * eps) / torch.sqrt(a)
        if t > 0:
            x = mean + torch.sqrt(b) * torch.randn_like(x)
        else:
            x = mean                      # no noise at the last step
        if t in trace_at:
            trace[t] = x.clone()
    return x.clamp(-1, 1), trace


@torch.no_grad()
def ddim_sample(model, n=16, steps=50):
    """Deterministic DDIM sampling with `steps` evenly spaced timesteps."""
    model.eval()
    x = torch.randn(n, 1, config.IMG_SIZE, config.IMG_SIZE, device=device)
    ts = list(reversed(torch.linspace(0, T - 1, steps).long().tolist()))
    for i, t in enumerate(ts):
        t_prev = ts[i + 1] if i + 1 < len(ts) else -1
        tt = torch.full((n,), t, device=device, dtype=torch.long)
        eps = model(x, tt)
        ab = alpha_bar[t]
        ab_prev = alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        x0 = ((x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)).clamp(-1, 1)
        x = torch.sqrt(ab_prev) * x0 + torch.sqrt(1 - ab_prev) * eps
    return x.clamp(-1, 1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data import get_loader

    x, _ = next(iter(get_loader()))
    x0 = x[:1].to(device)              # one digit
    noise = torch.randn_like(x0)       # same noise for every t

    print("alpha_bar[0]  :", alpha_bar[0].item())
    print("alpha_bar[-1] :", alpha_bar[-1].item())

    steps = [0, 50, 100, 200, T - 1]
    fig, axes = plt.subplots(1, len(steps), figsize=(12, 3))
    for ax, ti in zip(axes, steps):
        t = torch.full((1,), ti, device=device, dtype=torch.long)
        xt = q_sample(x0, t, noise)
        ax.imshow(((xt[0, 0].cpu() + 1) / 2).clamp(0, 1), cmap="gray")
        ax.set_title(f"t = {ti}")
        ax.axis("off")
    plt.savefig("results/step2_forward.png", bbox_inches="tight")
    plt.show()