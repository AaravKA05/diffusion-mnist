"""Step 3 and 4: sinusoidal time embedding + U-Net (optionally class-conditional)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.t_proj = nn.Linear(t_dim, out_ch)

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]   # inject time (+ label)
        h = F.silu(self.norm2(self.conv2(h)))
        return h


class UNet(nn.Module):
    def __init__(self, ch=config.BASE_CH, t_dim=config.T_DIM, num_classes=0):
        super().__init__()
        self.num_classes = num_classes
        self.t_mlp = nn.Sequential(
            SinusoidalEmbedding(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.SiLU(),
            nn.Linear(t_dim * 2, t_dim),
        )
        # index `num_classes` is the special "null" label
        self.label_emb = (
            nn.Embedding(num_classes + 1, t_dim) if num_classes > 0 else None
        )
        self.inc = Block(1, ch, t_dim)
        self.down1 = Block(ch, ch * 2, t_dim)
        self.down2 = Block(ch * 2, ch * 4, t_dim)
        self.mid = Block(ch * 4, ch * 4, t_dim)
        self.up2 = Block(ch * 4 + ch * 2, ch * 2, t_dim)
        self.up1 = Block(ch * 2 + ch, ch, t_dim)
        self.out = nn.Conv2d(ch, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t, y=None):
        te = self.t_mlp(t)
        if self.label_emb is not None:
            if y is None:   # no label given -> use the null label
                y = torch.full(
                    (x.size(0),), self.num_classes, device=x.device, dtype=torch.long
                )
            te = te + self.label_emb(y)
        x1 = self.inc(x, te)                        # full res
        x2 = self.down1(self.pool(x1), te)          # 1/2
        x3 = self.down2(self.pool(x2), te)          # 1/4
        m = self.mid(x3, te)
        u2 = F.interpolate(m, scale_factor=2)       # 1/2
        u2 = self.up2(torch.cat([u2, x2], 1), te)   # skip connection
        u1 = F.interpolate(u2, scale_factor=2)      # full res
        u1 = self.up1(torch.cat([u1, x1], 1), te)   # skip connection
        return self.out(u1)


if __name__ == "__main__":
    for nc in (0, 10):
        model = UNet(num_classes=nc)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"num_classes={nc}  parameters: {n_params:,}")

        x = torch.randn(4, 1, config.IMG_SIZE, config.IMG_SIZE)
        t = torch.randint(0, config.T, (4,))
        y = torch.randint(0, 10, (4,)) if nc else None
        out = model(x, t, y)
        print("  input :", x.shape, " output:", out.shape)

        # Does the model actually react to t?
        out2 = model(x, torch.zeros(4, dtype=torch.long), y)
        print("  output changes with t:", not torch.allclose(out, out2))

    # Does the conditional model react to the label?
    a = model(x, t, torch.zeros(4, dtype=torch.long))
    b = model(x, t, torch.ones(4, dtype=torch.long))
    print("output changes with label:", not torch.allclose(a, b))