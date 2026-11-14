"""Live demo: pick a digit, get new handwritten samples."""

import sys

import gradio as gr
import torch
from PIL import Image
from torchvision.utils import make_grid

from diffusion import device
from model import UNet
from sample_cond import cond_ddim_sample

model = UNet(num_classes=10).to(device)
model.load_state_dict(torch.load("checkpoints/cond_ema.pt", map_location=device))
model.eval()


def generate(digit, how_many, guidance, steps, seed):
    torch.manual_seed(int(seed))
    labels = torch.full((int(how_many),), int(digit), device=device, dtype=torch.long)
    imgs = cond_ddim_sample(model, labels, steps=int(steps), w=float(guidance))
    grid = make_grid((imgs.cpu() + 1) / 2, nrow=4, padding=1)
    arr = (grid * 255).byte().permute(1, 2, 0).numpy()
    img = Image.fromarray(arr)
    return img.resize((img.width * 8, img.height * 8), Image.NEAREST)


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Slider(0, 9, value=7, step=1, label="Digit"),
        gr.Slider(1, 16, value=8, step=1, label="How many samples"),
        gr.Slider(1.0, 2.0, value=1.2, step=0.1, label="Guidance w"),
        gr.Slider(5, 100, value=20, step=5, label="DDIM steps"),
        gr.Number(value=0, label="Seed"),
    ],
    outputs=gr.Image(type="pil", label="Generated digits"),
    title="Diffusion MNIST: class-conditional digit generator",
    description="A DDPM-style model built from scratch in PyTorch. Pick a digit and generate new handwritten samples from noise.",
)

if __name__ == "__main__":
    demo.launch(share="--share" in sys.argv)
