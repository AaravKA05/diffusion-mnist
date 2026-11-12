# Diffusion MNIST

A DDPM-style image generator built from scratch in PyTorch. Generates handwritten digits from pure noise.

## Structure
```
diffusion-mnist/
  config.py       # hyperparameters
  data.py         # MNIST loading
  diffusion.py    # schedule, q_sample, sampling
  model.py        # time embedding + U-Net
  train.py        # training loop
  sample.py       # generate grids from a checkpoint
  notebooks/      # exploration and plots
  results/        # figures, sample grids, loss curves
  checkpoints/    # saved weights (git-ignored)
```

## Setup
```bash
python -m venv venv
# Windows PowerShell: venv\Scripts\Activate.ps1   | Mac/Linux: source venv/bin/activate
# Install torch + torchvision from pytorch.org first, then:
pip install -r requirements.txt
python check_setup.py
```

## What went wrong
_(fill this in as you hit bugs)_
