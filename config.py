"""All hyperparameters in one place."""

import os

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data
IMG_SIZE = 32  # use 16 for fast CPU debugging
BATCH_SIZE = 128

# Diffusion
T = 400  # number of timesteps
BETA_START = 1e-4
BETA_END = 0.02
# Noise schedule: "linear" (BETA_START -> BETA_END, the original models) or
# "cosine" (Nichol & Dhariwal 2021). Set it per run without editing this file:
#   DIFFUSION_SCHEDULE=cosine python train.py --cond
SCHEDULE = os.environ.get("DIFFUSION_SCHEDULE", "linear")
if SCHEDULE not in ("linear", "cosine"):
    raise ValueError(f"DIFFUSION_SCHEDULE must be 'linear' or 'cosine', got {SCHEDULE!r}")
COSINE_S = 0.008
# Checkpoints and result files of non-linear runs get a prefix, so the
# original linear files (checkpoints/cond_ema.pt, ...) are never overwritten.
CKPT_PREFIX = "" if SCHEDULE == "linear" else f"{SCHEDULE}_"

# Model
BASE_CH = 64  # 32 is enough on CPU
T_DIM = 128

# Training
LR = 2e-4
EPOCHS = 30
EMA_DECAY = 0.995
P_UNCOND = 0.1  # prob. of dropping the label
