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
# Noise schedule: "cosine" (Nichol & Dhariwal 2021, the default since it beat
# linear in every paired run, README section 17) or "linear" (BETA_START ->
# BETA_END, used for every result in README sections 1-16). Set per run:
#   DIFFUSION_SCHEDULE=linear python guidance_seeds.py
SCHEDULE = os.environ.get("DIFFUSION_SCHEDULE", "cosine")
if SCHEDULE not in ("linear", "cosine"):
    raise ValueError(f"DIFFUSION_SCHEDULE must be 'linear' or 'cosine', got {SCHEDULE!r}")
COSINE_S = 0.008
# Each schedule has its own checkpoint files. The linear ones keep their
# original names (checkpoints/cond_ema.pt, ema.pt); cosine ones get a prefix.
CKPT_PREFIX = "" if SCHEDULE == "linear" else f"{SCHEDULE}_"


def ckpt(name):
    """Checkpoint path for the current schedule, e.g. ckpt("cond_ema.pt")."""
    return f"checkpoints/{CKPT_PREFIX}{name}"


# Model
BASE_CH = 64  # 32 is enough on CPU
T_DIM = 128

# Training
LR = 2e-4
EPOCHS = 30
EMA_DECAY = 0.995
P_UNCOND = 0.1  # prob. of dropping the label
