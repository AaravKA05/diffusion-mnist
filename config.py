"""All hyperparameters in one place."""

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data
IMG_SIZE = 32  # use 16 for fast CPU debugging
BATCH_SIZE = 128

# Diffusion
T = 400  # number of timesteps
BETA_START = 1e-4
BETA_END = 0.02

# Model
BASE_CH = 64  # 32 is enough on CPU
T_DIM = 128

# Training
LR = 2e-4
EPOCHS = 30
EMA_DECAY = 0.995
P_UNCOND = 0.1  # prob. of dropping the label
