"""Run this first: python check_setup.py"""
import torch, torchvision, matplotlib, numpy

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

x = torch.randn(2, 3)
print("tensor test OK:", x.shape)
