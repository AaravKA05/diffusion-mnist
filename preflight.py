"""Showcase pre-flight check: files, packages and free ports (loads no model)."""

import importlib.util
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).parent
FILES = [
    "checkpoints/cosine_cond_ema.pt",
    "checkpoints/cond_ema.pt",
    "checkpoints/classifier.pt",
    "server2.py",
    "static/index2.html",
    "app.py",
    "results/denoising.gif",
    "results/web_demo2.png",
    "results/w_sweep.png",
]
PACKAGES = ["torch", "torchvision", "fastapi", "uvicorn", "gradio", "PIL"]
PORTS = {
    8001: "web demo v2 (uvicorn server2:app --port 8001)",
    7860: "Gradio app (python app.py)",
}

ok = True
print("Python:", sys.executable)
print("\nFiles")
for f in FILES:
    p = ROOT / f
    if p.exists():
        print(f"  OK       {f}  ({p.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"  MISSING  {f}")
        ok = False

print("\nPackages")
for m in PACKAGES:
    if importlib.util.find_spec(m):
        print(f"  OK       {m}")
    else:
        print(f"  MISSING  {m}")
        ok = False

print("\nPorts")
for port, what in PORTS.items():
    with socket.socket() as s:
        s.settimeout(0.5)
        busy = s.connect_ex(("127.0.0.1", port)) == 0
    print(f"  {'BUSY    ' if busy else 'free    '} {port}  {what}")

print("\nAll good." if ok else "\nSomething is missing: fix it before the showcase.")
