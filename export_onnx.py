"""Export the trained models to ONNX so the website can run them in the browser.

Run once on the laptop (inside the venv), from the project folder:

    python export_onnx.py

It reads   checkpoints/cond_ema.pt     (conditional U-Net, EMA weights)
           checkpoints/classifier.pt   (CNN digit classifier)
and writes frontend/public/models/unet.onnx
           frontend/public/models/classifier.onnx
           frontend/public/models/schedule.json

After exporting it checks that ONNX Runtime gives the same numbers as
PyTorch, so a broken export is caught here and not on the website.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn

import config
from diffusion import alpha_bar
from evaluate import Classifier
from model import UNet

OUT_DIR = os.path.join("frontend", "public", "models")
OPSET = 17


class UNetWithFeatures(nn.Module):
    """Same U-Net, but also returns the bottleneck feature map.

    The Representation Explorer tab needs that feature map, so we export it
    as a second output instead of exporting a second model.
    """

    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.unet.capture_features = True

    def forward(self, x, t, y):
        eps = self.unet(x, t, y)
        return eps, self.unet.feature_maps["bottleneck"]


def load_unet(path):
    model = UNet(num_classes=10)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


def load_classifier(path):
    clf = Classifier()
    clf.load_state_dict(torch.load(path, map_location="cpu"))
    return clf.eval()


def export(module, args, path, input_names, output_names):
    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}
    torch.onnx.export(
        module,
        args,
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False,
    )


def check(path, feeds, expected, name):
    import onnxruntime as ort

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    outputs = session.run(None, feeds)
    for got, want, label in zip(outputs, expected, session.get_outputs()):
        diff = float(np.abs(got - want).max())
        print(f"  {name} / {label.name}: max difference vs PyTorch = {diff:.2e}")
        if diff > 1e-3:
            raise SystemExit(f"Export check FAILED for {name} ({label.name}).")


def main(unet_path="checkpoints/cond_ema.pt", clf_path="checkpoints/classifier.pt"):
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(0)

    unet = UNetWithFeatures(load_unet(unet_path))
    clf = load_classifier(clf_path)

    x = torch.randn(3, 1, config.IMG_SIZE, config.IMG_SIZE)
    t = torch.tensor([0, 150, config.T - 1], dtype=torch.long)
    y = torch.tensor([7, 3, 10], dtype=torch.long)  # 10 = null label

    unet_out = os.path.join(OUT_DIR, "unet.onnx")
    clf_out = os.path.join(OUT_DIR, "classifier.onnx")

    with torch.no_grad():
        export(unet, (x, t, y), unet_out, ["x", "t", "y"], ["eps", "bottleneck"])
        export(clf, (x,), clf_out, ["x"], ["logits"])

        eps, feats = unet(x, t, y)
        logits = clf(x)

    print("Checking ONNX outputs against PyTorch...")
    check(
        unet_out,
        {"x": x.numpy(), "t": t.numpy(), "y": y.numpy()},
        [eps.numpy(), feats.numpy()],
        "unet",
    )
    check(clf_out, {"x": x.numpy()}, [logits.numpy()], "classifier")

    schedule = {
        "T": config.T,
        "img_size": config.IMG_SIZE,
        "num_classes": 10,
        "alpha_bar": [float(v) for v in alpha_bar.cpu()],
    }
    with open(os.path.join(OUT_DIR, "schedule.json"), "w") as f:
        json.dump(schedule, f)

    for name in ("unet.onnx", "classifier.onnx", "schedule.json"):
        size = os.path.getsize(os.path.join(OUT_DIR, name)) / 1e6
        print(f"  wrote {OUT_DIR}/{name} ({size:.1f} MB)")
    print("Export OK.")


if __name__ == "__main__":
    main()
