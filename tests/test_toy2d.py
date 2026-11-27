"""Fast checks for toy2d.py (no server needed): python -m pytest tests/test_toy2d.py"""
import base64
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import toy2d  # noqa: E402


def test_schedule_end_values():
    _, _, ab_gentle = toy2d.make_schedule(toy2d.SCHEDULES["gentle"])
    _, _, ab_linear = toy2d.make_schedule(toy2d.SCHEDULES["linear"])
    assert 0.3 < ab_gentle[-1].item() < 0.4      # never reaches noise
    assert ab_linear[-1].item() < 1e-4           # reaches noise


def test_train_sample_and_export_roundtrip():
    data = toy2d.swiss_roll(200, np.random.default_rng(0))
    model, losses = toy2d.train(data, 0.2, steps=50)
    assert len(losses) == 50 and np.isfinite(losses).all()

    x, trace = toy2d.sample(model, 0.2, n=64, seed=0, keep=(50, 0))
    assert x.shape == (64, 2) and torch.isfinite(x).all()
    assert set(trace) == {toy2d.T, 50, 0}

    layers = toy2d.export_model(model)
    assert [(l["in"], l["out"]) for l in layers] == [(3, 128), (128, 128), (128, 2)]
    w0 = np.frombuffer(base64.b64decode(layers[0]["W"]), "<f4").reshape(128, 3)
    assert np.allclose(w0, model.net[0].weight.detach().numpy())
