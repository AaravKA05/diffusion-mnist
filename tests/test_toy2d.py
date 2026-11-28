"""Fast checks for toy2d.py (no server needed): python -m pytest tests/test_toy2d.py"""
import base64
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import toy2d  # noqa: E402


def test_schedule_end_values():
    _, _, ab_gentle = toy2d.make_schedule("gentle")
    _, _, ab_linear = toy2d.make_schedule("linear")
    betas, _, ab_cos = toy2d.make_schedule("cosine")
    assert 0.3 < ab_gentle[-1].item() < 0.4      # never reaches noise
    assert ab_linear[-1].item() < 1e-4           # reaches noise
    assert ab_cos[-1].item() < 1e-4
    assert betas.max().item() <= 0.999 + 1e-6    # capped
    assert torch.all(ab_cos[1:] <= ab_cos[:-1])  # signal only goes down


@pytest.mark.parametrize("name", toy2d.DATASETS)
def test_datasets_shape_and_range(name):
    data = toy2d.MAKERS[name](500, np.random.default_rng(0))
    assert data.shape == (500, 2) and data.dtype == np.float32
    assert np.abs(data).max() < toy2d.X0_CLIP


def test_ddim_frames():
    assert toy2d.ddim_frames(10) == [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
    assert toy2d.ddim_frames(100) == list(range(100, -1, -1))


def test_mode_metric_counts_modes():
    data = toy2d.gaussians8(2000, np.random.default_rng(0))
    one_mode = torch.from_numpy(np.tile(toy2d.GAUSS_CENTRES[:1], (2000, 1)))
    assert toy2d.quality(torch.from_numpy(data), data, "gaussians8")["modes_covered"] == 8
    assert toy2d.quality(one_mode, data, "gaussians8")["modes_covered"] == 1


def test_train_sample_and_export_roundtrip():
    data = toy2d.swiss_roll(200, np.random.default_rng(0))
    model, losses = toy2d.train(data, "cosine", steps=50)
    assert len(losses) == 50 and np.isfinite(losses).all()

    x, trace = toy2d.sample(model, "cosine", n=64, seed=0, keep=(50, 0))
    assert x.shape == (64, 2) and torch.isfinite(x).all()
    assert set(trace) == {toy2d.T, 50, 0}

    for steps in toy2d.DDIM_STEPS:
        y = toy2d.sample_ddim(model, "cosine", n=64, seed=0, steps=steps)
        assert y.shape == (64, 2) and torch.isfinite(y).all()

    layers = toy2d.export_model(model)
    assert [(l["in"], l["out"]) for l in layers] == [(3, 128), (128, 128), (128, 2)]
    w0 = np.frombuffer(base64.b64decode(layers[0]["W"]), "<f4").reshape(128, 3)
    assert np.allclose(w0, model.net[0].weight.detach().numpy())
