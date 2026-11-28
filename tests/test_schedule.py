"""Noise schedule option (DIFFUSION_SCHEDULE). Run: python -m pytest tests/test_schedule.py

The schedule is read when diffusion.py is imported, so each case runs in a
fresh Python process.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROBE = """
import json, torch, config, diffusion as d
print(json.dumps({
    "schedule": config.SCHEDULE,
    "prefix": config.CKPT_PREFIX,
    "betas": d.betas.cpu().tolist(),
    "ab_T": d.alpha_bar[-1].item(),
}))
"""


def probe(schedule=None):
    env = dict(os.environ)
    env.pop("DIFFUSION_SCHEDULE", None)
    if schedule is not None:
        env["DIFFUSION_SCHEDULE"] = schedule
    r = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    return r


def test_default_is_the_original_linear_schedule():
    import torch
    out = json.loads(probe().stdout)
    assert out["schedule"] == "linear" and out["prefix"] == ""
    expected = torch.linspace(1e-4, 0.02, 400).tolist()
    assert out["betas"] == expected                 # bit-for-bit unchanged
    assert abs(out["ab_T"] - 0.01747) < 1e-4


def test_cosine_schedule():
    out = json.loads(probe("cosine").stdout)
    assert out["prefix"] == "cosine_"
    assert len(out["betas"]) == 400
    assert max(out["betas"]) <= 0.999 + 1e-6
    assert min(out["betas"]) > 0
    assert out["ab_T"] < 1e-6                       # reaches pure noise


def test_unknown_schedule_fails_loudly():
    r = probe("quadratic")
    assert r.returncode != 0 and "DIFFUSION_SCHEDULE" in r.stderr
