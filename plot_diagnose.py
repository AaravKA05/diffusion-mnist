"""Make figures from results/diagnose.json (no model or sampling needed)."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("results/diagnose.json") as f:
    d = json.load(f)

n_unc = d["args"]["n_uncond"]
n_lab = d["args"]["n_per_label"]
ws = list(d["conditional"].keys())  # e.g. ["1.2", "3.0"]
digits = np.arange(10)

fig, ax = plt.subplots(2, 2, figsize=(11, 9))

# (a) unconditional samples per predicted digit
counts = [row["hard"] for row in d["unconditional"]]
ax[0, 0].bar(digits, counts, color="#0F8B8D")
ax[0, 0].axhline(n_unc / 10, color="gray", linestyle="--", linewidth=1)
ax[0, 0].set_title(
    f"Unconditional: samples per predicted digit ({n_unc} samples)\n(dashed line = equal share)",
    fontsize=10,
)
ax[0, 0].set_xlabel("predicted digit")
ax[0, 0].set_ylabel("samples")
ax[0, 0].set_xticks(digits)
for x, c in zip(digits, counts):
    ax[0, 0].text(x, c + 1, str(c), ha="center", fontsize=8)

# (b) label accuracy per digit at the first w
w1 = ws[0]
acc = d["conditional"][w1]["per_label"]
ax[0, 1].bar(digits, [a * 100 for a in acc], color="#E07A5F")
ax[0, 1].set_ylim(0, 105)
ax[0, 1].set_title(
    f"Conditional, w = {w1}: label accuracy per digit ({n_lab} each)\noverall {d['conditional'][w1]['overall'] * 100:.1f}%",
    fontsize=10,
)
ax[0, 1].set_xlabel("requested digit")
ax[0, 1].set_ylabel("accuracy (%)")
ax[0, 1].set_xticks(digits)
for x, a in zip(digits, acc):
    ax[0, 1].text(x, a * 100 + 1.5, f"{a * 100:.0f}", ha="center", fontsize=8)

# (c, d) confusion matrices: rows = requested digit, columns = digit the classifier predicted
for k, w in enumerate(ws[:2]):
    a = ax[1, k]
    cm = np.array(d["conditional"][w]["confusion"])
    a.imshow(cm, cmap="Blues")
    a.set_title(
        f"Conditional, w = {w}: requested vs predicted\noverall {d['conditional'][w]['overall'] * 100:.1f}%",
        fontsize=10,
    )
    a.set_xlabel("predicted digit")
    a.set_ylabel("requested digit")
    a.set_xticks(digits)
    a.set_yticks(digits)
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                a.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black",
                )

plt.tight_layout()
plt.savefig("results/diagnose_plots.png", dpi=150)
print("saved results/diagnose_plots.png")
