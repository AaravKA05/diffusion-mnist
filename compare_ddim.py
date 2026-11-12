"""Put DDPM and DDIM sample grids side by side."""
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

files = [
    ("results/samples_ema.png", "DDPM, 400 steps (9.2 s)"),
    ("results/samples_ema_ddim50.png", "DDIM, 50 steps (1.2 s)"),
    ("results/samples_ema_ddim20.png", "DDIM, 20 steps (0.5 s)"),
    ("results/samples_ema_ddim10.png", "DDIM, 10 steps (0.3 s)"),
]

fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
for ax, (path, title) in zip(axes, files):
    ax.imshow(mpimg.imread(path))
    ax.set_title(title)
    ax.axis("off")
plt.savefig("results/ddim_comparison.png", bbox_inches="tight")
plt.show()