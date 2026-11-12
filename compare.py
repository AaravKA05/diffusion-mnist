"""Put EMA and non-EMA sample grids side by side."""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

a = mpimg.imread("results/samples_ema.png")
b = mpimg.imread("results/samples_no_ema.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(a)
axes[0].set_title("With EMA")
axes[1].imshow(b)
axes[1].set_title("Without EMA")
for ax in axes:
    ax.axis("off")
plt.savefig("results/ema_comparison.png", bbox_inches="tight")
plt.show()