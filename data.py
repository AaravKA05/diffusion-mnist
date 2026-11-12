"""Step 1: Data. MNIST resized to IMG_SIZE, normalized to [-1, 1]."""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import config

def get_loader(train=True):
    tf = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),                  # [0, 1]
        transforms.Normalize((0.5,), (0.5,)),   # -> [-1, 1]
    ])
    ds = datasets.MNIST("./data", train=train, download=True, transform=tf)
    return DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,
        shuffle=train,
        drop_last=True,
    )

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    loader = get_loader()
    x, y = next(iter(loader))

    print("batch shape:", x.shape)
    print("min:", x.min().item(), " max:", x.max().item())
    print("labels:", y[:16].tolist())

    grid = make_grid((x[:32] + 1) / 2, nrow=8)   # back to [0, 1] for display
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.savefig("results/step1_data.png", bbox_inches="tight")
    plt.show()
