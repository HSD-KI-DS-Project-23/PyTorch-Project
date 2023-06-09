import torch
from torchvision import datasets


def load_datasets(transform, batch_size, train=True, download=True):
    dataset = datasets.MNIST(
        root="../data", train=train, download=download, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True
    )

    return dataset, loader
