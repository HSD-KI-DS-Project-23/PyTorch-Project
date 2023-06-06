import torch
from torchvision import datasets


def load_datasets(transform, batch_size):
    dataset = datasets.MNIST(
        root="../data", train=True, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True
    )

    return dataset, loader
