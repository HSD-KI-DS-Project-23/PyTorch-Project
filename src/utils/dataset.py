import torch
from torchvision import datasets


def load_datasets(transform, batch_size, train=True):
    dataset = datasets.MNIST(
        root="../data", train=True, download=train, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True
    )

    return dataset, loader
