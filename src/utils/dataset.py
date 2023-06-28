# Author: Leon Kleinschmidt
# extract dataset and dataloader creation to improve code readability

import torch
from torchvision import datasets


def load_datasets(datasettype, transform, batch_size, train=True, download=True):
    """Returns the dataset and dataloader of either "MNIST" or "CIFAR10" """
    if datasettype == "MNIST":
        dataset = datasets.MNIST(
            root="../data", train=train, download=download, transform=transform
        )
        loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True
        )
        return dataset, loader
    elif datasettype == "CIFAR10":
        dataset = datasets.CIFAR10(
            root="../data", train=train, download=download, transform=transform
        )
        loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True
        )
        return dataset, loader
