import torch.nn as nn


class AutoEncoder4x100(nn.Module):
    def __init__(self, datasettype):
        super().__init__()

        if datasettype == "MNIST":
            image_dim = 1 * 28 * 28
        elif datasettype == "CIFAR10":
            image_dim = 3 * 32 * 32

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 9),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(9, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
