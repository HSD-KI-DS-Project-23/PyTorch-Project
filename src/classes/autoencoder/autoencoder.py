import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, datasettype):
        super().__init__()

        if datasettype == "MNIST":
            image_dim = 1 * 28 * 28
        elif datasettype == "CIFAR10":
            image_dim = 3 * 32 * 32

        self.encoder = nn.Sequential(
            nn.Linear(image_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 36),
            nn.ReLU(),
            nn.Linear(36, 18),
            nn.ReLU(),
            nn.Linear(18, 9),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(9, 18),
            nn.ReLU(),
            nn.Linear(18, 36),
            nn.ReLU(),
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, image_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
