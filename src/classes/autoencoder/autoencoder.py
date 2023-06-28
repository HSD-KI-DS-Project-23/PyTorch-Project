# Author: Leon Kleinschmidt

import torch.nn as nn


class AutoEncoder(nn.Module):
    """Die Klasse beschreibt einen simplen Autoencoder welcher die Daten auf eine Größe von 9 reduziert.
    Als Übergabewert benötigt diese Klasse den Datensatztypen, dieser kann entweder "MNIST" oder "CIFAR10" sein
    """

    def __init__(self, datasettype):
        super().__init__()

        if datasettype == "MNIST":
            self.image_dim = 1 * 28 * 28
            self.output_shape = (1, 28, 28)
        elif datasettype == "CIFAR10":
            self.image_dim = 3 * 32 * 32
            self.output_shape = (3, 32, 32)

        self.encoder = nn.Sequential(
            nn.Linear(self.image_dim, 128),
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
            nn.Linear(128, self.image_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        flattened_x = x.view(x.size(0), -1)  # Flatten x
        encoded = self.encoder(flattened_x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(
            x.size(0), *self.output_shape
        )  # Reshape the decoded tensor
        return decoded
