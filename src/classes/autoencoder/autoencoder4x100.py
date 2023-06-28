import torch.nn as nn


# Author: Leon Kleinschmidt


class AutoEncoder4x100(nn.Module):
    """Die Klasse beschreibt einen Autoencoder welcher die Daten auf eine Größe von self.reduction_dim = 128 reduziert.
    Als Übergabewert benötigt diese Klasse den Datensatztypen, dieser kann entweder "MNIST" oder "CIFAR10" sein
    """

    def __init__(self, datasettype):
        super().__init__()

        # represents the dimensionality of the encoded (compressed) representation of the input data
        self.reduction_dim = 128

        if datasettype == "MNIST":
            self.image_dim = 1 * 28 * 28
            self.output_shape = (1, 28, 28)
        elif datasettype == "CIFAR10":
            self.image_dim = 3 * 32 * 32
            self.output_shape = (3, 32, 32)

        self.encoder = nn.Sequential(
            nn.Linear(self.image_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.reduction_dim),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.reduction_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.image_dim),
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
