# Author: Leon Kleinschmidt

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoderCNN(nn.Module):
    """Die Klasse beschreibt einen Autoencoder welcher zur Datenencodierung und Datendecodierung CNNs nutzt. Die Reduktionsgröße beträgt dabei (8,4,4) also 128
    Als Übergabewert benötigt diese Klasse den Datensatztypen, dieser kann entweder "MNIST" oder "CIFAR10" sein
    """

    def __init__(self, datasettype="MNIST", drop_prob=0.5):
        super().__init__()

        self.dataset = datasettype
        if self.dataset == "MNIST":
            img_channels = 1
            img_size = 28
        elif self.dataset == "CIFAR10":
            img_channels = 3
            img_size = 32

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),  # output shape [8, 4, 4]
            nn.Tanh(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, img_channels, 3, stride=1, padding=1),
            nn.Upsample(size=(img_size, img_size), mode="bilinear"),
        )

        self.drop = nn.Dropout2d(p=drop_prob)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
