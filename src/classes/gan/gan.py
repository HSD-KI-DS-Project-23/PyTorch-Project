# Author: Leon Kleinschmidt

import torch.nn as nn


class Generator(nn.Module):
    """Diese Klasse beschreibt einen 4 schichtiges Neurales Netzwerk, welches den Generator eines GAN bildet.
    Beim Initialisieren können Werte für die Größe des Input Layers, der Hidden Layers und des Output Layers angegeben werden.
    """

    def __init__(self, g_input_dim, g_hidden_dim, g_output_dim):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(g_input_dim, g_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(g_hidden_dim, g_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(g_hidden_dim, g_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(g_hidden_dim, g_output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """Diese Klasse beschreibt einen 4 schichtiges Neurales Netzwerk, welches den Diskriminator eines GAN bildet.
    Beim Initialisieren können Werte für die Größe des Input Layers und der Hidden Layers angegeben werden.
    """

    def __init__(self, d_input_dim, d_hidden_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(d_input_dim, d_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(d_hidden_dim, d_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(d_hidden_dim, d_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(d_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
