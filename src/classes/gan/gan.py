import torch.nn as nn


class Generator(nn.Module):
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
    def __init__(self, d_input_dim, d_hidden_dim, data_dim):
        super(Discriminator, self).__init__()

        self.data_dim = data_dim

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
        x.view(-1, self.data_dim)
        print(x.ndim)
        return self.model(x)
