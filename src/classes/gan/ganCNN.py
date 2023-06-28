import torch.nn as nn

# non functional


class GeneratorCNN(nn.Module):
    def __init__(self, latent_dim, img_channels, img_size):
        super(GeneratorCNN, self).__init__()

        self.img_size = img_size
        self.init_size = img_size // 4
        self.linear_dim = 256 * self.init_size**2

        self.main = nn.Sequential(
            nn.Linear(latent_dim, self.linear_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(self.linear_dim),
            nn.Unflatten(1, (256, self.init_size, self.init_size)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.main(x)
        x = nn.functional.interpolate(
            x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
        )
        return x


class DiscriminatorCNN(nn.Module):
    def __init__(self, img_channels, img_size):
        super(DiscriminatorCNN, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.output_size = img_size // 8
        self.fc = nn.Linear(256 * self.output_size**2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
