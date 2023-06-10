import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, drop_prob=0.5):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(16, 8, 3, stride=1, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(16, 32, 3, stride=1, padding=1)
        self.upconv4 = nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1)
        self.out = nn.ConvTranspose2d(64, 1, 3, stride=1, padding=1)

        self.drop = nn.Dropout2d(p=drop_prob)

    def forward(self, x):
        x = x.view(
            x.size(0), 1, 28, 28
        )  # Reshape flattened image to (batch_size, 1, 28, 28)
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        decoded = decoded.view(
            decoded.size(0), -1
        )  # Flatten the decoded image to (batch_size, 28*28)
        return decoded

    def encode(self, x):
        x = self.drop(F.relu(self.conv1(x)))
        x = self.drop(F.relu(self.conv2(x)))
        x = self.drop(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        return x

    def decode(self, x):
        x = self.drop(F.relu(self.upconv1(x, output_size=(7, 7))))
        x = self.drop(F.relu(self.upconv2(x, output_size=(14, 14))))
        x = self.drop(F.relu(self.upconv3(x, output_size=(14, 14))))
        x = self.drop(F.relu(self.upconv4(x, output_size=(28, 28))))
        x = self.out(x)
        return x
