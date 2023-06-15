import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets


class MNIST_Classification_Class(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(MNIST_Classification_Class, self).__init__()
        # self.input = nn.Linear(28 * 28, 20)  # input
        # self.hidden1 = nn.Linear(20, 20)  # hidden1
        # self.hidden2 = nn.Linear(20, 20)  # hidden2
        # self.out = nn.Linear(20, 10)  # output

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x = self.input(x)
        # x = nn.ReLU(x)
        # x = self.hidden1(x)
        # x = nn.ReLU(x)
        # x = self.hidden2(x)
        # x = nn.ReLU(x)
        # x = self.out(x)
        # x = nn.Softmax(x)
        # print(x.size())
        return self.network(x)
