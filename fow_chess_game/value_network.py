import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()

        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(20, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Fully connected layers
        conv_out_size = self.get_conv_output((8, 8, 20))
        self.fc1 = nn.Linear(conv_out_size, 512)  # 8x8 is the board size
        self.fc2 = nn.Linear(512, 1)  # Output is a scalar value for the position
        self.flatten = nn.Flatten()

    def forward(self, state):
        # Apply convolutional layers with ReLU activation
        x = self.conv(state)
        x = self.flatten(x)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = torch.tanh(
            self.fc2(x)
        )  # Output between -1 and 1 representing the value of the position

        return x

    def get_conv_output(self, shape):
        x = Variable(torch.rand(1, *shape))
        x = self.conv(x)
        return int(np.prod(x.size()))
