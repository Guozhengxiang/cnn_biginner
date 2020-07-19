import numpy as np
import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(True)
        )
        self.linear = nn.Linear(in_features=7*7*32, out_features=10, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out