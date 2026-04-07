import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, in_channels=1, channels=[16,32,64], out_dim=128):
        super().__init__()
        layers = []
        c_in = in_channels
        for c in channels:
            layers += [
                nn.Conv2d(c_in, c, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ]
            c_in = c
        self.conv = nn.Sequential(*layers)
        self._out_dim = channels[-1] * (8 * 8)  # assumes 64x64 input -> 8x8 after 3 pool
        self.fc = nn.Linear(self._out_dim, out_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        y = self.conv(x)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y
