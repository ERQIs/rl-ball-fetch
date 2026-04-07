import torch
import torch.nn as nn
from .cnn_encoder import SmallCNN


class SingleFrameRegressor(nn.Module):
    def __init__(self, encoder_channels=[16,32,64], hidden_dim=128):
        super().__init__()
        self.encoder = SmallCNN(in_channels=1, channels=encoder_channels, out_dim=hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 2)
        )

    def forward(self, frames):
        # frames: (B, T, C, H, W). For single-frame baseline, take last frame
        if frames.dim() == 5:
            x = frames[:, -1]
        else:
            x = frames
        feats = self.encoder(x)
        out = self.head(feats)
        return out
