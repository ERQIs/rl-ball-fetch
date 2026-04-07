import torch
import torch.nn as nn
from .cnn_encoder import SmallCNN


class CNNGRURegressor(nn.Module):
    def __init__(self, encoder_channels=[16,32,64], rnn_hidden=128, rnn_layers=1, out_dim=2):
        super().__init__()
        self.encoder = SmallCNN(in_channels=1, channels=encoder_channels, out_dim=rnn_hidden)
        self.gru = nn.GRU(input_size=rnn_hidden, hidden_size=rnn_hidden, num_layers=rnn_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(rnn_hidden, rnn_hidden//2),
            nn.ReLU(),
            nn.Linear(rnn_hidden//2, out_dim)
        )

    def forward(self, frames):
        # frames: (B, T, C, H, W)
        B, T = frames.shape[0], frames.shape[1]
        frames_flat = frames.view(B*T, frames.size(2), frames.size(3), frames.size(4))
        feats = self.encoder(frames_flat)  # (B*T, feat)
        feats = feats.view(B, T, -1)
        out_rnn, h = self.gru(feats)
        last = out_rnn[:, -1, :]
        out = self.head(last)
        return out
