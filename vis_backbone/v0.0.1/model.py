from __future__ import annotations

import torch
from torch import nn

from warp import warp_feature_map


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 1, channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),   # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(48, channels, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, channels: int = 64, out_channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(channels, 48, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 32, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f)


class StageABackbone(nn.Module):
    def __init__(self, channels: int = 64, input_channels: int = 1) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels=input_channels, channels=channels)
        self.decoder = Decoder(channels=channels, out_channels=input_channels)

    def forward(self, i_t: torch.Tensor, i_t1: torch.Tensor, flow_t: torch.Tensor) -> dict[str, torch.Tensor]:
        f_t = self.encoder(i_t)
        f_t1 = self.encoder(i_t1)
        i_hat_t = self.decoder(f_t)
        f_warp = warp_feature_map(f_t, flow_t)
        i_hat_t1_from_warp = self.decoder(f_warp)
        return {
            "f_t": f_t,
            "f_t1": f_t1,
            "i_hat_t": i_hat_t,
            "f_warp": f_warp,
            "i_hat_t1_from_warp": i_hat_t1_from_warp,
        }
