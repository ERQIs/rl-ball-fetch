from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    image_h: int = 64
    image_w: int = 64
    in_channels: int = 3
    seq_len: int = 12
    history_len: int = 8
    future_len: int = 4
    future_recon_len: int = 4
    enable_frame_recon: bool = True
    pos_dim: int = 3
    vel_dim: int = 3
    c1: int = 8
    c2: int = 8
    c3: int = 8
    s1: int = 8
    s2: int = 8
    s3: int = 8
    decoder_proj_ch: int = 8
    decoder_hidden_ch: int = 8
    lambda_frame: float = 1.0
    lambda_p: float = 1.0
    lambda_v: float = 0.2


class ResidualBlock(nn.Module):
    def __init__(self, ch: int, groups: int = 8):
        super().__init__()
        g = min(groups, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(g, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(g, ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + r)


class ConvGRUCell(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_ch = hidden_ch
        self.conv_zr = nn.Conv2d(in_ch + hidden_ch, 2 * hidden_ch, kernel_size, padding=pad)
        self.conv_h = nn.Conv2d(in_ch + hidden_ch, hidden_ch, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        xh = torch.cat([x, h], dim=1)
        zr = torch.sigmoid(self.conv_zr(xh))
        z, r = torch.chunk(zr, 2, dim=1)
        cand = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * cand


class MultiScaleEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.stem1 = nn.Sequential(
            nn.Conv2d(cfg.in_channels, cfg.c1, 5, stride=2, padding=2),
            nn.GroupNorm(min(4, cfg.c1), cfg.c1),
            nn.GELU(),
            ResidualBlock(cfg.c1),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(cfg.c1, cfg.c2, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, cfg.c2), cfg.c2),
            nn.GELU(),
            ResidualBlock(cfg.c2),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(cfg.c2, cfg.c3, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, cfg.c3), cfg.c3),
            nn.GELU(),
            ResidualBlock(cfg.c3),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self.stem1(x)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        return f1, f2, f3


class MultiScaleDynamics(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.in1 = nn.Conv2d(cfg.c1, cfg.s1, 3, padding=1)
        self.in2 = nn.Conv2d(cfg.c2 + cfg.s1, cfg.s2, 3, padding=1)
        self.in3 = nn.Conv2d(cfg.c3 + cfg.s2, cfg.s3, 3, padding=1)
        self.gru1 = ConvGRUCell(cfg.s1, cfg.s1)
        self.gru2 = ConvGRUCell(cfg.s2, cfg.s2)
        self.gru3 = ConvGRUCell(cfg.s3, cfg.s3)
        self.tr1 = nn.Sequential(nn.Conv2d(cfg.s1, cfg.s1, 3, padding=1), nn.GELU(), nn.Conv2d(cfg.s1, cfg.s1, 3, padding=1))
        self.tr2 = nn.Sequential(nn.Conv2d(cfg.s2, cfg.s2, 3, padding=1), nn.GELU(), nn.Conv2d(cfg.s2, cfg.s2, 3, padding=1))
        self.tr3 = nn.Sequential(nn.Conv2d(cfg.s3, cfg.s3, 3, padding=1), nn.GELU(), nn.Conv2d(cfg.s3, cfg.s3, 3, padding=1))

    def init_states(self, batch_size: int, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor):
        device = f1.device
        h1 = torch.zeros(batch_size, self.cfg.s1, f1.shape[-2], f1.shape[-1], device=device)
        h2 = torch.zeros(batch_size, self.cfg.s2, f2.shape[-2], f2.shape[-1], device=device)
        h3 = torch.zeros(batch_size, self.cfg.s3, f3.shape[-2], f3.shape[-1], device=device)
        return h1, h2, h3

    def update_with_observation(self, obs1, obs2, obs3, h1, h2, h3):
        x1 = self.in1(obs1)
        h1 = self.gru1(x1, h1)
        h1_down = F.avg_pool2d(h1, kernel_size=2, stride=2)
        x2 = self.in2(torch.cat([obs2, h1_down], dim=1))
        h2 = self.gru2(x2, h2)
        h2_down = F.avg_pool2d(h2, kernel_size=2, stride=2)
        x3 = self.in3(torch.cat([obs3, h2_down], dim=1))
        h3 = self.gru3(x3, h3)
        return h1, h2, h3

    def rollout_one_step(self, h1, h2, h3):
        return h1 + self.tr1(h1), h2 + self.tr2(h2), h3 + self.tr3(h3)


class PyramidDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.proj1 = nn.Conv2d(cfg.s1, cfg.decoder_proj_ch, 1)
        self.proj2 = nn.Conv2d(cfg.s2, cfg.decoder_proj_ch, 1)
        self.proj3 = nn.Conv2d(cfg.s3, cfg.decoder_proj_ch, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(cfg.decoder_proj_ch * 3, cfg.decoder_hidden_ch, 3, padding=1),
            nn.GroupNorm(min(8, cfg.decoder_hidden_ch), cfg.decoder_hidden_ch),
            nn.GELU(),
            ResidualBlock(cfg.decoder_hidden_ch),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(cfg.decoder_hidden_ch, cfg.decoder_proj_ch, 4, stride=2, padding=1),
            nn.GroupNorm(min(4, cfg.decoder_proj_ch), cfg.decoder_proj_ch),
            nn.GELU(),
            nn.Conv2d(cfg.decoder_proj_ch, cfg.in_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, h1, h2, h3):
        p1 = self.proj1(h1)
        p2 = F.interpolate(self.proj2(h2), size=h1.shape[-2:], mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.proj3(h3), size=h1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse(torch.cat([p1, p2, p3], dim=1))
        return self.up(x)


class MultiScaleFutureDynamicsModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = MultiScaleEncoder(cfg)
        self.dynamics = MultiScaleDynamics(cfg)
        self.decoder = PyramidDecoder(cfg) if cfg.enable_frame_recon else None
        self.pos_head = nn.Linear(cfg.s3, cfg.pos_dim)
        self.vel_head = nn.Linear(cfg.s3, cfg.vel_dim)

    def _probe(self, h3):
        g = h3.mean(dim=(2, 3))
        return self.pos_head(g), self.vel_head(g)

    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, t, _, _, _ = frames.shape
        assert t == self.cfg.seq_len
        assert self.cfg.history_len + self.cfg.future_len == self.cfg.seq_len

        f1_seq, f2_seq, f3_seq = [], [], []
        for i in range(t):
            f1, f2, f3 = self.encoder(frames[:, i])
            f1_seq.append(f1)
            f2_seq.append(f2)
            f3_seq.append(f3)

        f1_seq = torch.stack(f1_seq, dim=1)
        f2_seq = torch.stack(f2_seq, dim=1)
        f3_seq = torch.stack(f3_seq, dim=1)

        h1, h2, h3 = self.dynamics.init_states(b, f1_seq[:, 0], f2_seq[:, 0], f3_seq[:, 0])
        pred_pos, pred_vel, pred_future_frames = [], [], []

        for i in range(self.cfg.history_len):
            h1, h2, h3 = self.dynamics.update_with_observation(f1_seq[:, i], f2_seq[:, i], f3_seq[:, i], h1, h2, h3)
            p, v = self._probe(h3)
            pred_pos.append(p)
            pred_vel.append(v)

        rh1, rh2, rh3 = h1, h2, h3
        for _ in range(self.cfg.future_len):
            rh1, rh2, rh3 = self.dynamics.rollout_one_step(rh1, rh2, rh3)
            if self.decoder is not None:
                pred_future_frames.append(self.decoder(rh1, rh2, rh3))
            p, v = self._probe(rh3)
            pred_pos.append(p)
            pred_vel.append(v)

        outputs = {
            "pred_pos_all": torch.stack(pred_pos, dim=1),
            "pred_vel_all": torch.stack(pred_vel, dim=1),
            "pred_future_frames": torch.stack(pred_future_frames, dim=1) if pred_future_frames else None,
        }
        return outputs


def compute_losses(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: ModelConfig):
    loss_pos = F.l1_loss(out["pred_pos_all"], batch["pos_seq"])
    loss_vel = F.l1_loss(out["pred_vel_all"], batch["vel_seq"])
    loss_frame = torch.tensor(0.0, device=batch["frames"].device)

    if cfg.enable_frame_recon and out["pred_future_frames"] is not None and cfg.future_recon_len > 0:
        target_future = batch["frames"][:, cfg.history_len:cfg.history_len + cfg.future_recon_len]
        pred_future = out["pred_future_frames"][:, :cfg.future_recon_len]
        loss_frame = F.l1_loss(pred_future, target_future)

    loss = cfg.lambda_p * loss_pos + cfg.lambda_v * loss_vel + cfg.lambda_frame * loss_frame
    stats = {
        "loss": float(loss.item()),
        "frame": float(loss_frame.item()),
        "pos": float(loss_pos.item()),
        "vel": float(loss_vel.item()),
    }
    return loss, stats
