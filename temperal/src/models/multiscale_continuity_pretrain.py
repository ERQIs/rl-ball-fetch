from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiscale_future_dynamics import ModelConfig, MultiScaleEncoder


@dataclass
class ContinuityConfig:
    image_h: int = 64
    image_w: int = 64
    in_channels: int = 3
    c1: int = 8
    c2: int = 8
    c3: int = 8
    lambda_rec: float = 1.0
    lambda_trans: float = 1.0
    lambda_wd: float = 1.0
    lambda_nb: float = 0.1
    foreground_weight: float = 6.0
    foreground_dark_threshold: float = 0.55
    foreground_dilate_kernel: int = 5


def _base_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)


def _normalize_flow(flow: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if width > 1:
        flow_x = flow[:, 0] * (2.0 / (width - 1))
    else:
        flow_x = torch.zeros_like(flow[:, 0])
    if height > 1:
        flow_y = flow[:, 1] * (2.0 / (height - 1))
    else:
        flow_y = torch.zeros_like(flow[:, 1])
    return torch.stack([flow_x, flow_y], dim=-1)


def warp_feature_map(feature: torch.Tensor, flow_img: torch.Tensor) -> torch.Tensor:
    b, _, h_f, w_f = feature.shape
    _, _, h_i, w_i = flow_img.shape
    flow = F.interpolate(flow_img, size=(h_f, w_f), mode="bilinear", align_corners=True)
    if w_i > 0:
        flow[:, 0] *= float(w_f) / float(w_i)
    if h_i > 0:
        flow[:, 1] *= float(h_f) / float(h_i)
    grid = _base_grid(h_f, w_f, feature.device, feature.dtype).unsqueeze(0).expand(b, -1, -1, -1)
    sample_grid = grid - _normalize_flow(flow, h_f, w_f)
    return F.grid_sample(feature, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)


def _neighbor_smoothness(f: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(f[:, :, :, 1:] - f[:, :, :, :-1]).mean()
    dy = torch.abs(f[:, :, 1:, :] - f[:, :, :-1, :]).mean()
    return dx + dy


def _to_luma(img: torch.Tensor) -> torch.Tensor:
    if img.shape[1] == 1:
        return img
    r = img[:, 0:1]
    g = img[:, 1:2]
    b = img[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _foreground_weight_map(target: torch.Tensor, cfg: ContinuityConfig) -> torch.Tensor:
    luma = _to_luma(target)
    fg = (luma < cfg.foreground_dark_threshold).float()
    if cfg.foreground_dilate_kernel > 1:
        k = int(cfg.foreground_dilate_kernel)
        if k % 2 == 0:
            k += 1
        fg = F.max_pool2d(fg, kernel_size=k, stride=1, padding=k // 2)
    return 1.0 + cfg.foreground_weight * fg


def _weighted_l1(pred: torch.Tensor, target: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
    w = weight_map.expand_as(pred)
    err = torch.abs(pred - target)
    return (w * err).sum() / (w.sum() + 1e-8)


class ContinuityDecoder(nn.Module):
    def __init__(self, in_ch: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f)


class MultiScaleContinuityPretrainer(nn.Module):
    def __init__(self, cfg: ContinuityConfig):
        super().__init__()
        encoder_cfg = ModelConfig(
            image_h=cfg.image_h,
            image_w=cfg.image_w,
            in_channels=cfg.in_channels,
            c1=cfg.c1,
            c2=cfg.c2,
            c3=cfg.c3,
        )
        self.encoder = MultiScaleEncoder(encoder_cfg)
        self.decoder = ContinuityDecoder(cfg.c3, cfg.in_channels)
        self.cfg = cfg

    def forward(self, i_t: torch.Tensor, i_t1: torch.Tensor, flow_t: torch.Tensor):
        _, _, f_t = self.encoder(i_t)
        _, _, f_t1 = self.encoder(i_t1)
        i_hat_t = self.decoder(f_t)
        f_warp = warp_feature_map(f_t, flow_t)
        i_hat_t1_from_warp = self.decoder(f_warp)
        return {
            "f_t": f_t,
            "f_t1": f_t1,
            "f_warp": f_warp,
            "i_hat_t": i_hat_t,
            "i_hat_t1_from_warp": i_hat_t1_from_warp,
        }


def compute_continuity_losses(outputs, i_t, i_t1, cfg: ContinuityConfig):
    w_t = _foreground_weight_map(i_t, cfg)
    w_t1 = _foreground_weight_map(i_t1, cfg)
    l_rec = _weighted_l1(outputs["i_hat_t"], i_t, w_t)
    l_trans = F.mse_loss(outputs["f_warp"], outputs["f_t1"])
    l_wd = _weighted_l1(outputs["i_hat_t1_from_warp"], i_t1, w_t1)
    l_nb = _neighbor_smoothness(outputs["f_t"])
    l_total = cfg.lambda_rec * l_rec + cfg.lambda_trans * l_trans + cfg.lambda_wd * l_wd + cfg.lambda_nb * l_nb
    return l_total, {
        "l_total": float(l_total.item()),
        "l_rec": float(l_rec.item()),
        "l_trans": float(l_trans.item()),
        "l_wd": float(l_wd.item()),
        "l_nb": float(l_nb.item()),
    }
