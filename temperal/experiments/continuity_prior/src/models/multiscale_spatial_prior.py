from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MultiScaleSpatialPriorConfig:
    image_h: int = 64
    image_w: int = 64
    in_channels: int = 3
    c1: int = 16
    c2: int = 32
    c3: int = 64
    decoder_mode: str = "f3"
    lambda_rec: float = 1.0
    lambda_trans_f2: float = 0.15
    lambda_trans: float = 1.0
    lambda_wd: float = 1.0
    lambda_nb: float = 0.1
    foreground_weight: float = 6.0
    foreground_dark_threshold: float = 0.55
    foreground_dilate_kernel: int = 5


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


class MultiScaleSpatialEncoder(nn.Module):
    def __init__(self, cfg: MultiScaleSpatialPriorConfig):
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

    def forward(self, x: torch.Tensor):
        f1 = self.stem1(x)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        return f1, f2, f3


class F3Decoder(nn.Module):
    def __init__(self, in_ch: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 48, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(48, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, f3: torch.Tensor) -> torch.Tensor:
        return self.net(f3)


class PyramidDecoder(nn.Module):
    def __init__(self, cfg: MultiScaleSpatialPriorConfig):
        super().__init__()
        self.proj1 = nn.Conv2d(cfg.c1, 16, 1)
        self.proj2 = nn.Conv2d(cfg.c2, 16, 1)
        self.proj3 = nn.Conv2d(cfg.c3, 16, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(48, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            ResidualBlock(32),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, cfg.in_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor) -> torch.Tensor:
        p1 = self.proj1(f1)
        p2 = F.interpolate(self.proj2(f2), size=f1.shape[-2:], mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.proj3(f3), size=f1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse(torch.cat([p1, p2, p3], dim=1))
        return self.up(x)


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


def _foreground_weight_map(target: torch.Tensor, cfg: MultiScaleSpatialPriorConfig) -> torch.Tensor:
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


class MultiScaleSpatialPriorModel(nn.Module):
    def __init__(self, cfg: MultiScaleSpatialPriorConfig):
        super().__init__()
        self.encoder = MultiScaleSpatialEncoder(cfg)
        if cfg.decoder_mode == "pyramid":
            self.decoder = PyramidDecoder(cfg)
        else:
            self.decoder = F3Decoder(cfg.c3, cfg.in_channels)
        self.cfg = cfg

    def decode(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor) -> torch.Tensor:
        if isinstance(self.decoder, PyramidDecoder):
            return self.decoder(f1, f2, f3)
        return self.decoder(f3)

    def forward(self, i_t: torch.Tensor, i_t1: torch.Tensor, flow_t: torch.Tensor):
        f1_t, f2_t, f3_t = self.encoder(i_t)
        f1_t1, f2_t1, f3_t1 = self.encoder(i_t1)

        i_hat_t = self.decode(f1_t, f2_t, f3_t)
        f2_warp = warp_feature_map(f2_t, flow_t)
        f3_warp = warp_feature_map(f3_t, flow_t)
        i_hat_t1_from_warp = self.decode(f1_t1, f2_t1, f3_warp)

        return {
            "f1_t": f1_t,
            "f2_t": f2_t,
            "f3_t": f3_t,
            "f1_t1": f1_t1,
            "f2_t1": f2_t1,
            "f3_t1": f3_t1,
            "f2_warp": f2_warp,
            "f3_warp": f3_warp,
            "i_hat_t": i_hat_t,
            "i_hat_t1_from_warp": i_hat_t1_from_warp,
        }


def compute_spatial_prior_losses(outputs, i_t, i_t1, cfg: MultiScaleSpatialPriorConfig):
    w_t = _foreground_weight_map(i_t, cfg)
    w_t1 = _foreground_weight_map(i_t1, cfg)
    l_rec = _weighted_l1(outputs["i_hat_t"], i_t, w_t)
    l_trans_f2 = F.mse_loss(outputs["f2_warp"], outputs["f2_t1"])
    l_trans = F.mse_loss(outputs["f3_warp"], outputs["f3_t1"])
    l_wd = _weighted_l1(outputs["i_hat_t1_from_warp"], i_t1, w_t1)
    l_nb = _neighbor_smoothness(outputs["f3_t"])
    l_total = (
        cfg.lambda_rec * l_rec
        + cfg.lambda_trans * l_trans
        + cfg.lambda_trans_f2 * l_trans_f2
        + cfg.lambda_wd * l_wd
        + cfg.lambda_nb * l_nb
    )
    return l_total, {
        "l_total": float(l_total.item()),
        "l_rec": float(l_rec.item()),
        "l_trans_f2": float(l_trans_f2.item()),
        "l_trans": float(l_trans.item()),
        "l_wd": float(l_wd.item()),
        "l_nb": float(l_nb.item()),
    }
