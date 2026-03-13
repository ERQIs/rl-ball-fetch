from __future__ import annotations

import torch
import torch.nn.functional as F

from config import StageAConfig


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


def _foreground_weight_map(target: torch.Tensor, cfg: StageAConfig) -> torch.Tensor:
    luma = _to_luma(target)
    fg = (luma < cfg.foreground_dark_threshold).float()
    if cfg.foreground_dilate_kernel > 1:
        k = int(cfg.foreground_dilate_kernel)
        if k % 2 == 0:
            k += 1
        fg = F.max_pool2d(fg, kernel_size=k, stride=1, padding=k // 2)
    return 1.0 + cfg.foreground_weight * fg


def _weighted_l1(pred: torch.Tensor, target: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
    # weight_map: [B,1,H,W], pred/target: [B,C,H,W]
    w = weight_map.expand_as(pred)
    err = torch.abs(pred - target)
    return (w * err).sum() / (w.sum() + 1e-8)


def compute_stage_a_losses(
    outputs: dict[str, torch.Tensor],
    i_t: torch.Tensor,
    i_t1: torch.Tensor,
    cfg: StageAConfig,
) -> dict[str, torch.Tensor]:
    w_t = _foreground_weight_map(i_t, cfg)
    w_t1 = _foreground_weight_map(i_t1, cfg)
    l_rec = _weighted_l1(outputs["i_hat_t"], i_t, w_t)
    l_trans = F.mse_loss(outputs["f_warp"], outputs["f_t1"])
    l_wd = _weighted_l1(outputs["i_hat_t1_from_warp"], i_t1, w_t1)
    l_nb = _neighbor_smoothness(outputs["f_t"])
    l_total = (
        cfg.lambda_rec * l_rec
        + cfg.lambda_trans * l_trans
        + cfg.lambda_wd * l_wd
        + cfg.lambda_nb * l_nb
    )
    return {
        "l_total": l_total,
        "l_rec": l_rec,
        "l_trans": l_trans,
        "l_wd": l_wd,
        "l_nb": l_nb,
    }
