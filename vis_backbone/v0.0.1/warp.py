from __future__ import annotations

import torch
import torch.nn.functional as F


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
    """
    Warp a feature map with forward flow (I_t -> I_t1).

    We use backward sampling for grid_sample, so the sampling grid uses `base - flow`.
    """
    if feature.ndim != 4 or flow_img.ndim != 4:
        raise ValueError("feature and flow_img must be 4D tensors")
    if flow_img.shape[1] != 2:
        raise ValueError("flow_img channel dimension must be 2")

    b, _, h_f, w_f = feature.shape
    _, _, h_i, w_i = flow_img.shape
    if b != flow_img.shape[0]:
        raise ValueError("feature and flow_img batch sizes must match")

    flow = F.interpolate(flow_img, size=(h_f, w_f), mode="bilinear", align_corners=True)

    if w_i > 0:
        flow[:, 0] *= float(w_f) / float(w_i)
    if h_i > 0:
        flow[:, 1] *= float(h_f) / float(h_i)

    grid = _base_grid(h_f, w_f, feature.device, feature.dtype).unsqueeze(0).expand(b, -1, -1, -1)
    flow_norm = _normalize_flow(flow, h_f, w_f)
    sample_grid = grid - flow_norm
    return F.grid_sample(feature, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)
