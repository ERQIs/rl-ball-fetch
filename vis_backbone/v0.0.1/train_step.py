from __future__ import annotations

import torch

from config import StageAConfig
from losses import compute_stage_a_losses
from model import StageABackbone


def train_step(
    model: StageABackbone,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    cfg: StageAConfig,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch["i_t"], batch["i_t1"], batch["flow_t"])
    losses = compute_stage_a_losses(outputs, batch["i_t"], batch["i_t1"], cfg)
    losses["l_total"].backward()
    optimizer.step()
    return {k: float(v.detach().cpu().item()) for k, v in losses.items()}
