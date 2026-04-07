from dataclasses import dataclass

import torch
import torch.nn as nn

from .multiscale_future_dynamics import ModelConfig, MultiScaleEncoder, MultiScaleDynamics


@dataclass
class TransferConfig:
    image_h: int = 64
    image_w: int = 64
    in_channels: int = 3
    observation_length: int = 8
    c1: int = 8
    c2: int = 8
    c3: int = 8
    s1: int = 8
    s2: int = 8
    s3: int = 8
    head_hidden_dim: int = 64
    out_dim: int = 2
    pre_head_layernorm: bool = False


class MultiScaleTransferRegressor(nn.Module):
    def __init__(self, cfg: TransferConfig):
        super().__init__()
        model_cfg = ModelConfig(
            image_h=cfg.image_h,
            image_w=cfg.image_w,
            in_channels=cfg.in_channels,
            seq_len=cfg.observation_length,
            history_len=cfg.observation_length,
            future_len=0,
            future_recon_len=0,
            enable_frame_recon=False,
            pos_dim=3,
            vel_dim=3,
            c1=cfg.c1,
            c2=cfg.c2,
            c3=cfg.c3,
            s1=cfg.s1,
            s2=cfg.s2,
            s3=cfg.s3,
        )
        self.encoder = MultiScaleEncoder(model_cfg)
        self.dynamics = MultiScaleDynamics(model_cfg)
        self.pre_head_norm = nn.LayerNorm(cfg.s3) if cfg.pre_head_layernorm else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(cfg.s3, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden_dim, cfg.out_dim),
        )
        self.cfg = cfg
        self.model_cfg = model_cfg

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[2] == 1 and self.cfg.in_channels == 3:
            frames = frames.repeat(1, 1, 3, 1, 1)
        b, t, _, _, _ = frames.shape
        f1, f2, f3 = self.encoder(frames[:, 0])
        h1, h2, h3 = self.dynamics.init_states(b, f1, f2, f3)

        for i in range(t):
            fi1, fi2, fi3 = self.encoder(frames[:, i])
            h1, h2, h3 = self.dynamics.update_with_observation(fi1, fi2, fi3, h1, h2, h3)

        pooled = h3.mean(dim=(2, 3))
        pooled = self.pre_head_norm(pooled)
        return self.head(pooled)

    def load_pretrained_backbone(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict") or ckpt.get("model")
        if state is None:
            raise RuntimeError(f"checkpoint {ckpt_path} does not contain model_state_dict/model")
        encoder_state = {k.replace("encoder.", "", 1): v for k, v in state.items() if k.startswith("encoder.")}
        dynamics_state = {k.replace("dynamics.", "", 1): v for k, v in state.items() if k.startswith("dynamics.")}
        if not encoder_state:
            encoder_state = ckpt.get("encoder_state_dict") or ckpt.get("encoder")
        if not encoder_state:
            raise RuntimeError(f"checkpoint {ckpt_path} does not contain encoder weights")
        self.encoder.load_state_dict(encoder_state, strict=True)
        if dynamics_state:
            self.dynamics.load_state_dict(dynamics_state, strict=True)

    def set_backbone_trainable(self, trainable: bool):
        for module in [self.encoder, self.dynamics]:
            for p in module.parameters():
                p.requires_grad = trainable
