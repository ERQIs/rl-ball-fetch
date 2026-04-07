from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = THIS_DIR.parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[2]
TEMPERAL_ROOT = REPO_ROOT / "temperal"


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


future_module = load_module(
    "temperal_multiscale_future_dynamics_for_landing",
    TEMPERAL_ROOT / "src" / "models" / "multiscale_future_dynamics.py",
)

ModelConfig = future_module.ModelConfig
MultiScaleEncoder = future_module.MultiScaleEncoder
MultiScaleDynamics = future_module.MultiScaleDynamics


@dataclass
class LandingRolloutConfig:
    image_h: int = 64
    image_w: int = 64
    in_channels: int = 3
    observation_length: int = 8
    rollout_steps: int = 36
    c1: int = 8
    c2: int = 8
    c3: int = 8
    s1: int = 8
    s2: int = 8
    s3: int = 8
    head_hidden_dim: int = 64
    out_dim: int = 2
    pre_head_layernorm: bool = False


class FixedRolloutLandingHeadModel(nn.Module):
    def __init__(self, cfg: LandingRolloutConfig):
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
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.encoder = MultiScaleEncoder(model_cfg)
        self.dynamics = MultiScaleDynamics(model_cfg)
        self.pre_head_norm = nn.LayerNorm(cfg.s3) if cfg.pre_head_layernorm else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(cfg.s3, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden_dim, cfg.out_dim),
        )

    def encode_observation(self, frames: torch.Tensor):
        if frames.shape[2] == 1 and self.cfg.in_channels == 3:
            frames = frames.repeat(1, 1, 3, 1, 1)
        b, t, _, _, _ = frames.shape
        f1, f2, f3 = self.encoder(frames[:, 0])
        h1, h2, h3 = self.dynamics.init_states(b, f1, f2, f3)
        for i in range(t):
            fi1, fi2, fi3 = self.encoder(frames[:, i])
            h1, h2, h3 = self.dynamics.update_with_observation(fi1, fi2, fi3, h1, h2, h3)
        return h1, h2, h3

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        h1, h2, h3 = self.encode_observation(frames)
        for _ in range(self.cfg.rollout_steps):
            h1, h2, h3 = self.dynamics.rollout_one_step(h1, h2, h3)
        pooled = h3.mean(dim=(2, 3))
        pooled = self.pre_head_norm(pooled)
        return self.head(pooled)

    def load_pretrained_backbone(self, ckpt_path: str, strict: bool = True) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict") or ckpt.get("model")
        if state is None:
            raise RuntimeError(f"checkpoint {ckpt_path} does not contain model_state_dict/model")
        encoder_state = {k.replace("encoder.", "", 1): v for k, v in state.items() if k.startswith("encoder.")}
        dynamics_state = {k.replace("dynamics.", "", 1): v for k, v in state.items() if k.startswith("dynamics.")}
        if not encoder_state or not dynamics_state:
            raise RuntimeError(f"checkpoint {ckpt_path} does not contain encoder/dynamics weights")
        self.encoder.load_state_dict(encoder_state, strict=strict)
        self.dynamics.load_state_dict(dynamics_state, strict=strict)
