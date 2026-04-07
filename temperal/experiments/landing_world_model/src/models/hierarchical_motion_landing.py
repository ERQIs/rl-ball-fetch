from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


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
    "temperal_multiscale_future_dynamics_hierarchical",
    TEMPERAL_ROOT / "src" / "models" / "multiscale_future_dynamics.py",
)

ModelConfig = future_module.ModelConfig
MultiScaleEncoder = future_module.MultiScaleEncoder
MultiScaleDynamics = future_module.MultiScaleDynamics


@dataclass
class HierarchicalMotionLandingConfig:
    image_h: int = 64
    image_w: int = 64
    in_channels: int = 3
    local_window_length: int = 8
    c1: int = 8
    c2: int = 8
    c3: int = 8
    s1: int = 8
    s2: int = 8
    s3: int = 8
    token_dim: int = 64
    gru_hidden_dim: int = 64
    gru_layers: int = 1
    head_hidden_dim: int = 64
    out_dim: int = 2
    freeze_local_backbone: bool = True
    freeze_local_encoder: bool = True
    freeze_local_dynamics: bool = True


class HierarchicalMotionLandingModel(nn.Module):
    def __init__(self, cfg: HierarchicalMotionLandingConfig):
        super().__init__()
        self.cfg = cfg
        local_cfg = ModelConfig(
            image_h=cfg.image_h,
            image_w=cfg.image_w,
            in_channels=cfg.in_channels,
            seq_len=cfg.local_window_length,
            history_len=cfg.local_window_length,
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
        self.local_cfg = local_cfg
        self.encoder = MultiScaleEncoder(local_cfg)
        self.dynamics = MultiScaleDynamics(local_cfg)
        self.token_proj = nn.Sequential(
            nn.Linear(cfg.s3, cfg.token_dim),
            nn.LayerNorm(cfg.token_dim),
            nn.GELU(),
        )
        self.long_gru = nn.GRU(
            input_size=cfg.token_dim,
            hidden_size=cfg.gru_hidden_dim,
            num_layers=cfg.gru_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.gru_hidden_dim, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden_dim, cfg.out_dim),
        )

    def set_local_trainable(self, encoder_trainable: bool, dynamics_trainable: bool) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = encoder_trainable
        for p in self.dynamics.parameters():
            p.requires_grad = dynamics_trainable

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

    def _encode_local_windows_impl(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[2] == 1 and self.cfg.in_channels == 3:
            frames = frames.repeat(1, 1, 3, 1, 1)
        b, t, _, _, _ = frames.shape

        def encode_frame(x: torch.Tensor):
            if self.cfg.freeze_local_encoder:
                with torch.no_grad():
                    return self.encoder(x)
            return self.encoder(x)

        f1, f2, f3 = encode_frame(frames[:, 0])
        h1, h2, h3 = self.dynamics.init_states(b, f1, f2, f3)
        for i in range(t):
            fi1, fi2, fi3 = encode_frame(frames[:, i])
            h1, h2, h3 = self.dynamics.update_with_observation(fi1, fi2, fi3, h1, h2, h3)
        return h3.mean(dim=(2, 3))

    def encode_local_windows(self, frames: torch.Tensor) -> torch.Tensor:
        if self.cfg.freeze_local_encoder and self.cfg.freeze_local_dynamics:
            with torch.no_grad():
                pooled = self._encode_local_windows_impl(frames)
        else:
            pooled = self._encode_local_windows_impl(frames)
        return self.token_proj(pooled)

    def forward(self, windows: torch.Tensor, num_windows: torch.Tensor) -> torch.Tensor:
        batch_size, max_windows, t, c, h, w = windows.shape
        flat = windows.reshape(batch_size * max_windows, t, c, h, w)

        valid_mask = torch.arange(max_windows, device=windows.device).unsqueeze(0) < num_windows.unsqueeze(1)
        valid_flat = valid_mask.reshape(-1)
        flat_valid = flat[valid_flat]

        tokens_valid = self.encode_local_windows(flat_valid)
        tokens = torch.zeros(batch_size * max_windows, self.cfg.token_dim, device=windows.device, dtype=tokens_valid.dtype)
        tokens[valid_flat] = tokens_valid
        tokens = tokens.view(batch_size, max_windows, self.cfg.token_dim)

        packed = pack_padded_sequence(tokens, lengths=num_windows.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.long_gru(packed)
        long_feat = h_n[-1]
        return self.head(long_feat)
