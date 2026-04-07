from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
    "temperal_multiscale_future_dynamics_embodied_policy",
    TEMPERAL_ROOT / "src" / "models" / "multiscale_future_dynamics.py",
)

ModelConfig = future_module.ModelConfig
MultiScaleEncoder = future_module.MultiScaleEncoder
MultiScaleDynamics = future_module.MultiScaleDynamics


@dataclass
class EmbodiedRecurrentPolicyConfig:
    image_h: int = 64
    image_w: int = 64
    in_channels: int = 3
    local_window_length: int = 8
    local_window_stride: int = 4
    self_state_dim: int = 2
    c1: int = 8
    c2: int = 8
    c3: int = 8
    s1: int = 8
    s2: int = 8
    s3: int = 8
    token_dim: int = 64
    self_state_hidden_dim: int = 32
    self_state_token_dim: int = 32
    gru_hidden_dim: int = 64
    gru_layers: int = 1
    policy_hidden_dim: int = 64
    value_hidden_dim: int = 64
    warmup_head_hidden_dim: int = 64
    action_dim: int = 2
    freeze_local_encoder: bool = True
    freeze_local_dynamics: bool = False


class EmbodiedRecurrentPolicy(nn.Module):
    def __init__(self, cfg: EmbodiedRecurrentPolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.freeze_local_encoder = bool(cfg.freeze_local_encoder)
        self.freeze_local_dynamics = bool(cfg.freeze_local_dynamics)
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
        self.self_state_encoder = nn.Sequential(
            nn.Linear(cfg.self_state_dim, cfg.self_state_hidden_dim),
            nn.LayerNorm(cfg.self_state_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.self_state_hidden_dim, cfg.self_state_token_dim),
            nn.LayerNorm(cfg.self_state_token_dim),
            nn.GELU(),
        )
        self.memory_fuse = nn.Sequential(
            nn.Linear(cfg.token_dim + cfg.self_state_token_dim, cfg.token_dim),
            nn.LayerNorm(cfg.token_dim),
            nn.GELU(),
        )
        self.long_gru = nn.GRU(
            input_size=cfg.token_dim,
            hidden_size=cfg.gru_hidden_dim,
            num_layers=cfg.gru_layers,
            batch_first=True,
        )

        readout_dim = cfg.token_dim + cfg.gru_hidden_dim + cfg.self_state_token_dim
        self.policy_head = nn.Sequential(
            nn.Linear(readout_dim, cfg.policy_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.policy_hidden_dim, cfg.action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(readout_dim, cfg.value_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.value_hidden_dim, 1),
        )
        self.landing_head = nn.Sequential(
            nn.Linear(readout_dim, cfg.warmup_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.warmup_head_hidden_dim, 2),
        )
        self.time_head = nn.Sequential(
            nn.Linear(readout_dim, cfg.warmup_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.warmup_head_hidden_dim, 1),
        )

    def set_local_trainable(self, encoder_trainable: bool, dynamics_trainable: bool) -> None:
        self.freeze_local_encoder = not bool(encoder_trainable)
        self.freeze_local_dynamics = not bool(dynamics_trainable)
        for p in self.encoder.parameters():
            p.requires_grad = encoder_trainable
        for p in self.dynamics.parameters():
            p.requires_grad = dynamics_trainable

    def load_pretrained_local_backbone(self, ckpt_path: str, strict: bool = True) -> None:
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

    def load_pretrained_hierarchical(self, ckpt_path: str, strict: bool = False, load_long_memory: bool = True) -> dict:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict") or ckpt.get("model")
        if state is None:
            raise RuntimeError(f"checkpoint {ckpt_path} does not contain model_state_dict/model")

        report = {}
        module_specs = [
            ("encoder.", self.encoder),
            ("dynamics.", self.dynamics),
            ("token_proj.", self.token_proj),
        ]
        if load_long_memory:
            module_specs.append(("long_gru.", self.long_gru))

        for prefix, module in module_specs:
            sub_state = {k.replace(prefix, "", 1): v for k, v in state.items() if k.startswith(prefix)}
            if not sub_state:
                continue
            missing, unexpected = module.load_state_dict(sub_state, strict=strict)
            report[prefix.rstrip(".")] = {
                "missing_keys": list(missing),
                "unexpected_keys": list(unexpected),
            }
        return report

    def _encode_local_windows_impl(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[2] == 1 and self.cfg.in_channels == 3:
            frames = frames.repeat(1, 1, 3, 1, 1)
        b, t, _, _, _ = frames.shape

        def encode_frame(x: torch.Tensor):
            if self.freeze_local_encoder:
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
        if self.freeze_local_encoder and self.freeze_local_dynamics:
            with torch.no_grad():
                pooled = self._encode_local_windows_impl(frames)
        else:
            pooled = self._encode_local_windows_impl(frames)
        return self.token_proj(pooled)

    def encode_self_state(self, self_state: torch.Tensor) -> torch.Tensor:
        return self.self_state_encoder(self_state)

    def fuse_memory_input(self, short_tokens: torch.Tensor, self_tokens: torch.Tensor) -> torch.Tensor:
        return self.memory_fuse(torch.cat([short_tokens, self_tokens], dim=-1))

    def build_readout_input(
        self,
        short_tokens: torch.Tensor,
        long_features: torch.Tensor,
        self_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([short_tokens, long_features, self_tokens], dim=-1)

    def step(
        self,
        window: torch.Tensor,
        self_state: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> dict:
        short_token = self.encode_local_windows(window)
        self_token = self.encode_self_state(self_state)
        memory_token = self.fuse_memory_input(short_token, self_token)
        long_out, next_hidden = self.long_gru(memory_token.unsqueeze(1), hidden)
        long_feat = long_out[:, 0]
        readout = self.build_readout_input(short_token, long_feat, self_token)
        action = self.policy_head(readout)
        value = self.value_head(readout).squeeze(-1)
        landing_offset = self.landing_head(readout)
        time_to_land = F.softplus(self.time_head(readout)).squeeze(-1)
        return {
            "short_token": short_token,
            "self_token": self_token,
            "memory_token": memory_token,
            "long_feature": long_feat,
            "action": action,
            "value": value,
            "landing_offset": landing_offset,
            "time_to_land": time_to_land,
            "hidden": next_hidden,
        }

    def forward(
        self,
        windows: torch.Tensor,
        self_state_seq: torch.Tensor,
        num_windows: torch.Tensor | None = None,
    ) -> dict:
        batch_size, max_windows, t, c, h, w = windows.shape
        flat = windows.reshape(batch_size * max_windows, t, c, h, w)
        short_tokens = self.encode_local_windows(flat).view(batch_size, max_windows, self.cfg.token_dim)

        flat_self = self_state_seq.reshape(batch_size * max_windows, self.cfg.self_state_dim)
        self_tokens = self.encode_self_state(flat_self).view(batch_size, max_windows, self.cfg.self_state_token_dim)
        memory_tokens = self.fuse_memory_input(
            short_tokens.reshape(batch_size * max_windows, self.cfg.token_dim),
            self_tokens.reshape(batch_size * max_windows, self.cfg.self_state_token_dim),
        ).view(batch_size, max_windows, self.cfg.token_dim)

        if num_windows is None:
            long_features, hidden = self.long_gru(memory_tokens)
            valid_mask = torch.ones(batch_size, max_windows, device=windows.device, dtype=torch.bool)
        else:
            packed = pack_padded_sequence(memory_tokens, lengths=num_windows.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, hidden = self.long_gru(packed)
            long_features, _ = pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=max_windows,
            )
            valid_mask = torch.arange(max_windows, device=windows.device).unsqueeze(0) < num_windows.unsqueeze(1)

        readout = self.build_readout_input(short_tokens, long_features, self_tokens)
        actions = self.policy_head(readout)
        values = self.value_head(readout).squeeze(-1)
        landing_offsets = self.landing_head(readout)
        time_to_land = F.softplus(self.time_head(readout)).squeeze(-1)

        if num_windows is not None:
            scale = valid_mask.unsqueeze(-1).to(actions.dtype)
            short_tokens = short_tokens * scale
            self_tokens = self_tokens * scale
            memory_tokens = memory_tokens * scale
            long_features = long_features * scale
            actions = actions * scale
            landing_offsets = landing_offsets * scale
            values = values * valid_mask.to(values.dtype)
            time_to_land = time_to_land * valid_mask.to(time_to_land.dtype)

        return {
            "short_tokens": short_tokens,
            "self_tokens": self_tokens,
            "memory_tokens": memory_tokens,
            "long_features": long_features,
            "actions": actions,
            "values": values,
            "landing_offsets": landing_offsets,
            "time_to_land": time_to_land,
            "hidden": hidden,
            "valid_mask": valid_mask,
        }
