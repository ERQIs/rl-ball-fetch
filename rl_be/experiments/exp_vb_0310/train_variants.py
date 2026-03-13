from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from mlagents.torch_utils import torch, nn
from mlagents.trainers.settings import EncoderType
from mlagents.trainers.torch_entities.utils import ModelUtils


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class _StageAEncoder(nn.Module):
    # Mirrors vis_backbone/v0.0.1 encoder.
    def __init__(self, in_channels: int = 1, channels: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GlobalBackbone(nn.Module):
    def __init__(self, in_channels: int, output_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 40, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(40, output_size),
            nn.LeakyReLU(),
        )

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        return self.net(visual_obs)


class _SpatialBackboneBase(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        initial_channels: int,
        output_size: int,
        stage_encoder: nn.Module,
        label: str,
        freeze_backbone: bool,
    ) -> None:
        super().__init__()
        self.target_h = 64
        self.target_w = 64
        self.use_input_adapter = initial_channels != 1
        self.input_adapter = nn.Conv2d(initial_channels, 1, kernel_size=1)
        with torch.no_grad():
            self.input_adapter.weight.fill_(1.0 / float(initial_channels))
            self.input_adapter.bias.zero_()

        self.backbone = stage_encoder
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        enc_channels = self._infer_output_channels()
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(enc_channels, output_size),
            nn.LeakyReLU(),
        )
        print(
            f"[exp_vb_0310] mode={label} input={initial_channels}x{height}x{width} "
            f"backbone_params={_count_params(self.backbone)} projector_params={_count_params(self.projector)} "
            f"freeze_backbone={freeze_backbone}"
        )

    def _infer_output_channels(self) -> int:
        sample = torch.zeros(1, 1, self.target_h, self.target_w)
        with torch.no_grad():
            feats = self.backbone(sample)
        return int(feats.shape[1])

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        x = visual_obs
        if x.shape[-2:] != (self.target_h, self.target_w):
            x = nn.functional.interpolate(
                x,
                size=(self.target_h, self.target_w),
                mode="bilinear",
                align_corners=False,
            )
        if self.use_input_adapter:
            x = self.input_adapter(x)
        feats = self.backbone(x)
        return self.projector(feats)


class SpatialScratchVisualEncoder(_SpatialBackboneBase):
    def __init__(
        self, height: int, width: int, initial_channels: int, output_size: int
    ) -> None:
        channels = int(os.getenv("VB_SPATIAL_CHANNELS", "8"))
        super().__init__(
            height=height,
            width=width,
            initial_channels=initial_channels,
            output_size=output_size,
            stage_encoder=_StageAEncoder(in_channels=1, channels=channels),
            label="spatial_scratch",
            freeze_backbone=False,
        )


class PretrainedSpatialVisualEncoder(_SpatialBackboneBase):
    def __init__(
        self, height: int, width: int, initial_channels: int, output_size: int
    ) -> None:
        ckpt_path = os.getenv("VB_CKPT", "").strip()
        if not ckpt_path:
            raise RuntimeError("VB_CKPT is required for pretrained spatial encoder.")
        ckpt_file = Path(ckpt_path)
        if not ckpt_file.exists():
            raise RuntimeError(f"VB_CKPT does not exist: {ckpt_path}")

        ckpt = torch.load(ckpt_file, map_location="cpu")
        cfg = ckpt.get("config", {})
        channels = int(cfg.get("channels", 8))
        stage_encoder = _StageAEncoder(in_channels=1, channels=channels)
        enc_state = ckpt.get("encoder")
        if enc_state is None:
            model_state = ckpt.get("model", ckpt)
            enc_state = {
                k[len("encoder.") :]: v
                for k, v in model_state.items()
                if isinstance(k, str) and k.startswith("encoder.")
            }
        if not enc_state:
            raise RuntimeError("No encoder weights found in checkpoint.")
        stage_encoder.load_state_dict(enc_state, strict=True)

        freeze_backbone = os.getenv("VB_FREEZE_BACKBONE", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        super().__init__(
            height=height,
            width=width,
            initial_channels=initial_channels,
            output_size=output_size,
            stage_encoder=stage_encoder,
            label="spatial_pretrained",
            freeze_backbone=freeze_backbone,
        )


class GlobalScratchVisualEncoder(nn.Module):
    def __init__(
        self, height: int, width: int, initial_channels: int, output_size: int
    ) -> None:
        super().__init__()
        self.net = _GlobalBackbone(initial_channels, output_size)
        print(
            f"[exp_vb_0310] mode=global_scratch input={initial_channels}x{height}x{width} "
            f"params={_count_params(self.net)}"
        )

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        return self.net(visual_obs)


def patch_encoder() -> None:
    old_get_encoder = ModelUtils.get_encoder_for_type
    mode = os.getenv("VB_MODE", "spatial_pretrain_frozen").strip().lower()

    def _patched_get_encoder(encoder_type: EncoderType) -> Any:
        if encoder_type != EncoderType.SIMPLE:
            return old_get_encoder(encoder_type)
        if mode == "global_scratch":
            return GlobalScratchVisualEncoder
        if mode == "spatial_scratch":
            return SpatialScratchVisualEncoder
        if mode in {"spatial_pretrain_frozen", "spatial_pretrain_finetune"}:
            return PretrainedSpatialVisualEncoder
        raise RuntimeError(f"Unsupported VB_MODE: {mode}")

    ModelUtils.get_encoder_for_type = staticmethod(_patched_get_encoder)


def main() -> None:
    patch_encoder()
    from mlagents.trainers.learn import parse_command_line, run_cli

    options = parse_command_line(sys.argv[1:])
    run_cli(options)


if __name__ == "__main__":
    main()
