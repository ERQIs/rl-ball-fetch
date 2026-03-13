from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from mlagents.torch_utils import torch, nn
from mlagents.trainers.settings import EncoderType
from mlagents.trainers.torch_entities.utils import ModelUtils


class _StageAEncoder(nn.Module):
    # Same encoder layout as vis_backbone/v0.0.1/model.py
    def __init__(self, in_channels: int = 1, channels: int = 64) -> None:
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


class PretrainedStageAVisualEncoder(nn.Module):
    """
    ML-Agents visual encoder wrapper:
    input [B, C_in, H, W] -> output [B, h_size]
    """

    def __init__(
        self, height: int, width: int, initial_channels: int, output_size: int
    ) -> None:
        super().__init__()
        ckpt_path = os.getenv("VBB_CKPT", "").strip()
        if not ckpt_path:
            raise RuntimeError("VBB_CKPT is required for PretrainedStageAVisualEncoder.")
        if not Path(ckpt_path).exists():
            raise RuntimeError(f"VBB_CKPT does not exist: {ckpt_path}")

        self.target_h = 64
        self.target_w = 64
        self.use_input_adapter = initial_channels != 1
        self.input_adapter = nn.Conv2d(initial_channels, 1, kernel_size=1)
        with torch.no_grad():
            self.input_adapter.weight.fill_(1.0 / float(initial_channels))
            self.input_adapter.bias.zero_()

        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt.get("config", {})
        enc_channels = int(cfg.get("channels", 64))
        self.backbone = _StageAEncoder(in_channels=1, channels=enc_channels)

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
        self.backbone.load_state_dict(enc_state, strict=True)

        freeze_backbone = os.getenv("VBB_FREEZE_BACKBONE", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(enc_channels, output_size),
            nn.LeakyReLU(),
        )

        # Optional lightweight debug print in trainer logs.
        print(
            f"[exp_vb_0001] Loaded VBB ckpt={ckpt_path} "
            f"freeze_backbone={freeze_backbone} in_ch={initial_channels} -> h={output_size}"
        )

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


def patch_encoder() -> None:
    old_get_encoder = ModelUtils.get_encoder_for_type

    def _patched_get_encoder(encoder_type: EncoderType) -> Any:
        enable = os.getenv("VBB_ENABLE", "1").lower() not in {"0", "false", "no"}
        if enable and encoder_type == EncoderType.SIMPLE:
            return PretrainedStageAVisualEncoder
        return old_get_encoder(encoder_type)

    ModelUtils.get_encoder_for_type = staticmethod(_patched_get_encoder)


def main() -> None:
    patch_encoder()
    from mlagents.trainers.learn import parse_command_line, run_cli

    options = parse_command_line(sys.argv[1:])
    run_cli(options)


if __name__ == "__main__":
    main()
