from __future__ import annotations

import os
import sys
from typing import Any

from mlagents.torch_utils import torch, nn
from mlagents.trainers.settings import EncoderType
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.torch_entities.encoders import VectorInput
from mlagents_envs.base_env import ObservationSpec


class BlindVisualEncoder(nn.Module):
    """
    Ignore visual input and output zeros.
    This creates a strict "no image information" baseline in the trainer.
    """

    def __init__(
        self, height: int, width: int, initial_channels: int, output_size: int
    ) -> None:
        super().__init__()
        self.output_size = output_size
        print(
            f"[exp_vb_0002_blind] BlindVisualEncoder enabled "
            f"(in_ch={initial_channels}, h={height}, w={width}, out={output_size})"
        )

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        batch = visual_obs.shape[0]
        return torch.zeros(
            batch, self.output_size, device=visual_obs.device, dtype=visual_obs.dtype
        )


class ZeroVectorInput(VectorInput):
    """
    Keep vector obs shape unchanged but erase information content.
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(inputs)


def patch_encoder() -> None:
    old_get_encoder = ModelUtils.get_encoder_for_type
    old_get_encoder_for_obs = ModelUtils.get_encoder_for_obs

    def _patched_get_encoder(encoder_type: EncoderType) -> Any:
        enable = os.getenv("BLIND_ENABLE", "1").lower() not in {"0", "false", "no"}
        if enable and encoder_type == EncoderType.SIMPLE:
            return BlindVisualEncoder
        return old_get_encoder(encoder_type)

    def _patched_get_encoder_for_obs(
        obs_spec: ObservationSpec,
        normalize: bool,
        h_size: int,
        attention_embedding_size: int,
        vis_encode_type: EncoderType,
    ):
        zero_vec = os.getenv("BLIND_ZERO_VECTOR", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        if zero_vec and obs_spec.dimension_property in ModelUtils.VALID_VECTOR_PROP:
            # Keep embedding size identical to original vector shape.
            return (ZeroVectorInput(obs_spec.shape[0], normalize=False), obs_spec.shape[0])
        return old_get_encoder_for_obs(
            obs_spec, normalize, h_size, attention_embedding_size, vis_encode_type
        )

    ModelUtils.get_encoder_for_type = staticmethod(_patched_get_encoder)
    ModelUtils.get_encoder_for_obs = staticmethod(_patched_get_encoder_for_obs)


def main() -> None:
    if os.getenv("BLIND_ZERO_VECTOR", "1").lower() not in {"0", "false", "no"}:
        print("[exp_vb_0002_blind] strict mode: visual + vector observations are zeroed.")
    else:
        print("[exp_vb_0002_blind] visual-only blind mode: vector observations are still available.")
    patch_encoder()
    from mlagents.trainers.learn import parse_command_line, run_cli

    options = parse_command_line(sys.argv[1:])
    run_cli(options)


if __name__ == "__main__":
    main()
