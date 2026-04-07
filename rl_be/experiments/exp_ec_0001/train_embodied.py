from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from mlagents.torch_utils import torch, nn
from mlagents_envs.base_env import ActionSpec, ObservationSpec
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.decoders import ValueHeads
from mlagents.trainers.torch_entities.networks import (
    ActionModel,
    Actor,
    Critic,
)


THIS_FILE = Path(__file__).resolve()
EXPERIMENT_ROOT = THIS_FILE.parent
REPO_ROOT = THIS_FILE.parents[3]
EMBODIED_POLICY_PATH = (
    REPO_ROOT
    / "temperal"
    / "experiments"
    / "embodied_control"
    / "src"
    / "models"
    / "embodied_recurrent_policy.py"
)


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if not EMBODIED_POLICY_PATH.exists():
    raise FileNotFoundError(
        f"Embodied policy module not found: {EMBODIED_POLICY_PATH}"
    )

embodied_module = load_module(
    "temperal_embodied_recurrent_policy_for_rl",
    EMBODIED_POLICY_PATH,
)
EmbodiedRecurrentPolicy = embodied_module.EmbodiedRecurrentPolicy
EmbodiedRecurrentPolicyConfig = embodied_module.EmbodiedRecurrentPolicyConfig


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no"}


def _find_visual_index(observation_specs: List[ObservationSpec]) -> int:
    for i, spec in enumerate(observation_specs):
        if len(spec.shape) == 3:
            return i
    raise RuntimeError("No visual observation found for embodied policy.")


def _find_vector_index(observation_specs: List[ObservationSpec]) -> Optional[int]:
    for i, spec in enumerate(observation_specs):
        if len(spec.shape) == 1:
            return i
    return None


class EmbodiedFeatureBody(nn.Module):
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        encoded_act_size: int = 0,
    ):
        super().__init__()
        del network_settings, encoded_act_size

        self.visual_index = _find_visual_index(observation_specs)
        self.vector_index = _find_vector_index(observation_specs)
        self.visual_spec = observation_specs[self.visual_index]
        self.obs_channels = int(self.visual_spec.shape[0])
        self.obs_height = int(self.visual_spec.shape[1])
        self.obs_width = int(self.visual_spec.shape[2])
        self.window_length = int(os.getenv("EC_WINDOW_LENGTH", "8"))
        self.frame_channels = int(os.getenv("EC_FRAME_CHANNELS", "1"))
        self.self_state_dim = int(os.getenv("EC_SELF_STATE_DIM", "2"))
        self.freeze_local_encoder = _env_flag("EC_FREEZE_LOCAL_ENCODER", True)
        self.freeze_local_dynamics = _env_flag("EC_FREEZE_LOCAL_DYNAMICS", False)
        self.invert_visual = _env_flag("EC_INVERT_VISUAL", False)
        self.ablate_short_path = _env_flag("EC_ABLATE_SHORT", False)
        self.ablate_self_path = _env_flag("EC_ABLATE_SELF", False)
        self.ablate_long_path = _env_flag("EC_ABLATE_LONG", False)
        self.ablate_short_readout = _env_flag("EC_ABLATE_SHORT_READOUT", False)
        self.ablate_self_readout = _env_flag("EC_ABLATE_SELF_READOUT", False)
        self.ablate_long_readout = _env_flag("EC_ABLATE_LONG_READOUT", False)
        self.target_channels = self.window_length * self.frame_channels

        cfg = EmbodiedRecurrentPolicyConfig(
            image_h=self.obs_height,
            image_w=self.obs_width,
            in_channels=3 if self.frame_channels == 1 else self.frame_channels,
            local_window_length=self.window_length,
            local_window_stride=4,
            self_state_dim=self.self_state_dim,
            action_dim=2,
            freeze_local_encoder=self.freeze_local_encoder,
            freeze_local_dynamics=self.freeze_local_dynamics,
        )
        model = EmbodiedRecurrentPolicy(cfg)
        model.set_local_trainable(
            encoder_trainable=not self.freeze_local_encoder,
            dynamics_trainable=not self.freeze_local_dynamics,
        )

        ckpt_path = os.getenv("EC_CKPT", "").strip()
        if not ckpt_path:
            raise RuntimeError("EC_CKPT is required for embodied RL training.")
        ckpt_file = Path(ckpt_path)
        if not ckpt_file.exists():
            raise RuntimeError(f"EC_CKPT does not exist: {ckpt_file}")

        ckpt = torch.load(ckpt_file, map_location="cpu")
        state = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            "[exp_ec_0001] Loaded embodied ckpt="
            f"{ckpt_file} missing={len(missing)} unexpected={len(unexpected)} "
            f"freeze_encoder={self.freeze_local_encoder} "
            f"freeze_dynamics={self.freeze_local_dynamics} "
            f"invert_visual={self.invert_visual} "
            f"ablate_short_path={self.ablate_short_path} "
            f"ablate_self_path={self.ablate_self_path} "
            f"ablate_long_path={self.ablate_long_path} "
            f"ablate_short_readout={self.ablate_short_readout} "
            f"ablate_self_readout={self.ablate_self_readout} "
            f"ablate_long_readout={self.ablate_long_readout}"
        )

        self.encoder = model.encoder
        self.dynamics = model.dynamics
        self.token_proj = model.token_proj
        self.self_state_encoder = model.self_state_encoder
        self.memory_fuse = model.memory_fuse
        self.long_gru = model.long_gru
        self.token_dim = cfg.token_dim
        self.self_token_dim = cfg.self_state_token_dim
        self.long_dim = cfg.gru_hidden_dim
        self.encoding_size = self.token_dim + self.self_token_dim + self.long_dim
        self._memory_size = cfg.gru_hidden_dim * cfg.gru_layers
        self.in_channels = cfg.in_channels

    @property
    def memory_size(self) -> int:
        return self._memory_size

    def update_normalization(self, buffer: AgentBuffer) -> None:
        del buffer

    def _prepare_visual_windows(self, visual_obs: torch.Tensor) -> torch.Tensor:
        x = visual_obs.float()
        if x.ndim != 4:
            raise RuntimeError(f"Expected visual obs with rank 4, got shape={tuple(x.shape)}")

        # ML-Agents supplies visual tensors in NCHW during training, but keep
        # a fallback for NHWC-like inputs to make debugging less brittle.
        if x.shape[1] == self.obs_channels and x.shape[2] == self.obs_height and x.shape[3] == self.obs_width:
            pass
        elif x.shape[-1] == self.obs_channels and x.shape[1] == self.obs_height and x.shape[2] == self.obs_width:
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            raise RuntimeError(
                "Unexpected visual obs shape "
                f"{tuple(x.shape)} for spec {(self.obs_channels, self.obs_height, self.obs_width)}"
            )

        if self.invert_visual:
            # Visual observations are normalized floats in [0, 1], so grayscale
            # inversion is a simple photometric transform.
            x = 1.0 - x

        if x.shape[1] < self.target_channels:
            deficit = self.target_channels - x.shape[1]
            tail = x[:, -self.frame_channels :, :, :]
            repeats = (deficit + self.frame_channels - 1) // self.frame_channels
            pad = tail.repeat(1, repeats, 1, 1)[:, :deficit, :, :]
            x = torch.cat([x, pad], dim=1)
        elif x.shape[1] > self.target_channels:
            x = x[:, -self.target_channels :, :, :]

        if x.shape[1] != self.target_channels:
            raise RuntimeError(
                f"Visual channel prep failed: got {x.shape[1]}, expected {self.target_channels}"
            )

        b, _, h, w = x.shape
        return x.view(b, self.window_length, self.frame_channels, h, w)

    def _prepare_self_state(self, inputs: List[torch.Tensor], batch_size: int) -> torch.Tensor:
        if self.vector_index is None:
            return torch.zeros(batch_size, self.self_state_dim, device=inputs[0].device)

        x = inputs[self.vector_index].float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[1] < self.self_state_dim:
            pad = torch.zeros(
                x.shape[0],
                self.self_state_dim - x.shape[1],
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat([x, pad], dim=1)
        elif x.shape[1] > self.self_state_dim:
            x = x[:, : self.self_state_dim]
        return x

    def _encode_step(
        self,
        window: torch.Tensor,
        self_state: torch.Tensor,
        hidden: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        short_token = self.token_proj(
            self._encode_local_windows(window)
        )
        self_token = self.self_state_encoder(self_state)
        short_for_memory = short_token
        self_for_memory = self_token
        if self.ablate_short_path:
            short_for_memory = torch.zeros_like(short_for_memory)
        if self.ablate_self_path:
            self_for_memory = torch.zeros_like(self_for_memory)
        memory_token = self.memory_fuse(
            torch.cat([short_for_memory, self_for_memory], dim=-1)
        )
        long_out, next_hidden = self.long_gru(memory_token.unsqueeze(1), hidden)
        long_feature = long_out[:, 0]
        if self.ablate_long_path:
            long_feature = torch.zeros_like(long_feature)
            next_hidden = torch.zeros_like(next_hidden)
        short_for_readout = short_token
        self_for_readout = self_token
        long_for_readout = long_feature
        if self.ablate_short_path or self.ablate_short_readout:
            short_for_readout = torch.zeros_like(short_for_readout)
        if self.ablate_self_path or self.ablate_self_readout:
            self_for_readout = torch.zeros_like(self_for_readout)
        if self.ablate_long_path or self.ablate_long_readout:
            long_for_readout = torch.zeros_like(long_for_readout)
        encoding = torch.cat(
            [short_for_readout, long_for_readout, self_for_readout], dim=-1
        )
        return encoding, next_hidden

    def _encode_local_windows(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[2] == 1 and self.in_channels == 3:
            frames = frames.repeat(1, 1, 3, 1, 1)
        batch_size, seq_len, _, _, _ = frames.shape

        def encode_frame(x: torch.Tensor):
            if self.freeze_local_encoder:
                with torch.no_grad():
                    return self.encoder(x)
            return self.encoder(x)

        f1, f2, f3 = encode_frame(frames[:, 0])
        h1, h2, h3 = self.dynamics.init_states(batch_size, f1, f2, f3)
        for i in range(seq_len):
            fi1, fi2, fi3 = encode_frame(frames[:, i])
            h1, h2, h3 = self.dynamics.update_with_observation(
                fi1, fi2, fi3, h1, h2, h3
            )
        return h3.mean(dim=(2, 3))

    def forward(
        self,
        inputs: List[torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del actions
        windows = self._prepare_visual_windows(inputs[self.visual_index])
        self_state = self._prepare_self_state(inputs, windows.shape[0])

        if sequence_length <= 0:
            raise RuntimeError(f"sequence_length must be positive, got {sequence_length}")
        if windows.shape[0] % sequence_length != 0:
            raise RuntimeError(
                f"Batch {windows.shape[0]} is not divisible by sequence_length {sequence_length}"
            )

        batch_size = windows.shape[0] // sequence_length
        windows = windows.view(batch_size, sequence_length, *windows.shape[1:])
        self_state = self_state.view(batch_size, sequence_length, self.self_state_dim)

        hidden = memories
        outputs: List[torch.Tensor] = []
        for t in range(sequence_length):
            step_encoding, hidden = self._encode_step(
                windows[:, t],
                self_state[:, t],
                hidden,
            )
            outputs.append(step_encoding)

        encoding = torch.stack(outputs, dim=1).reshape(
            batch_size * sequence_length, self.encoding_size
        )
        return encoding, hidden


class EmbodiedActor(nn.Module, Actor):
    MODEL_EXPORT_VERSION = 3

    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        super().__init__()
        self.action_spec = action_spec
        self.version_number = torch.nn.Parameter(
            torch.Tensor([self.MODEL_EXPORT_VERSION]), requires_grad=False
        )
        self.is_continuous_int_deprecated = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.is_continuous())]), requires_grad=False
        )
        self.continuous_act_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.continuous_size)]), requires_grad=False
        )
        self.discrete_act_size_vector = torch.nn.Parameter(
            torch.Tensor([self.action_spec.discrete_branches]), requires_grad=False
        )
        self.act_size_vector_deprecated = torch.nn.Parameter(
            torch.Tensor(
                [self.action_spec.continuous_size + sum(self.action_spec.discrete_branches)]
            ),
            requires_grad=False,
        )
        self.network_body = EmbodiedFeatureBody(observation_specs, network_settings)
        self.encoding_size = self.network_body.encoding_size
        self.memory_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.network_body.memory_size)]), requires_grad=False
        )
        self.action_model = ActionModel(
            self.encoding_size,
            action_spec,
            conditional_sigma=conditional_sigma,
            tanh_squash=tanh_squash,
            deterministic=network_settings.deterministic,
        )

    @property
    def memory_size(self) -> int:
        return self.network_body.memory_size

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.network_body.update_normalization(buffer)

    def get_action_and_stats(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[AgentAction, Dict[str, Any], torch.Tensor]:
        encoding, memories = self.network_body(
            inputs, memories=memories, sequence_length=sequence_length
        )
        action, log_probs, entropies = self.action_model(encoding, masks)
        run_out: Dict[str, Any] = {}
        run_out["env_action"] = action.to_action_tuple(
            clip=self.action_model.clip_action
        )
        run_out["log_probs"] = log_probs
        run_out["entropy"] = entropies
        return action, run_out, memories

    def get_stats(
        self,
        inputs: List[torch.Tensor],
        actions: AgentAction,
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Dict[str, Any]:
        encoding, _ = self.network_body(
            inputs, memories=memories, sequence_length=sequence_length
        )
        log_probs, entropies = self.action_model.evaluate(encoding, masks, actions)
        return {"log_probs": log_probs, "entropy": entropies}

    def forward(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[int, torch.Tensor], ...]:
        encoding, memories_out = self.network_body(
            inputs, memories=memories, sequence_length=1
        )
        (
            cont_action_out,
            disc_action_out,
            action_out_deprecated,
            deterministic_cont_action_out,
            deterministic_disc_action_out,
        ) = self.action_model.get_action_out(encoding, masks)

        export_out: List[Union[int, torch.Tensor]] = [
            self.version_number,
            self.memory_size_vector,
        ]
        if self.action_spec.continuous_size > 0:
            export_out += [
                cont_action_out,
                self.continuous_act_size_vector,
                deterministic_cont_action_out,
            ]
        if self.action_spec.discrete_size > 0:
            export_out += [
                disc_action_out,
                self.discrete_act_size_vector,
                deterministic_disc_action_out,
            ]
        export_out += [
            action_out_deprecated,
            self.act_size_vector_deprecated,
            memories_out,
        ]
        return tuple(export_out)


class EmbodiedValueNetwork(nn.Module, Critic):
    def __init__(
        self,
        stream_names: List[str],
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        encoded_act_size: int = 0,
        outputs_per_stream: int = 1,
    ):
        nn.Module.__init__(self)
        self.network_body = EmbodiedFeatureBody(
            observation_specs,
            network_settings,
            encoded_act_size=encoded_act_size,
        )
        self.value_heads = ValueHeads(
            stream_names,
            self.network_body.encoding_size,
            outputs_per_stream,
        )

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.network_body.update_normalization(buffer)

    @property
    def memory_size(self) -> int:
        return self.network_body.memory_size

    def critic_pass(
        self,
        inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return self.forward(inputs, memories=memories, sequence_length=sequence_length)

    def forward(
        self,
        inputs: List[torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        encoding, memories = self.network_body(
            inputs,
            actions=actions,
            memories=memories,
            sequence_length=sequence_length,
        )
        return self.value_heads(encoding), memories


def patch_actor_and_value() -> None:
    import mlagents.trainers.ppo.optimizer_torch as ppo_optimizer
    import mlagents.trainers.ppo.trainer as ppo_trainer
    import mlagents.trainers.torch_entities.networks as networks

    networks.SimpleActor = EmbodiedActor
    networks.ValueNetwork = EmbodiedValueNetwork
    ppo_trainer.SimpleActor = EmbodiedActor
    ppo_optimizer.ValueNetwork = EmbodiedValueNetwork


def patch_optional_onnx_export() -> None:
    import mlagents.trainers.model_saver.torch_model_saver as torch_model_saver

    export_enabled = _env_flag("EC_EXPORT_ONNX", False)
    if export_enabled:
        return

    def _skip_export(self, output_filepath: str, behavior_name: str) -> None:
        del self, output_filepath, behavior_name
        return None

    torch_model_saver.TorchModelSaver.export = _skip_export
    print("[exp_ec_0001] ONNX export disabled for training checkpoints.")


def main() -> None:
    patch_actor_and_value()
    patch_optional_onnx_export()
    from mlagents.trainers.learn import parse_command_line, run_cli

    options = parse_command_line(sys.argv[1:])
    run_cli(options)


if __name__ == "__main__":
    main()
