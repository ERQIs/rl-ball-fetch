from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = THIS_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[2]


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


adapter_module = load_module(
    "continuity_prior_future_with_adapter",
    REPO_ROOT / "temperal" / "experiments" / "continuity_prior" / "scripts" / "train_future_with_decoder_adapter.py",
)
dataset_module = load_module(
    "landing_world_model_dataset",
    EXPERIMENT_ROOT / "src" / "datasets" / "landing_rollout_dataset.py",
)

FutureWithVisualDecoderAdapterModel = adapter_module.FutureWithVisualDecoderAdapterModel
build_future_cfg = adapter_module.build_future_cfg
build_spatial_cfg = adapter_module.build_spatial_cfg
load_cfg = adapter_module.load_cfg
LandingRolloutDataset = dataset_module.LandingRolloutDataset


def choose_device(device_pref: str) -> torch.device:
    if device_pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_pref)


def load_model(cfg: dict, device: torch.device):
    ckpt = torch.load(cfg["model"]["checkpoint"], map_location=device)
    future_cfg_dict = ckpt.get("future_config")
    spatial_cfg_dict = ckpt.get("spatial_config")
    if future_cfg_dict is not None:
        future_cfg = adapter_module.ModelConfig(**future_cfg_dict)
    else:
        future_cfg = build_future_cfg(cfg)
    if spatial_cfg_dict is not None:
        spatial_cfg = adapter_module.MultiScaleSpatialPriorConfig(**spatial_cfg_dict)
    else:
        spatial_cfg = build_spatial_cfg(cfg)

    model = FutureWithVisualDecoderAdapterModel(future_cfg, spatial_cfg).to(device)
    state = ckpt.get("model_state_dict") or ckpt.get("model")
    if state is None:
        raise RuntimeError("Checkpoint does not contain model weights")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, future_cfg


@torch.no_grad()
def encode_observation(model, frames: torch.Tensor):
    b, t, _, _, _ = frames.shape
    f1, f2, f3 = model.encoder(frames[:, 0])
    h1, h2, h3 = model.dynamics.init_states(b, f1, f2, f3)
    for i in range(t):
        fi1, fi2, fi3 = model.encoder(frames[:, i])
        h1, h2, h3 = model.dynamics.update_with_observation(fi1, fi2, fi3, h1, h2, h3)
    return h1, h2, h3


def interpolate_touchdown(prev_pos: np.ndarray, curr_pos: np.ndarray, landing_py: float) -> np.ndarray:
    y0 = float(prev_pos[1])
    y1 = float(curr_pos[1])
    if abs(y1 - y0) < 1e-8:
        return curr_pos
    alpha = float((landing_py - y0) / (y1 - y0))
    alpha = min(1.0, max(0.0, alpha))
    return prev_pos + alpha * (curr_pos - prev_pos)


@torch.no_grad()
def rollout_to_landing(model, start_states, max_steps: int, landing_py: float, use_linear_interp: bool):
    h1, h2, h3 = start_states
    prev_pos = None
    trajectory = []
    touchdown_step = None
    touchdown_xyz = None

    for step in range(1, max_steps + 1):
        h1, h2, h3 = model.dynamics.rollout_one_step(h1, h2, h3)
        pos, vel = model._probe(h3)
        pos_np = pos.squeeze(0).detach().cpu().numpy()
        vel_np = vel.squeeze(0).detach().cpu().numpy()
        trajectory.append({"step": step, "pos": pos_np.copy(), "vel": vel_np.copy()})

        if prev_pos is not None and prev_pos[1] > landing_py and pos_np[1] <= landing_py:
            touchdown_step = step
            touchdown_xyz = interpolate_touchdown(prev_pos, pos_np, landing_py) if use_linear_interp else pos_np.copy()
            break
        prev_pos = pos_np.copy()

    if touchdown_xyz is None:
        touchdown_step = max_steps
        touchdown_xyz = trajectory[-1]["pos"].copy() if trajectory else np.zeros(3, dtype=np.float32)

    return {
        "touchdown_step": int(touchdown_step),
        "touchdown_xyz": touchdown_xyz.astype(np.float32),
        "trajectory": trajectory,
    }


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    out_dir = Path(cfg["evaluation"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    device = choose_device(cfg["evaluation"].get("device", "auto"))
    model, _future_cfg = load_model(cfg, device)

    ds_cfg = cfg["dataset"]
    dataset = LandingRolloutDataset(
        root=ds_cfg["root"],
        split_file=ds_cfg["split_file"],
        img_size=ds_cfg.get("img_size", 64),
        observation_length=ds_cfg.get("observation_length", 8),
        observation_end_fraction=ds_cfg.get("observation_end_fraction", 0.5),
        sampling_mode=ds_cfg.get("sampling_mode", "uniform_visible"),
        grayscale=ds_cfg.get("grayscale", True),
        out_channels=ds_cfg.get("out_channels", 3),
        max_episodes=cfg["evaluation"].get("max_episodes"),
    )

    rollout_cfg = cfg["rollout"]
    landing_py = float(rollout_cfg.get("landing_py_threshold", 0.6))
    max_steps = int(rollout_cfg.get("max_steps", 96))
    use_linear_interp = bool(rollout_cfg.get("use_linear_interp", True))

    rows = []
    traj_dir = out_dir / "per_episode_rollouts"
    traj_dir.mkdir(parents=True, exist_ok=True)

    for item in dataset:
        frames = item["frames"].unsqueeze(0).to(device)
        states = encode_observation(model, frames)
        rollout = rollout_to_landing(model, states, max_steps=max_steps, landing_py=landing_py, use_linear_interp=use_linear_interp)

        pred_xyz = rollout["touchdown_xyz"]
        true_xyz = item["landing_xyz"].numpy()
        xz_l2 = float(np.linalg.norm(pred_xyz[[0, 2]] - true_xyz[[0, 2]]))
        xyz_l2 = float(np.linalg.norm(pred_xyz - true_xyz))

        per_step_path = traj_dir / f"{item['episode_id']}.csv"
        with per_step_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "pred_px", "pred_py", "pred_pz", "pred_vx", "pred_vy", "pred_vz"])
            writer.writeheader()
            for step_row in rollout["trajectory"]:
                writer.writerow(
                    {
                        "step": step_row["step"],
                        "pred_px": float(step_row["pos"][0]),
                        "pred_py": float(step_row["pos"][1]),
                        "pred_pz": float(step_row["pos"][2]),
                        "pred_vx": float(step_row["vel"][0]),
                        "pred_vy": float(step_row["vel"][1]),
                        "pred_vz": float(step_row["vel"][2]),
                    }
                )

        rows.append(
            {
                "episode_id": item["episode_id"],
                "last_obs_idx": int(item["last_obs_idx"]),
                "num_total_frames": int(item["num_total_frames"]),
                "pred_touchdown_step": int(rollout["touchdown_step"]),
                "true_landing_px": float(true_xyz[0]),
                "true_landing_py": float(true_xyz[1]),
                "true_landing_pz": float(true_xyz[2]),
                "pred_landing_px": float(pred_xyz[0]),
                "pred_landing_py": float(pred_xyz[1]),
                "pred_landing_pz": float(pred_xyz[2]),
                "landing_xz_l2": xz_l2,
                "landing_xyz_l2": xyz_l2,
            }
        )

    detailed_path = out_dir / "landing_rollout_detailed.csv"
    with detailed_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    xz = np.array([r["landing_xz_l2"] for r in rows], dtype=np.float64)
    xyz = np.array([r["landing_xyz_l2"] for r in rows], dtype=np.float64)
    steps = np.array([r["pred_touchdown_step"] for r in rows], dtype=np.float64)
    summary = {
        "num_episodes": int(len(rows)),
        "landing_xz_l2_mean": float(xz.mean()),
        "landing_xz_l2_median": float(np.median(xz)),
        "landing_xyz_l2_mean": float(xyz.mean()),
        "landing_xyz_l2_median": float(np.median(xyz)),
        "pred_touchdown_step_mean": float(steps.mean()),
        "pred_touchdown_step_median": float(np.median(steps)),
        "checkpoint": cfg["model"]["checkpoint"],
    }
    save_json(summary, out_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
