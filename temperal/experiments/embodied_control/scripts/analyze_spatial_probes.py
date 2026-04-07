from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
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


model_module = load_module(
    "embodied_control_model_for_probe",
    EXPERIMENT_ROOT / "src" / "models" / "embodied_recurrent_policy.py",
)

EmbodiedRecurrentPolicy = model_module.EmbodiedRecurrentPolicy
EmbodiedRecurrentPolicyConfig = model_module.EmbodiedRecurrentPolicyConfig


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_cfg(cfg: dict) -> EmbodiedRecurrentPolicyConfig:
    ds = cfg["dataset"]
    mdl = cfg["model"]
    return EmbodiedRecurrentPolicyConfig(
        image_h=ds.get("img_size", 64),
        image_w=ds.get("img_size", 64),
        in_channels=ds.get("out_channels", 3),
        local_window_length=ds.get("window_length", 8),
        local_window_stride=ds.get("window_stride", 4),
        self_state_dim=mdl.get("self_state_dim", 2),
        c1=mdl.get("c1", 8),
        c2=mdl.get("c2", 8),
        c3=mdl.get("c3", 8),
        s1=mdl.get("s1", 8),
        s2=mdl.get("s2", 8),
        s3=mdl.get("s3", 8),
        token_dim=mdl.get("token_dim", 64),
        self_state_hidden_dim=mdl.get("self_state_hidden_dim", 32),
        self_state_token_dim=mdl.get("self_state_token_dim", 32),
        gru_hidden_dim=mdl.get("gru_hidden_dim", 64),
        gru_layers=mdl.get("gru_layers", 1),
        policy_hidden_dim=mdl.get("policy_hidden_dim", 64),
        value_hidden_dim=mdl.get("value_hidden_dim", 64),
        warmup_head_hidden_dim=mdl.get("warmup_head_hidden_dim", 64),
        action_dim=mdl.get("action_dim", 2),
        freeze_local_encoder=mdl.get("freeze_local_encoder", True),
        freeze_local_dynamics=mdl.get("freeze_local_dynamics", False),
    )


def load_policy_weights(model: torch.nn.Module, checkpoint_path: Path) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "Policy" in ckpt:
        policy_state = ckpt["Policy"]
        mapped = {}
        for key, value in policy_state.items():
            if key.startswith("network_body."):
                mapped[key.replace("network_body.", "", 1)] = value
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        return {
            "source": "rl_policy",
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }

    state = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        "source": "direct_model",
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def load_frame(path: Path, grayscale: bool, img_size: int, out_channels: int) -> torch.Tensor:
    from PIL import Image

    if grayscale:
        image = Image.open(path).convert("L")
    else:
        image = Image.open(path).convert("RGB")
    image = image.resize((img_size, img_size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    if grayscale and out_channels > 1:
        arr = np.repeat(arr[..., :1], out_channels, axis=2)
    elif grayscale:
        arr = arr[..., :1]
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def list_episode_ids(root: Path) -> List[str]:
    return sorted(
        [
            p.name
            for p in root.iterdir()
            if p.is_dir() and (p / "frames.csv").exists() and (p / "frames").exists()
        ]
    )


def split_episode_ids(ids: List[str], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    rng = np.random.default_rng(seed)
    ids = ids[:]
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(n_train, 1), n - 2) if n >= 3 else max(n, 1)
    remain = n - n_train
    n_val = min(max(n_val, 1), max(1, remain - 1)) if remain >= 2 else remain
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    if not test_ids and val_ids:
        test_ids = [val_ids[-1]]
        val_ids = val_ids[:-1]
    return train_ids, val_ids, test_ids


def build_window_starts(num_frames: int, window_length: int, window_stride: int) -> List[int]:
    starts = list(range(0, num_frames - window_length + 1, window_stride))
    last_start = num_frames - window_length
    if starts and starts[-1] != last_start:
        starts.append(last_start)
    elif not starts and num_frames >= window_length:
        starts = [0]
    return starts


@torch.no_grad()
def extract_episode_representations(
    model: EmbodiedRecurrentPolicy,
    ep_dir: Path,
    img_size: int,
    grayscale: bool,
    out_channels: int,
    window_length: int,
    window_stride: int,
    forward_speed_scale: float,
    lateral_speed_scale: float,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    df = pd.read_csv(ep_dir / "frames.csv")
    frame_files = sorted((ep_dir / "frames").glob("*.png"))
    usable = min(len(df), len(frame_files))
    if usable < window_length:
        raise RuntimeError(f"Episode too short: {ep_dir}")

    df = df.iloc[:usable].reset_index(drop=True)
    frame_files = frame_files[:usable]
    starts = build_window_starts(usable, window_length, window_stride)
    cached_frames = torch.stack(
        [
            load_frame(path, grayscale=grayscale, img_size=img_size, out_channels=out_channels)
            for path in frame_files
        ],
        dim=0,
    )

    hidden = None
    rows = []
    for start in starts:
        end = start + window_length - 1
        window = cached_frames[start : start + window_length].unsqueeze(0).to(device)

        self_state = torch.tensor(
            [
                [
                    float(df.loc[end, "car_vz"]) / forward_speed_scale,
                    float(df.loc[end, "car_vx"]) / lateral_speed_scale,
                ]
            ],
            dtype=torch.float32,
            device=device,
        )
        step_out = model.step(window, self_state, hidden)
        hidden = step_out["hidden"]

        car_px = float(df.loc[end, "car_px"])
        car_pz = float(df.loc[end, "car_pz"])
        ball_px = float(df.loc[end, "ball_px"])
        ball_pz = float(df.loc[end, "ball_pz"])
        ball_vx = float(df.loc[end, "ball_vx"])
        ball_vz = float(df.loc[end, "ball_vz"])
        landing_px = float(df.loc[end, "landing_px"])
        landing_pz = float(df.loc[end, "landing_pz"])
        predicted_t = float(df.loc[end, "predicted_t"])
        sim_time = float(df.loc[end, "sim_time"]) if "sim_time" in df.columns else end * 0.02
        elapsed = sim_time - float(df.loc[0, "sim_time"]) if "sim_time" in df.columns else end * 0.02
        time_to_land = max(0.0, predicted_t - elapsed)

        rows.append(
            {
                "episode_id": ep_dir.name,
                "window_end": end,
                "short": step_out["short_token"].cpu().numpy()[0],
                "self": step_out["self_token"].cpu().numpy()[0],
                "memory": step_out["memory_token"].cpu().numpy()[0],
                "long": step_out["long_feature"].cpu().numpy()[0],
                "readout": np.concatenate(
                    [
                        step_out["short_token"].cpu().numpy()[0],
                        step_out["long_feature"].cpu().numpy()[0],
                        step_out["self_token"].cpu().numpy()[0],
                    ],
                    axis=0,
                ),
                "landing_offset": np.array([landing_px - car_px, landing_pz - car_pz], dtype=np.float32),
                "ball_rel_pos": np.array([ball_px - car_px, ball_pz - car_pz], dtype=np.float32),
                "ball_rel_vel": np.array([ball_vx, ball_vz], dtype=np.float32),
                "self_velocity": np.array([float(df.loc[end, "car_vx"]), float(df.loc[end, "car_vz"])], dtype=np.float32),
                "time_to_land": np.array([time_to_land], dtype=np.float32),
            }
        )

    collated = {}
    for key in ["short", "self", "memory", "long", "readout", "landing_offset", "ball_rel_pos", "ball_rel_vel", "self_velocity", "time_to_land"]:
        collated[key] = np.stack([row[key] for row in rows], axis=0)
    collated["episode_id"] = np.array([row["episode_id"] for row in rows])
    collated["window_end"] = np.array([row["window_end"] for row in rows], dtype=np.int64)
    return collated


def zscore_train_test(train_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def fit_linear_probe(train_x: np.ndarray, train_y: np.ndarray, reg: float = 1e-3) -> np.ndarray:
    x = np.concatenate([train_x, np.ones((train_x.shape[0], 1), dtype=train_x.dtype)], axis=1)
    xtx = x.T @ x
    xtx += reg * np.eye(xtx.shape[0], dtype=train_x.dtype)
    xty = x.T @ train_y
    w = np.linalg.solve(xtx, xty)
    return w


def predict_linear_probe(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x_aug @ w


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    numer = float(np.sum((y_true - y_pred) ** 2))
    r2 = 1.0 - numer / denom if denom > 1e-12 else 0.0
    return {"mse": mse, "mae": mae, "r2": r2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-episodes", type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    ds_cfg = cfg["dataset"]
    model_cfg = build_model_cfg(cfg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = EmbodiedRecurrentPolicy(model_cfg).to(device)
    load_report = load_policy_weights(model, Path(args.checkpoint))
    model.eval()

    root = Path(ds_cfg["root"])
    episode_ids = list_episode_ids(root)
    if args.max_episodes > 0:
        episode_ids = episode_ids[: args.max_episodes]
    train_ids, val_ids, test_ids = split_episode_ids(
        episode_ids,
        train_ratio=float(ds_cfg.get("train_ratio", 0.8)),
        val_ratio=float(ds_cfg.get("val_ratio", 0.1)),
        seed=int(ds_cfg.get("split_seed", 42)),
    )

    split_map = {"train": train_ids, "val": val_ids, "test": test_ids}
    split_repr: Dict[str, Dict[str, np.ndarray]] = {}

    for split_name, ids in split_map.items():
        collected = []
        for episode_id in ids:
            ep_repr = extract_episode_representations(
                model=model,
                ep_dir=root / episode_id,
                img_size=int(ds_cfg.get("img_size", 64)),
                grayscale=bool(ds_cfg.get("grayscale", True)),
                out_channels=int(ds_cfg.get("out_channels", 3)),
                window_length=int(ds_cfg.get("window_length", 8)),
                window_stride=int(ds_cfg.get("window_stride", 4)),
                forward_speed_scale=float(ds_cfg.get("forward_speed_scale", 8.0)),
                lateral_speed_scale=float(ds_cfg.get("lateral_speed_scale", 8.0)),
                device=device,
            )
            collected.append(ep_repr)

        merged: Dict[str, np.ndarray] = {}
        for key in collected[0].keys():
            merged[key] = np.concatenate([item[key] for item in collected], axis=0)
        split_repr[split_name] = merged

    feature_spaces = ["short", "self", "memory", "long", "readout"]
    targets = ["landing_offset", "ball_rel_pos", "ball_rel_vel", "self_velocity", "time_to_land"]
    probe_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for feature_name in feature_spaces:
        probe_results[feature_name] = {}
        train_x = split_repr["train"][feature_name].astype(np.float64)
        test_x = split_repr["test"][feature_name].astype(np.float64)
        train_x, test_x = zscore_train_test(train_x, test_x)
        for target_name in targets:
            train_y = split_repr["train"][target_name].astype(np.float64)
            test_y = split_repr["test"][target_name].astype(np.float64)
            weights = fit_linear_probe(train_x, train_y)
            pred = predict_linear_probe(test_x, weights)
            probe_results[feature_name][target_name] = regression_metrics(test_y, pred)

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
        "load_report": load_report,
        "model_config": asdict(model_cfg),
        "num_episodes": {k: len(v) for k, v in split_map.items()},
        "num_windows": {k: int(split_repr[k]["readout"].shape[0]) for k in split_repr},
        "probe_results": probe_results,
    }

    (output_dir / "spatial_probe_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Spatial Probe Summary",
        "",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- train/test episodes: `{summary['num_episodes']['train']}` / `{summary['num_episodes']['test']}`",
        f"- train/test windows: `{summary['num_windows']['train']}` / `{summary['num_windows']['test']}`",
        "",
        "## Test R2 by Representation",
        "",
        "| feature | landing_offset | ball_rel_pos | ball_rel_vel | self_velocity | time_to_land |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for feature_name in feature_spaces:
        lines.append(
            "| {feature} | {landing:.4f} | {ball_pos:.4f} | {ball_vel:.4f} | {self_vel:.4f} | {ttl:.4f} |".format(
                feature=feature_name,
                landing=probe_results[feature_name]["landing_offset"]["r2"],
                ball_pos=probe_results[feature_name]["ball_rel_pos"]["r2"],
                ball_vel=probe_results[feature_name]["ball_rel_vel"]["r2"],
                self_vel=probe_results[feature_name]["self_velocity"]["r2"],
                ttl=probe_results[feature_name]["time_to_land"]["r2"],
            )
        )
    (output_dir / "spatial_probe_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary["probe_results"], indent=2))


if __name__ == "__main__":
    main()
