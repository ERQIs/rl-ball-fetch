from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader


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
    "embodied_control_model_module",
    EXPERIMENT_ROOT / "src" / "models" / "embodied_recurrent_policy.py",
)
dataset_module = load_module(
    "embodied_control_warmup_dataset_module",
    EXPERIMENT_ROOT / "src" / "datasets" / "embodied_warmup_dataset.py",
)
seed_module = load_module(
    "temperal_seed_utils_for_embodied_warmup",
    REPO_ROOT / "temperal" / "src" / "utils" / "seed.py",
)

EmbodiedRecurrentPolicy = model_module.EmbodiedRecurrentPolicy
EmbodiedRecurrentPolicyConfig = model_module.EmbodiedRecurrentPolicyConfig
EmbodiedWarmupDataset = dataset_module.EmbodiedWarmupDataset
embodied_warmup_collate = dataset_module.embodied_warmup_collate
set_seed = seed_module.set_seed


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def choose_device(device_pref: str) -> torch.device:
    if device_pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_pref)


def list_trajectory_ids(root: Path) -> list[str]:
    ids = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "frames.csv").exists() and (path / "frames").exists():
            ids.append(path.name)
    if not ids:
        raise RuntimeError(f"No trajectory directories found under {root}")
    return ids


def split_ids(trajectory_ids: list[str], train_ratio: float, val_ratio: float, seed: int) -> tuple[list[str], list[str], list[str]]:
    ids = trajectory_ids[:]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    train_count = int(round(n * train_ratio))
    val_count = int(round(n * val_ratio))
    train_count = min(max(train_count, 1), n - 2) if n >= 3 else max(1, n)
    remaining = n - train_count
    val_count = min(max(val_count, 1), max(1, remaining - 1)) if remaining >= 2 else remaining
    test_count = max(0, n - train_count - val_count)
    train_ids = ids[:train_count]
    val_ids = ids[train_count : train_count + val_count]
    test_ids = ids[train_count + val_count : train_count + val_count + test_count]
    if not test_ids and val_ids:
        test_ids = [val_ids[-1]]
        val_ids = val_ids[:-1]
    return train_ids, val_ids, test_ids


def write_split_file(path: Path, episode_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(episode_ids) + "\n", encoding="utf-8")


def prepare_split_files(cfg: dict, out_dir: Path) -> dict:
    ds = cfg["dataset"]
    root = Path(ds["root"])
    explicit = {
        "train": ds.get("train_split_file"),
        "val": ds.get("val_split_file"),
        "test": ds.get("test_split_file"),
    }
    if all(explicit.values()):
        return {k: str(Path(v)) for k, v in explicit.items()}

    split_dir = out_dir / "splits"
    split_paths = {
        "train": split_dir / "train.txt",
        "val": split_dir / "val.txt",
        "test": split_dir / "test.txt",
    }
    if all(path.exists() for path in split_paths.values()):
        return {k: str(v) for k, v in split_paths.items()}

    trajectory_ids = list_trajectory_ids(root)
    train_ids, val_ids, test_ids = split_ids(
        trajectory_ids=trajectory_ids,
        train_ratio=float(ds.get("train_ratio", 0.8)),
        val_ratio=float(ds.get("val_ratio", 0.1)),
        seed=int(ds.get("split_seed", 42)),
    )
    write_split_file(split_paths["train"], train_ids)
    write_split_file(split_paths["val"], val_ids)
    write_split_file(split_paths["test"], test_ids)
    return {k: str(v) for k, v in split_paths.items()}


def build_dataset(ds_cfg: dict, split_file: str, split_name: str):
    return EmbodiedWarmupDataset(
        root=ds_cfg["root"],
        split_file=split_file,
        img_size=ds_cfg.get("img_size", 64),
        window_length=ds_cfg.get("window_length", 8),
        window_stride=ds_cfg.get("window_stride", 4),
        grayscale=ds_cfg.get("grayscale", True),
        out_channels=ds_cfg.get("out_channels", 3),
        forward_speed_scale=ds_cfg.get("forward_speed_scale", 8.0),
        lateral_speed_scale=ds_cfg.get("lateral_speed_scale", 8.0),
        max_episodes=ds_cfg.get(f"{split_name}_max_episodes"),
    )


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


def masked_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    mask = valid_mask.to(values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def compute_losses_and_metrics(outputs: dict, batch: dict, time_loss_weight: float) -> dict:
    valid_mask = outputs["valid_mask"]
    pred_landing = outputs["landing_offsets"]
    pred_time = outputs["time_to_land"]
    target_landing = batch["landing_offset"]
    target_time = batch["time_to_land"]

    landing_mse = ((pred_landing - target_landing) ** 2).mean(dim=-1)
    time_l1 = F.smooth_l1_loss(pred_time, target_time, reduction="none")
    landing_loss = masked_mean(landing_mse, valid_mask)
    time_loss = masked_mean(time_l1, valid_mask)
    total_loss = landing_loss + float(time_loss_weight) * time_loss

    landing_l2 = torch.linalg.norm(pred_landing - target_landing, dim=-1)
    time_abs = torch.abs(pred_time - target_time)
    return {
        "loss": total_loss,
        "landing_loss": landing_loss,
        "time_loss": time_loss,
        "landing_l2": masked_mean(landing_l2, valid_mask),
        "time_mae": masked_mean(time_abs, valid_mask),
    }


@torch.no_grad()
def evaluate(model, loader, device, time_loss_weight: float) -> dict:
    model.eval()
    totals = {
        "loss": 0.0,
        "landing_loss": 0.0,
        "time_loss": 0.0,
        "landing_l2": 0.0,
        "time_mae": 0.0,
    }
    n = 0
    for batch in loader:
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        outputs = model(batch["windows"], batch["self_state"], batch["num_windows"])
        stats = compute_losses_and_metrics(outputs, batch, time_loss_weight=time_loss_weight)
        bs = batch["windows"].shape[0]
        for key in totals:
            totals[key] += float(stats[key].item()) * bs
        n += bs
    denom = max(n, 1)
    return {k: v / denom for k, v in totals.items()}


def save_history(history: list[dict], out_dir: Path) -> None:
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    out_dir = Path(cfg["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    set_seed(int(cfg["training"].get("seed", 42)))
    device = choose_device(cfg["training"].get("device", "auto"))

    split_files = prepare_split_files(cfg, out_dir)
    train_ds = build_dataset(cfg["dataset"], split_files["train"], "train")
    val_ds = build_dataset(cfg["dataset"], split_files["val"], "val")
    test_ds = build_dataset(cfg["dataset"], split_files["test"], "test")

    batch_size = int(cfg["training"].get("batch_size", 4))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=embodied_warmup_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=embodied_warmup_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=embodied_warmup_collate,
    )

    model_cfg = build_model_cfg(cfg)
    model = EmbodiedRecurrentPolicy(model_cfg)
    model.set_local_trainable(
        encoder_trainable=not model_cfg.freeze_local_encoder,
        dynamics_trainable=not model_cfg.freeze_local_dynamics,
    )

    init_ckpt = cfg["model"].get("init_checkpoint")
    init_mode = cfg["model"].get("init_mode", "hierarchical")
    init_report = None
    if init_ckpt:
        if init_mode == "hierarchical":
            init_report = model.load_pretrained_hierarchical(
                init_ckpt,
                strict=bool(cfg["model"].get("init_strict", False)),
                load_long_memory=bool(cfg["model"].get("load_long_memory", True)),
            )
        elif init_mode == "local_backbone":
            model.load_pretrained_local_backbone(init_ckpt, strict=bool(cfg["model"].get("init_strict", True)))
            init_report = {"mode": "local_backbone"}
        else:
            raise ValueError(f"Unsupported init_mode: {init_mode}")

    model = model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg["training"].get("lr", 3e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )

    epochs = int(cfg["training"].get("epochs", 20))
    time_loss_weight = float(cfg["training"].get("time_loss_weight", 0.25))
    history = []
    best_val = float("inf")
    best_epoch = 0

    print("started embodied offline warmup")
    print("device:", device)
    print("train/val/test episodes:", len(train_ds), len(val_ds), len(test_ds))
    print("freeze_local_encoder:", model.freeze_local_encoder)
    print("freeze_local_dynamics:", model.freeze_local_dynamics)
    if init_ckpt:
        print("loaded init checkpoint:", init_ckpt)
        if init_report is not None:
            print("init report:", json.dumps(init_report, indent=2))

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "landing_loss": 0.0,
            "time_loss": 0.0,
            "landing_l2": 0.0,
            "time_mae": 0.0,
        }
        n = 0

        for batch in train_loader:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            outputs = model(batch["windows"], batch["self_state"], batch["num_windows"])
            stats = compute_losses_and_metrics(outputs, batch, time_loss_weight=time_loss_weight)

            optimizer.zero_grad()
            stats["loss"].backward()
            optimizer.step()

            bs = batch["windows"].shape[0]
            for key in totals:
                totals[key] += float(stats[key].item()) * bs
            n += bs

        denom = max(n, 1)
        train_stats = {k: v / denom for k, v in totals.items()}
        val_stats = evaluate(model, val_loader, device, time_loss_weight=time_loss_weight)
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_landing_loss": train_stats["landing_loss"],
            "train_time_loss": train_stats["time_loss"],
            "train_landing_l2": train_stats["landing_l2"],
            "train_time_mae": train_stats["time_mae"],
            "val_loss": val_stats["loss"],
            "val_landing_loss": val_stats["landing_loss"],
            "val_time_loss": val_stats["time_loss"],
            "val_landing_l2": val_stats["landing_l2"],
            "val_time_mae": val_stats["time_mae"],
        }
        history.append(row)
        save_history(history, out_dir)
        print(
            f"epoch {epoch:03d} | "
            f"train_loss={row['train_loss']:.4f} train_l2={row['train_landing_l2']:.4f} train_t={row['train_time_mae']:.4f} | "
            f"val_loss={row['val_loss']:.4f} val_l2={row['val_landing_l2']:.4f} val_t={row['val_time_mae']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": cfg,
            "model_config": asdict(model_cfg),
            "init_checkpoint": init_ckpt,
            "init_mode": init_mode,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_epoch = epoch
            torch.save(ckpt, out_dir / "best.pt")

    best_ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    val_stats = evaluate(model, val_loader, device, time_loss_weight=time_loss_weight)
    test_stats = evaluate(model, test_loader, device, time_loss_weight=time_loss_weight)
    summary = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "val_loss": float(val_stats["loss"]),
        "val_landing_loss": float(val_stats["landing_loss"]),
        "val_time_loss": float(val_stats["time_loss"]),
        "val_landing_l2": float(val_stats["landing_l2"]),
        "val_time_mae": float(val_stats["time_mae"]),
        "test_loss": float(test_stats["loss"]),
        "test_landing_loss": float(test_stats["landing_loss"]),
        "test_time_loss": float(test_stats["time_loss"]),
        "test_landing_l2": float(test_stats["landing_l2"]),
        "test_time_mae": float(test_stats["time_mae"]),
        "train_episodes": int(len(train_ds)),
        "val_episodes": int(len(val_ds)),
        "test_episodes": int(len(test_ds)),
        "window_length": int(cfg["dataset"].get("window_length", 8)),
        "window_stride": int(cfg["dataset"].get("window_stride", 4)),
        "init_checkpoint": init_ckpt,
        "init_mode": init_mode,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
