from __future__ import annotations

import argparse
import csv
import importlib.util
import json
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
    "landing_world_model_fixed_rollout_model",
    EXPERIMENT_ROOT / "src" / "models" / "fixed_rollout_landing_head.py",
)
dataset_module = load_module(
    "landing_world_model_dataset_train",
    EXPERIMENT_ROOT / "src" / "datasets" / "landing_rollout_dataset.py",
)
seed_module = load_module(
    "temperal_seed_utils_for_landing",
    REPO_ROOT / "temperal" / "src" / "utils" / "seed.py",
)

LandingRolloutConfig = model_module.LandingRolloutConfig
FixedRolloutLandingHeadModel = model_module.FixedRolloutLandingHeadModel
LandingRolloutDataset = dataset_module.LandingRolloutDataset
set_seed = seed_module.set_seed


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def choose_device(device_pref: str) -> torch.device:
    if device_pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_pref)


def build_dataset(cfg, split: str):
    ds = cfg["dataset"]
    return LandingRolloutDataset(
        root=ds["root"],
        split_file=ds[f"{split}_split_file"],
        img_size=ds.get("img_size", 64),
        observation_length=ds.get("observation_length", 8),
        observation_end_fraction=ds.get("observation_end_fraction", 0.5),
        sampling_mode=ds.get("sampling_mode", "uniform_visible"),
        grayscale=ds.get("grayscale", True),
        out_channels=ds.get("out_channels", 3),
        max_episodes=ds.get(f"{split}_max_episodes"),
    )


def build_model_cfg(cfg) -> LandingRolloutConfig:
    ds = cfg["dataset"]
    mdl = cfg["model"]
    return LandingRolloutConfig(
        image_h=ds.get("img_size", 64),
        image_w=ds.get("img_size", 64),
        in_channels=ds.get("out_channels", 3),
        observation_length=ds.get("observation_length", 8),
        rollout_steps=mdl.get("rollout_steps", 36),
        c1=mdl.get("c1", 8),
        c2=mdl.get("c2", 8),
        c3=mdl.get("c3", 8),
        s1=mdl.get("s1", 8),
        s2=mdl.get("s2", 8),
        s3=mdl.get("s3", 8),
        head_hidden_dim=mdl.get("head_hidden_dim", 64),
        out_dim=2,
        pre_head_layernorm=mdl.get("pre_head_layernorm", False),
    )


def build_target(batch) -> torch.Tensor:
    landing_xyz = batch["landing_xyz"]
    return torch.stack([landing_xyz[:, 0], landing_xyz[:, 2]], dim=1)


def compute_metrics(pred_xz: torch.Tensor, target_xz: torch.Tensor, success_threshold: float):
    diff = pred_xz - target_xz
    l2 = torch.linalg.norm(diff, dim=1)
    return {
        "loss": F.mse_loss(pred_xz, target_xz),
        "mean_l2": float(l2.mean().item()),
        "success": float((l2 <= success_threshold).float().mean().item()),
    }


@torch.no_grad()
def evaluate(model, loader, device, success_threshold: float):
    model.eval()
    agg_loss = 0.0
    agg_l2 = 0.0
    agg_success = 0.0
    n = 0
    rows = []
    for batch in loader:
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        target_xz = build_target(batch)
        pred_xz = model(batch["frames"])
        stats = compute_metrics(pred_xz, target_xz, success_threshold)
        bs = pred_xz.shape[0]
        agg_loss += float(stats["loss"].item()) * bs
        agg_l2 += stats["mean_l2"] * bs
        agg_success += stats["success"] * bs
        n += bs
        pred_np = pred_xz.detach().cpu().numpy()
        target_np = target_xz.detach().cpu().numpy()
        for i in range(bs):
            rows.append(
                {
                    "episode_id": batch["episode_id"][i],
                    "pred_landing_px": float(pred_np[i, 0]),
                    "pred_landing_pz": float(pred_np[i, 1]),
                    "true_landing_px": float(target_np[i, 0]),
                    "true_landing_pz": float(target_np[i, 1]),
                    "landing_xz_l2": float(np.linalg.norm(pred_np[i] - target_np[i])),
                }
            )
    denom = max(n, 1)
    return {
        "loss": agg_loss / denom,
        "mean_l2": agg_l2 / denom,
        "success": agg_success / denom,
        "rows": rows,
    }


def save_history(history, out_dir):
    path = out_dir / "history.csv"
    fieldnames = ["epoch", "train_loss", "train_mean_l2", "train_success", "val_loss", "val_mean_l2", "val_success"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_detailed(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_curve_svg(history, out_path):
    if not history:
        return
    width, height = 900, 520
    ml, mr, mt, mb = 70, 40, 35, 55
    pw, ph = width - ml - mr, height - mt - mb
    epochs = [row["epoch"] for row in history]
    train_vals = [row["train_mean_l2"] for row in history]
    val_vals = [row["val_mean_l2"] for row in history]
    y_min, y_max = min(train_vals + val_vals), max(train_vals + val_vals)
    if abs(y_max - y_min) < 1e-8:
        y_max = y_min + 1.0

    def x_pos(epoch):
        if len(epochs) == 1:
            return ml + pw / 2
        return ml + (epoch - epochs[0]) / (epochs[-1] - epochs[0]) * pw

    def y_pos(value):
        return mt + (y_max - value) / (y_max - y_min) * ph

    def polyline(key):
        return " ".join(f"{x_pos(row['epoch']):.2f},{y_pos(row[key]):.2f}" for row in history)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#333;stroke-width:1.5}.train{fill:none;stroke:#2563eb;stroke-width:2.5}.val{fill:none;stroke:#dc2626;stroke-width:2.5}.title{font-size:18px;font-weight:bold}</style>',
        f'<text x="{width/2:.0f}" y="22" text-anchor="middle" class="title">Rollout36 Landing Head Mean L2</text>',
    ]
    for tick in np.linspace(y_min, y_max, num=5):
        y = y_pos(float(tick))
        svg.append(f'<line x1="{ml}" y1="{y:.2f}" x2="{width-mr}" y2="{y:.2f}" class="grid" />')
        svg.append(f'<text x="{ml-10}" y="{y+4:.2f}" text-anchor="end">{tick:.4f}</text>')
    for tick in sorted(set([epochs[0], epochs[-1], max(1, epochs[-1] // 2)])):
        x = x_pos(tick)
        svg.append(f'<line x1="{x:.2f}" y1="{mt}" x2="{x:.2f}" y2="{height-mb}" class="grid" />')
        svg.append(f'<text x="{x:.2f}" y="{height-mb+20}" text-anchor="middle">{tick}</text>')
    svg.extend(
        [
            f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" class="axis" />',
            f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" class="axis" />',
            f'<text x="{width/2:.0f}" y="{height-15}" text-anchor="middle">Epoch</text>',
            f'<text x="18" y="{height/2:.0f}" text-anchor="middle" transform="rotate(-90 18 {height/2:.0f})">Mean L2</text>',
            f'<polyline points="{polyline("train_mean_l2")}" class="train" />',
            f'<polyline points="{polyline("val_mean_l2")}" class="val" />',
            f'<line x1="{width-mr-170}" y1="{mt+18}" x2="{width-mr-146}" y2="{mt+18}" class="train" />',
            f'<text x="{width-mr-138}" y="{mt+22}">train_mean_l2</text>',
            f'<line x1="{width-mr-170}" y1="{mt+42}" x2="{width-mr-146}" y2="{mt+42}" class="val" />',
            f'<text x="{width-mr-138}" y="{mt+46}">val_mean_l2</text>',
            "</svg>",
        ]
    )
    out_path.write_text("\n".join(svg), encoding="utf-8")


def maybe_resume_training(model, optimizer, out_dir: Path, device: torch.device, resume: bool):
    if not resume:
        return 1, float("inf"), [], None
    ckpt_path = out_dir / "last.pt"
    if not ckpt_path.exists():
        print("resume requested but no last.pt found in", out_dir)
        return 1, float("inf"), [], None
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    history = ckpt.get("history", [])
    last_epoch = int(ckpt.get("epoch", len(history)))
    best_val = min((row["val_mean_l2"] for row in history), default=float("inf"))
    init_checkpoint = ckpt.get("init_checkpoint")
    print("resumed training from", ckpt_path)
    print("resume epoch:", last_epoch)
    print("resume best val mean_l2:", best_val)
    return last_epoch + 1, best_val, history, init_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    out_dir = Path(cfg["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    set_seed(int(cfg["training"].get("seed", 42)))
    device = choose_device(cfg["training"].get("device", "auto"))
    success_threshold = float(cfg["evaluation"].get("success_threshold", 0.2))

    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    test_ds = build_dataset(cfg, "test")
    batch_size = int(cfg["training"].get("batch_size", 8))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model_cfg = build_model_cfg(cfg)
    model = FixedRolloutLandingHeadModel(model_cfg).to(device)
    init_ckpt = cfg["model"].get("init_checkpoint")
    start_epoch = 1
    history = []
    best_val = float("inf")
    best_epoch = None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("lr", 3e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )

    if args.resume:
        start_epoch, best_val, history, resume_init_ckpt = maybe_resume_training(model, optimizer, out_dir, device, resume=True)
        if resume_init_ckpt:
            init_ckpt = resume_init_ckpt
        if history:
            best_epoch = min(history, key=lambda row: row["val_mean_l2"])["epoch"]
    elif init_ckpt:
        model.load_pretrained_backbone(init_ckpt, strict=bool(cfg["model"].get("init_strict", True)))
        print("loaded pretrained encoder+dynamics from", init_ckpt)

    epochs = int(cfg["training"].get("epochs", 20))

    print("started rollout36 landing-head training")
    print("device:", device)
    print("train/val/test episodes:", len(train_ds), len(val_ds), len(test_ds))
    if start_epoch > epochs:
        print(f"nothing to do: start_epoch={start_epoch} > epochs={epochs}")
        return

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        agg_loss = 0.0
        agg_l2 = 0.0
        agg_success = 0.0
        n = 0

        for batch in train_loader:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            target_xz = build_target(batch)
            pred_xz = model(batch["frames"])
            loss = F.mse_loss(pred_xz, target_xz)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                stats = compute_metrics(pred_xz, target_xz, success_threshold)
            bs = pred_xz.shape[0]
            agg_loss += float(loss.item()) * bs
            agg_l2 += stats["mean_l2"] * bs
            agg_success += stats["success"] * bs
            n += bs

        train_stats = {
            "loss": agg_loss / max(n, 1),
            "mean_l2": agg_l2 / max(n, 1),
            "success": agg_success / max(n, 1),
        }
        val_stats = evaluate(model, val_loader, device, success_threshold)
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_mean_l2": train_stats["mean_l2"],
            "train_success": train_stats["success"],
            "val_loss": val_stats["loss"],
            "val_mean_l2": val_stats["mean_l2"],
            "val_success": val_stats["success"],
        }
        history.append(row)
        save_history(history, out_dir)
        save_curve_svg(history, out_dir / "mean_l2_curve.svg")
        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_stats['loss']:.4f} train_l2={train_stats['mean_l2']:.4f} train_s={train_stats['success']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} val_l2={val_stats['mean_l2']:.4f} val_s={val_stats['success']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": cfg,
            "model_config": asdict(model_cfg),
            "init_checkpoint": init_ckpt,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_stats["mean_l2"] < best_val:
            best_val = val_stats["mean_l2"]
            best_epoch = epoch
            torch.save(ckpt, out_dir / "best.pt")

    best_ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    val_stats = evaluate(model, val_loader, device, success_threshold)
    test_stats = evaluate(model, test_loader, device, success_threshold)
    save_detailed(val_stats["rows"], out_dir / "val_predictions.csv")
    save_detailed(test_stats["rows"], out_dir / "test_predictions.csv")

    summary = {
        "best_epoch": int(best_epoch or 0),
        "best_val_mean_l2": float(best_val),
        "val_loss": float(val_stats["loss"]),
        "val_mean_l2": float(val_stats["mean_l2"]),
        "val_success": float(val_stats["success"]),
        "test_loss": float(test_stats["loss"]),
        "test_mean_l2": float(test_stats["mean_l2"]),
        "test_success": float(test_stats["success"]),
        "rollout_steps": int(model_cfg.rollout_steps),
        "init_checkpoint": init_ckpt,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
