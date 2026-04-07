import argparse
import csv
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from src.datasets.trajectory_dataset import TrajectoryDataset
from src.models.multiscale_transfer_regressor import MultiScaleTransferRegressor, TransferConfig
from src.utils.seed import set_seed


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(cfg, split):
    ds = cfg["dataset"]
    dataset = TrajectoryDataset(
        root=ds["root"],
        split_file=ds[f"{split}_split_file"],
        img_size=ds.get("img_size", 64),
        observation_length=ds.get("observation_length", 8),
        frame_stride=ds.get("frame_stride", 1),
        use_last_n_frames=ds.get("use_last_n_frames", False),
        observation_end_fraction=ds.get("observation_end_fraction", 1.0),
        sampling_mode=ds.get("sampling_mode", "uniform_visible"),
    )
    fraction = float(ds.get("train_fraction", 1.0)) if split == "train" else 1.0
    if split == "train" and fraction < 1.0:
        n = max(1, int(len(dataset) * fraction))
        rng = np.random.default_rng(int(cfg["training"].get("seed", 42)))
        indices = np.sort(rng.choice(len(dataset), size=n, replace=False)).tolist()
        dataset = Subset(dataset, indices)
    return dataset


def build_model_cfg(cfg):
    ds = cfg["dataset"]
    mdl = cfg["model"]
    return TransferConfig(
        image_h=ds.get("img_size", 64),
        image_w=ds.get("img_size", 64),
        in_channels=3,
        observation_length=ds.get("observation_length", 8),
        c1=mdl.get("c1", 8),
        c2=mdl.get("c2", 8),
        c3=mdl.get("c3", 8),
        s1=mdl.get("s1", 8),
        s2=mdl.get("s2", 8),
        s3=mdl.get("s3", 8),
        head_hidden_dim=mdl.get("head_hidden_dim", 64),
        out_dim=2,
        pre_head_layernorm=bool(mdl.get("pre_head_layernorm", False)),
    )


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    losses = []
    preds_all = []
    targets_all = []
    for batch in loader:
        frames = batch["frames"].to(device)
        targets = batch["target_xy"].to(device)
        preds = model(frames)
        loss = loss_fn(preds, targets)
        losses.append(loss.item())
        preds_all.append(preds.cpu().numpy())
        targets_all.append(targets.cpu().numpy())
    preds = np.concatenate(preds_all, axis=0)
    targets = np.concatenate(targets_all, axis=0)
    l2 = np.linalg.norm(preds - targets, axis=1)
    return {
        "loss": float(np.mean(losses)),
        "mean_l2": float(np.mean(l2)),
        "median_l2": float(np.median(l2)),
        "success_at_0.2": float((l2 <= 0.2).mean()),
    }


def save_history(history, out_dir):
    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_mean_l2", "val_median_l2", "val_success_at_0.2"],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_curve_svg(history, out_path, title, y_label, keys_and_colors):
    if not history:
        return

    width = 800
    height = 480
    margin_left = 70
    margin_right = 30
    margin_top = 30
    margin_bottom = 55
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    epochs = [row["epoch"] for row in history]
    y_values = []
    present_series = []
    for key, color, label in keys_and_colors:
        series = [row[key] for row in history if row[key] is not None]
        if series:
            y_values.extend(series)
            present_series.append((key, color, label))
    if not y_values:
        return
    y_min = min(y_values)
    y_max = max(y_values)
    if abs(y_max - y_min) < 1e-8:
        y_max = y_min + 1.0

    def x_pos(epoch):
        if len(epochs) == 1:
            return margin_left + plot_w / 2.0
        return margin_left + (epoch - epochs[0]) / (epochs[-1] - epochs[0]) * plot_w

    def y_pos(value):
        return margin_top + (y_max - value) / (y_max - y_min) * plot_h

    def polyline(values_key):
        pts = []
        for row in history:
            value = row[values_key]
            if value is None:
                continue
            pts.append(f"{x_pos(row['epoch']):.2f},{y_pos(value):.2f}")
        return " ".join(pts)

    x_ticks = sorted(set([epochs[0], epochs[-1], max(1, epochs[-1] // 2)]))
    y_ticks = np.linspace(y_min, y_max, num=5)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial, sans-serif;font-size:12px;fill:#222} .grid{stroke:#ddd;stroke-width:1} .axis{stroke:#333;stroke-width:1.5} .title{font-size:18px;font-weight:bold}</style>',
        f'<text x="{width/2:.0f}" y="20" text-anchor="middle" class="title">{title}</text>',
    ]

    for tick in y_ticks:
        y = y_pos(float(tick))
        lines.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width-margin_right}" y2="{y:.2f}" class="grid" />')
        lines.append(f'<text x="{margin_left-10}" y="{y+4:.2f}" text-anchor="end">{tick:.4f}</text>')

    for tick in x_ticks:
        x = x_pos(tick)
        lines.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height-margin_bottom}" class="grid" />')
        lines.append(f'<text x="{x:.2f}" y="{height-margin_bottom+20}" text-anchor="middle">{tick}</text>')

    lines.extend([
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" class="axis" />',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" class="axis" />',
        f'<text x="{width/2:.0f}" y="{height-15}" text-anchor="middle">Epoch</text>',
        f'<text x="18" y="{height/2:.0f}" text-anchor="middle" transform="rotate(-90 18 {height/2:.0f})">{y_label}</text>',
    ])

    legend_x = width - margin_right - 140
    legend_y = margin_top + 10
    for idx, (key, color, label) in enumerate(present_series):
        class_name = f"series{idx}"
        lines.insert(1, f'<style>.{class_name}{{fill:none;stroke:{color};stroke-width:2.5}}</style>')
        lines.append(f'<polyline points="{polyline(key)}" class="{class_name}" />')
        lines.extend([
            f'<line x1="{legend_x}" y1="{legend_y + 24 * idx}" x2="{legend_x+24}" y2="{legend_y + 24 * idx}" class="{class_name}" />',
            f'<text x="{legend_x+32}" y="{legend_y + 24 * idx + 4}">{label}</text>',
        ])

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_metrics_json(metrics, path):
    serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in metrics.items()}
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def build_runtime(cfg):
    tr_cfg = cfg["training"]
    set_seed(int(tr_cfg.get("seed", 42)))
    device_pref = tr_cfg.get("device", "auto")
    if device_pref == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_pref if device_pref in ["cpu", "cuda"] else "cpu")

    model_cfg = build_model_cfg(cfg)
    model = MultiScaleTransferRegressor(model_cfg)

    init_mode = cfg["model"].get("init_mode", "scratch")
    pretrained_ckpt = cfg["model"].get("pretrained_checkpoint", "")
    if init_mode in ["frozen", "finetune"]:
        if not pretrained_ckpt:
            raise RuntimeError("pretrained_checkpoint is required for frozen/finetune mode")
        model.load_pretrained_backbone(pretrained_ckpt)
        model.set_backbone_trainable(init_mode == "finetune")

    model.to(device)
    return device, model_cfg, model


def build_optimizer(model, cfg):
    tr_cfg = cfg["training"]
    return torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(tr_cfg.get("lr", 1e-3)),
        weight_decay=float(tr_cfg.get("weight_decay", 0.0)),
    )


@torch.no_grad()
def evaluate_checkpoint(cfg, checkpoint_path, split):
    cfg_local = deepcopy(cfg)
    device, model_cfg, model = build_runtime(cfg_local)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict") or ckpt.get("model")
    if state_dict is None:
        raise RuntimeError(f"checkpoint {checkpoint_path} does not contain model weights")
    model.load_state_dict(state_dict, strict=True)
    ds = build_dataset(cfg_local, split)
    loader = DataLoader(ds, batch_size=int(cfg_local["training"].get("batch_size", 8)), shuffle=False, num_workers=0)
    metrics = evaluate(model, loader, device, torch.nn.SmoothL1Loss())
    metrics["split"] = split
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["num_samples"] = int(len(ds))
    metrics["init_mode"] = cfg_local["model"].get("init_mode", "scratch")
    metrics["train_fraction"] = float(cfg_local["dataset"].get("train_fraction", 1.0))
    metrics["observation_end_fraction"] = float(cfg_local["dataset"].get("observation_end_fraction", 1.0))
    metrics["seed"] = int(cfg_local["training"].get("seed", 42))
    metrics["model_out_dim"] = int(model_cfg.out_dim)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    cfg = load_cfg(args.config)
    tr_cfg = cfg["training"]
    out_dir = Path(tr_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(tr_cfg.get("tensorboard_dir", out_dir / "tb"))
    tb_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    device, model_cfg, model = build_runtime(cfg)

    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    train_loader = DataLoader(train_ds, batch_size=int(tr_cfg.get("batch_size", 8)), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(tr_cfg.get("batch_size", 8)), shuffle=False, num_workers=0)
    init_mode = cfg["model"].get("init_mode", "scratch")
    freeze_backbone_epochs = int(tr_cfg.get("freeze_backbone_epochs", 0))
    backbone_is_trainable = cfg["model"].get("init_mode", "scratch") != "finetune" or freeze_backbone_epochs <= 0
    if cfg["model"].get("init_mode", "scratch") == "finetune" and freeze_backbone_epochs > 0:
        model.set_backbone_trainable(False)
    optimizer = build_optimizer(model, cfg)
    loss_fn = torch.nn.SmoothL1Loss()
    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("config/yaml", yaml.safe_dump(cfg, sort_keys=False))

    best_val = float("inf")
    history = []
    epochs = int(tr_cfg.get("epochs", 20))
    log_every = int(tr_cfg.get("log_every_batches", 20))

    for epoch in range(1, epochs + 1):
        if cfg["model"].get("init_mode", "scratch") == "finetune" and freeze_backbone_epochs > 0:
            should_train_backbone = epoch > freeze_backbone_epochs
            if should_train_backbone != backbone_is_trainable:
                model.set_backbone_trainable(should_train_backbone)
                optimizer = build_optimizer(model, cfg)
                backbone_is_trainable = should_train_backbone

        model.train()
        losses = []
        total_batches = len(train_loader)
        running = []
        print(f"epoch {epoch:03d} started | batches={total_batches} | device={device} | mode={init_mode}")
        for batch_idx, batch in enumerate(train_loader, start=1):
            frames = batch["frames"].to(device)
            targets = batch["target_xy"].to(device)
            preds = model(frames)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            value = loss.item()
            losses.append(value)
            running.append(value)
            if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches:
                pct = 100.0 * batch_idx / max(total_batches, 1)
                print(f"  epoch {epoch:03d} progress {batch_idx}/{total_batches} ({pct:.1f}%) | recent train_loss={float(np.mean(running)):.4f}")
                running = []

        train_loss = float(np.mean(losses))
        val_stats = evaluate(model, val_loader, device, loss_fn)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_stats["loss"],
            "val_mean_l2": val_stats["mean_l2"],
            "val_median_l2": val_stats["median_l2"],
            "val_success_at_0.2": val_stats["success_at_0.2"],
        }
        history.append(row)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_stats["loss"], epoch)
        writer.add_scalar("metric/val_mean_l2", val_stats["mean_l2"], epoch)
        writer.add_scalar("metric/val_median_l2", val_stats["median_l2"], epoch)
        writer.add_scalar("metric/val_success_at_0.2", val_stats["success_at_0.2"], epoch)

        print(
            f"epoch {epoch:03d} done | train_loss={train_loss:.4f} val_loss={val_stats['loss']:.4f} "
            f"val_mean_l2={val_stats['mean_l2']:.4f} val_success@0.2={val_stats['success_at_0.2']:.4f}"
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "model": model.state_dict(),
            "config": cfg,
            "transfer_config": asdict(model_cfg),
            "epoch": epoch,
            "history": history,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(ckpt, out_dir / "best.pt")
            save_metrics_json(val_stats, out_dir / "best_val_metrics.json")
        save_history(history, out_dir)
        save_curve_svg(
            history,
            out_dir / "loss_curve.svg",
            title="Loss Curve",
            y_label="Loss",
            keys_and_colors=[
                ("train_loss", "#2563eb", "train_loss"),
                ("val_loss", "#dc2626", "val_loss"),
            ],
        )
        save_curve_svg(
            history,
            out_dir / "success_curve.svg",
            title="Success Curve @ 0.2",
            y_label="Success Rate",
            keys_and_colors=[
                ("val_success_at_0.2", "#059669", "val_success@0.2"),
            ],
        )

    writer.flush()
    writer.close()
    print("Best validation loss:", best_val)
    print("Saved to", out_dir)


if __name__ == "__main__":
    main()
