import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = THIS_DIR.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.datasets.frame_pair_dataset import FramePairDataset
from src.models.multiscale_spatial_prior import (
    MultiScaleSpatialPriorConfig,
    MultiScaleSpatialPriorModel,
    compute_spatial_prior_losses,
)
from src.utils.seed import set_seed


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(cfg, split):
    ds = cfg["dataset"]
    return FramePairDataset(
        root=ds["root"],
        split_file=ds[f"{split}_split_file"],
        img_size=ds.get("img_size", 64),
        grayscale=ds.get("grayscale", True),
        out_channels=ds.get("out_channels", 3),
        pair_step=ds.get("pair_step", 1),
        max_pairs=ds.get(f"{split}_max_pairs"),
    )


def build_model_cfg(cfg):
    ds = cfg["dataset"]
    mdl = cfg["model"]
    loss = cfg["loss"]
    return MultiScaleSpatialPriorConfig(
        image_h=ds.get("img_size", 64),
        image_w=ds.get("img_size", 64),
        in_channels=ds.get("out_channels", 3),
        c1=mdl.get("c1", 16),
        c2=mdl.get("c2", 32),
        c3=mdl.get("c3", 64),
        decoder_mode=mdl.get("decoder_mode", "f3"),
        lambda_rec=loss.get("lambda_rec", 1.0),
        lambda_trans_f2=loss.get("lambda_trans_f2", 0.15),
        lambda_trans=loss.get("lambda_trans", 1.0),
        lambda_wd=loss.get("lambda_wd", 1.0),
        lambda_nb=loss.get("lambda_nb", 0.1),
        foreground_weight=loss.get("foreground_weight", 6.0),
        foreground_dark_threshold=loss.get("foreground_dark_threshold", 0.55),
        foreground_dilate_kernel=loss.get("foreground_dilate_kernel", 5),
    )


@torch.no_grad()
def evaluate(model, loader, model_cfg, device):
    model.eval()
    agg = {"l_total": 0.0, "l_rec": 0.0, "l_trans_f2": 0.0, "l_trans": 0.0, "l_wd": 0.0, "l_nb": 0.0, "n": 0}
    for batch in loader:
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        outputs = model(batch["i_t"], batch["i_t1"], batch["flow_t"])
        _, stats = compute_spatial_prior_losses(outputs, batch["i_t"], batch["i_t1"], model_cfg)
        for key in ["l_total", "l_rec", "l_trans_f2", "l_trans", "l_wd", "l_nb"]:
            agg[key] += stats[key]
        agg["n"] += 1
    for key in ["l_total", "l_rec", "l_trans_f2", "l_trans", "l_wd", "l_nb"]:
        agg[key] /= max(agg["n"], 1)
    return agg


def save_history(history, out_dir):
    path = out_dir / "history.csv"
    fieldnames = [
        "epoch",
        "train_total",
        "train_rec",
        "train_trans_f2",
        "train_trans",
        "train_wd",
        "train_nb",
        "val_total",
        "val_rec",
        "val_trans_f2",
        "val_trans",
        "val_wd",
        "val_nb",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_batch_log(rows, out_dir):
    path = out_dir / "batch_log.csv"
    fieldnames = [
        "epoch",
        "batch_idx",
        "total_batches",
        "progress_pct",
        "recent_total",
        "recent_rec",
        "recent_trans_f2",
        "recent_trans",
        "recent_wd",
        "recent_nb",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_curve_svg(history, out_path):
    if not history:
        return
    width, height = 800, 480
    ml, mr, mt, mb = 70, 30, 30, 55
    pw, ph = width - ml - mr, height - mt - mb
    epochs = [row["epoch"] for row in history]
    train_vals = [row["train_total"] for row in history]
    val_vals = [row["val_total"] for row in history]
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
        f'<text x="{width/2:.0f}" y="20" text-anchor="middle" class="title">Spatial Prior Loss Curve</text>',
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
            f'<text x="18" y="{height/2:.0f}" text-anchor="middle" transform="rotate(-90 18 {height/2:.0f})">Loss</text>',
            f'<polyline points="{polyline("train_total")}" class="train" />',
            f'<polyline points="{polyline("val_total")}" class="val" />',
            f'<line x1="{width-mr-140}" y1="{mt+18}" x2="{width-mr-116}" y2="{mt+18}" class="train" />',
            f'<text x="{width-mr-108}" y="{mt+22}">train_total</text>',
            f'<line x1="{width-mr-140}" y1="{mt+42}" x2="{width-mr-116}" y2="{mt+42}" class="val" />',
            f'<text x="{width-mr-108}" y="{mt+46}">val_total</text>',
            "</svg>",
        ]
    )
    out_path.write_text("\n".join(svg), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    tr_cfg = cfg["training"]
    out_dir = Path(tr_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(tr_cfg.get("tensorboard_dir", out_dir / "tb"))
    tb_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    set_seed(int(tr_cfg.get("seed", 42)))
    if tr_cfg.get("device", "auto") == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(tr_cfg.get("device", "cpu"))

    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    train_loader = DataLoader(train_ds, batch_size=int(tr_cfg.get("batch_size", 32)), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(tr_cfg.get("batch_size", 32)), shuffle=False, num_workers=0)

    model_cfg = build_model_cfg(cfg)
    model = MultiScaleSpatialPriorModel(model_cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(tr_cfg.get("lr", 1e-3)),
        weight_decay=float(tr_cfg.get("weight_decay", 0.0)),
    )
    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("config/yaml", yaml.safe_dump(cfg, sort_keys=False))

    best_val = float("inf")
    history = []
    batch_log_rows = []
    epochs = int(tr_cfg.get("epochs", 5))
    log_every = int(tr_cfg.get("log_every_batches", 20))

    for epoch in range(1, epochs + 1):
        model.train()
        agg = {"l_total": 0.0, "l_rec": 0.0, "l_trans_f2": 0.0, "l_trans": 0.0, "l_wd": 0.0, "l_nb": 0.0, "n": 0}
        running = {"l_total": 0.0, "l_rec": 0.0, "l_trans_f2": 0.0, "l_trans": 0.0, "l_wd": 0.0, "l_nb": 0.0, "n": 0}
        total_batches = len(train_loader)
        print(f"epoch {epoch:03d} started | batches={total_batches} | device={device}")
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            outputs = model(batch["i_t"], batch["i_t1"], batch["flow_t"])
            loss, stats = compute_spatial_prior_losses(outputs, batch["i_t"], batch["i_t1"], model_cfg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for key in ["l_total", "l_rec", "l_trans_f2", "l_trans", "l_wd", "l_nb"]:
                agg[key] += stats[key]
                running[key] += stats[key]
            agg["n"] += 1
            running["n"] += 1
            if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches:
                denom = max(running["n"], 1)
                pct = 100.0 * batch_idx / max(total_batches, 1)
                batch_log_rows.append(
                    {
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "total_batches": total_batches,
                        "progress_pct": pct,
                        "recent_total": running["l_total"] / denom,
                        "recent_rec": running["l_rec"] / denom,
                        "recent_trans_f2": running["l_trans_f2"] / denom,
                        "recent_trans": running["l_trans"] / denom,
                        "recent_wd": running["l_wd"] / denom,
                        "recent_nb": running["l_nb"] / denom,
                    }
                )
                print(
                    f"  epoch {epoch:03d} progress {batch_idx}/{total_batches} ({pct:.1f}%) | "
                    f"recent total={running['l_total']/denom:.4f} rec={running['l_rec']/denom:.4f} "
                    f"trans_f2={running['l_trans_f2']/denom:.4f} trans={running['l_trans']/denom:.4f} "
                    f"wd={running['l_wd']/denom:.4f} nb={running['l_nb']/denom:.4f}"
                )
                running = {"l_total": 0.0, "l_rec": 0.0, "l_trans_f2": 0.0, "l_trans": 0.0, "l_wd": 0.0, "l_nb": 0.0, "n": 0}

        for key in ["l_total", "l_rec", "l_trans_f2", "l_trans", "l_wd", "l_nb"]:
            agg[key] /= max(agg["n"], 1)

        val_stats = evaluate(model, val_loader, model_cfg, device)
        row = {
            "epoch": epoch,
            "train_total": agg["l_total"],
            "train_rec": agg["l_rec"],
            "train_trans_f2": agg["l_trans_f2"],
            "train_trans": agg["l_trans"],
            "train_wd": agg["l_wd"],
            "train_nb": agg["l_nb"],
            "val_total": val_stats["l_total"],
            "val_rec": val_stats["l_rec"],
            "val_trans_f2": val_stats["l_trans_f2"],
            "val_trans": val_stats["l_trans"],
            "val_wd": val_stats["l_wd"],
            "val_nb": val_stats["l_nb"],
        }
        history.append(row)

        writer.add_scalar("loss/train_total", agg["l_total"], epoch)
        writer.add_scalar("loss/train_rec", agg["l_rec"], epoch)
        writer.add_scalar("loss/train_trans_f2", agg["l_trans_f2"], epoch)
        writer.add_scalar("loss/train_trans", agg["l_trans"], epoch)
        writer.add_scalar("loss/train_wd", agg["l_wd"], epoch)
        writer.add_scalar("loss/train_nb", agg["l_nb"], epoch)
        writer.add_scalar("loss/val_total", val_stats["l_total"], epoch)
        writer.add_scalar("loss/val_rec", val_stats["l_rec"], epoch)
        writer.add_scalar("loss/val_trans_f2", val_stats["l_trans_f2"], epoch)
        writer.add_scalar("loss/val_trans", val_stats["l_trans"], epoch)
        writer.add_scalar("loss/val_wd", val_stats["l_wd"], epoch)
        writer.add_scalar("loss/val_nb", val_stats["l_nb"], epoch)

        print(
            f"epoch {epoch:03d} done | train_total={agg['l_total']:.4f} val_total={val_stats['l_total']:.4f} "
            f"rec={val_stats['l_rec']:.4f} trans_f2={val_stats['l_trans_f2']:.4f} "
            f"trans={val_stats['l_trans']:.4f} wd={val_stats['l_wd']:.4f} nb={val_stats['l_nb']:.4f}"
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "decoder_state_dict": model.decoder.state_dict(),
            "epoch": epoch,
            "history": history,
            "config": cfg,
            "spatial_prior_config": asdict(model_cfg),
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_stats["l_total"] < best_val:
            best_val = val_stats["l_total"]
            torch.save(ckpt, out_dir / "best.pt")
        save_history(history, out_dir)
        save_batch_log(batch_log_rows, out_dir)
        save_curve_svg(history, out_dir / "loss_curve.svg")

    writer.flush()
    writer.close()
    print("Best validation loss:", best_val)
    print("Saved to", out_dir)


if __name__ == "__main__":
    main()
