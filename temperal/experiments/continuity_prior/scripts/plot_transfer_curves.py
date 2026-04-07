from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot aggregated downstream transfer curves from history.csv files."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-fraction-pct", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def discover_histories(output_root: Path, train_fraction_pct: int) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = defaultdict(list)
    pattern = f"*_{train_fraction_pct}pct_*"
    for run_dir in output_root.glob(pattern):
        history_path = run_dir / "history.csv"
        if not history_path.exists():
            continue
        name = run_dir.name
        if "_scratch_" in name:
            grouped["scratch"].append(history_path)
        elif "_spatial_finetune_" in name:
            grouped["spatial_finetune"].append(history_path)
    return grouped


def load_history(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_epoch_series(rows: List[dict], key: str, max_epochs: int) -> np.ndarray:
    series = np.full(max_epochs, np.nan, dtype=np.float64)
    for row in rows:
        epoch = int(row["epoch"])
        if 1 <= epoch <= max_epochs:
            series[epoch - 1] = float(row[key])
    return series


def aggregate_mode(history_paths: List[Path], max_epochs: int) -> dict:
    train_loss = []
    val_loss = []
    val_success = []
    val_mean_l2 = []
    for path in sorted(history_paths):
        rows = load_history(path)
        train_loss.append(to_epoch_series(rows, "train_loss", max_epochs))
        val_loss.append(to_epoch_series(rows, "val_loss", max_epochs))
        val_success.append(to_epoch_series(rows, "val_success_at_0.2", max_epochs))
        val_mean_l2.append(to_epoch_series(rows, "val_mean_l2", max_epochs))
    return {
        "train_loss": np.vstack(train_loss),
        "val_loss": np.vstack(val_loss),
        "val_success_at_0.2": np.vstack(val_success),
        "val_mean_l2": np.vstack(val_mean_l2),
    }


def mean_std(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.nanmean(array, axis=0), np.nanstd(array, axis=0)


def draw_band(ax, epochs: np.ndarray, mean: np.ndarray, std: np.ndarray, color: str, label: str, alpha: float = 0.18):
    ax.plot(epochs, mean, color=color, linewidth=2.0, label=label)
    ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=alpha, linewidth=0)


def main() -> None:
    args = parse_args()
    grouped = discover_histories(args.output_root, args.train_fraction_pct)
    if not grouped:
        raise SystemExit(f"No matching history.csv found under {args.output_root}")

    modes = {
        "scratch": {"label": "Scratch", "color": "#1f77b4"},
        "spatial_finetune": {"label": "Spatial Finetune", "color": "#d62728"},
    }

    aggregates = {
        mode: aggregate_mode(paths, args.epochs)
        for mode, paths in grouped.items()
        if paths
    }

    epochs = np.arange(1, args.epochs + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    for mode, meta in modes.items():
        if mode not in aggregates:
            continue
        agg = aggregates[mode]
        train_mean, train_std = mean_std(agg["train_loss"])
        val_mean, val_std = mean_std(agg["val_loss"])
        success_mean, success_std = mean_std(agg["val_success_at_0.2"])
        l2_mean, l2_std = mean_std(agg["val_mean_l2"])

        draw_band(
            axes[0],
            epochs,
            train_mean,
            train_std,
            meta["color"],
            f"{meta['label']} train",
            alpha=0.10,
        )
        draw_band(
            axes[0],
            epochs,
            val_mean,
            val_std,
            meta["color"],
            f"{meta['label']} val",
            alpha=0.22,
        )
        draw_band(
            axes[1],
            epochs,
            success_mean,
            success_std,
            meta["color"],
            meta["label"],
        )
        draw_band(
            axes[2],
            epochs,
            l2_mean,
            l2_std,
            meta["color"],
            meta["label"],
        )

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].set_title("Validation Success@0.2")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=9)

    axes[2].set_title("Validation Mean L2")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Mean L2")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=9)

    fig.suptitle(
        f"Spatial Prior Transfer Curves ({args.train_fraction_pct}% data, {args.epochs} epochs)",
        fontsize=13,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
