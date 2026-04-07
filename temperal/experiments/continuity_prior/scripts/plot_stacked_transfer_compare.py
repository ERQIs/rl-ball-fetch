from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot downstream transfer comparison curves.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("d:/projects/rl-ball-fetch"),
    )
    parser.add_argument(
        "--stacked-root",
        type=Path,
        default=Path(
            "d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/stacked_transfer_888_20260325"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/stacked_transfer_888_20260325/stacked_transfer_vs_baselines_curves.png"
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def series_from_rows(rows: list[dict], key: str, max_epochs: int) -> np.ndarray:
    arr = np.full(max_epochs, np.nan, dtype=np.float64)
    for row in rows:
        epoch = int(row["epoch"])
        if 1 <= epoch <= max_epochs:
            arr[epoch - 1] = float(row[key])
    return arr


def aggregate_histories(run_dirs: list[Path], max_epochs: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    train_loss = []
    val_loss = []
    val_success = []
    for run_dir in run_dirs:
        rows = load_csv_rows(run_dir / "history.csv")
        train_loss.append(series_from_rows(rows, "train_loss", max_epochs))
        val_loss.append(series_from_rows(rows, "val_loss", max_epochs))
        val_success.append(series_from_rows(rows, "val_success_at_0.2", max_epochs))

    def mean_std(stack: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        arr = np.vstack(stack)
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

    return {
        "train_loss": mean_std(train_loss),
        "val_loss": mean_std(val_loss),
        "val_success": mean_std(val_success),
    }


def load_aggregate_history(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows = load_csv_rows(path)

    def col(name: str) -> np.ndarray:
        return np.array([float(row[name]) for row in rows], dtype=np.float64)

    return {
        "train_loss": (col("train_loss_mean"), col("train_loss_std")),
        "val_loss": (col("val_loss_mean"), col("val_loss_std")),
        "val_success": (col("val_success_mean"), col("val_success_std")),
    }


def draw_band(ax, epochs: np.ndarray, mean: np.ndarray, std: np.ndarray, color: str, label: str, alpha: float = 0.18):
    ax.plot(epochs, mean, color=color, linewidth=2.0, label=label)
    ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=alpha, linewidth=0)


def find_stacked_run_dirs(stacked_root: Path, fraction_pct: int) -> list[Path]:
    pattern = f"multiscale_transfer_stacked888_finetune_{fraction_pct}pct_obs50_seed*_20260325"
    return sorted([p for p in stacked_root.glob(pattern) if p.is_dir()])


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root
    temperal_outputs = repo_root / "temperal" / "outputs"

    methods = {
        "scratch": {"label": "Scratch", "color": "#1f77b4"},
        "future": {"label": "Future Finetune", "color": "#d62728"},
        "stacked": {"label": "Stacked888 Finetune", "color": "#059669"},
    }

    fractions = [20, 40, 100]
    epochs = np.arange(1, args.epochs + 1)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True, dpi=180)

    for col_idx, fraction in enumerate(fractions):
        scratch_hist = load_aggregate_history(
            temperal_outputs / f"multiscale_transfer_scratch_{fraction}pct_obs50_3seed_20260316_220723" / "aggregate_history.csv"
        )
        future_hist = load_aggregate_history(
            temperal_outputs / f"multiscale_transfer_finetune_{fraction}pct_obs50_3seed_20260316_220723" / "aggregate_history.csv"
        )
        stacked_dirs = find_stacked_run_dirs(args.stacked_root, fraction)
        if not stacked_dirs:
            raise SystemExit(f"No stacked run dirs found for {fraction}% under {args.stacked_root}")
        stacked_hist = aggregate_histories(stacked_dirs, args.epochs)

        histories = {
            "scratch": scratch_hist,
            "future": future_hist,
            "stacked": stacked_hist,
        }

        loss_ax = axes[0, col_idx]
        success_ax = axes[1, col_idx]

        for key, meta in methods.items():
            val_loss_mean, val_loss_std = histories[key]["val_loss"]
            success_mean, success_std = histories[key]["val_success"]
            draw_band(loss_ax, epochs, val_loss_mean, val_loss_std, meta["color"], meta["label"])
            draw_band(success_ax, epochs, success_mean, success_std, meta["color"], meta["label"])

        loss_ax.set_title(f"{fraction}% Data: Val Loss")
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Loss")
        loss_ax.grid(alpha=0.25)

        success_ax.set_title(f"{fraction}% Data: Val Success@0.2")
        success_ax.set_xlabel("Epoch")
        success_ax.set_ylabel("Success Rate")
        success_ax.set_ylim(0.0, 1.0)
        success_ax.grid(alpha=0.25)

    axes[0, 2].legend(frameon=False, fontsize=9, loc="upper right")
    axes[1, 2].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Downstream Comparison: Scratch vs Future vs Stacked888", fontsize=14)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
