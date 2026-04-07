from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot default stacked888 vs smaller-backbone-lr stacked888 by seed."
    )
    parser.add_argument(
        "--default-root",
        type=Path,
        default=Path(
            "d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/stacked_transfer_888_20260325"
        ),
    )
    parser.add_argument(
        "--blr-root",
        type=Path,
        default=Path(
            "d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/stacked_transfer_888_blr01_20260326"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/stacked_transfer_888_blr01_20260326"
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


def extract_seed(name: str) -> int:
    match = re.search(r"seed(\d+)", name)
    if not match:
        raise ValueError(f"Could not parse seed from run name: {name}")
    return int(match.group(1))


def load_history(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_index(root: Path, fraction_pct: int, with_blr_suffix: bool) -> dict[int, Path]:
    if with_blr_suffix:
        pattern = f"multiscale_transfer_stacked888_finetune_{fraction_pct}pct_obs50_seed*_blr01"
    else:
        pattern = f"multiscale_transfer_stacked888_finetune_{fraction_pct}pct_obs50_seed*"
    result: dict[int, Path] = {}
    for run_dir in root.glob(pattern):
        if not run_dir.is_dir():
            continue
        history_path = run_dir / "history.csv"
        if not history_path.exists():
            continue
        result[extract_seed(run_dir.name)] = history_path
    return dict(sorted(result.items()))


def epoch_series(rows: list[dict], key: str, max_epochs: int) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for row in rows:
        epoch = int(row["epoch"])
        if epoch <= max_epochs:
            xs.append(epoch)
            ys.append(float(row[key]))
    return xs, ys


def plot_fraction(args: argparse.Namespace, fraction_pct: int) -> Path:
    default_histories = build_index(args.default_root, fraction_pct, with_blr_suffix=False)
    blr_histories = build_index(args.blr_root, fraction_pct, with_blr_suffix=True)
    seeds = sorted(set(default_histories.keys()) & set(blr_histories.keys()))
    if not seeds:
        raise RuntimeError(f"No overlapping seeds found for {fraction_pct}%")

    fig, axes = plt.subplots(2, len(seeds), figsize=(5.4 * len(seeds), 7.2), dpi=180, constrained_layout=True)
    if len(seeds) == 1:
        axes = axes.reshape(2, 1)

    styles = {
        "default": {"color": "#059669", "label": "Stacked888 Default"},
        "blr01": {"color": "#7c3aed", "label": "Stacked888 Backbone LR 0.1x"},
    }

    for col, seed in enumerate(seeds):
        ax_loss = axes[0, col]
        ax_success = axes[1, col]

        for mode, history_path in [("default", default_histories[seed]), ("blr01", blr_histories[seed])]:
            rows = load_history(history_path)
            epochs, val_loss = epoch_series(rows, "val_loss", args.epochs)
            _, val_success = epoch_series(rows, "val_success_at_0.2", args.epochs)
            ax_loss.plot(epochs, val_loss, color=styles[mode]["color"], linewidth=2.2, label=styles[mode]["label"])
            ax_success.plot(epochs, val_success, color=styles[mode]["color"], linewidth=2.2, label=styles[mode]["label"])

        ax_loss.set_title(f"Seed {seed} Val Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(alpha=0.25)
        ax_loss.legend(frameon=False, fontsize=8)

        ax_success.set_title(f"Seed {seed} Val Success@0.2")
        ax_success.set_xlabel("Epoch")
        ax_success.set_ylabel("Success Rate")
        ax_success.set_ylim(0.0, 1.0)
        ax_success.grid(alpha=0.25)
        ax_success.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"Stacked888 Default vs Smaller Backbone LR by Seed ({fraction_pct}% data)",
        fontsize=14,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"stacked888_blr01_compare_by_seed_{fraction_pct}pct.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    for fraction_pct in (20, 100):
        path = plot_fraction(args, fraction_pct)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
