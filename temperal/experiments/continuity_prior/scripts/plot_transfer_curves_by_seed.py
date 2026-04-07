from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot downstream transfer curves by seed without aggregation bands."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--warmup-root", type=Path, default=None)
    parser.add_argument("--warmup10-root", type=Path, default=None)
    parser.add_argument("--train-fraction-pct", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def extract_seed(name: str) -> int:
    match = re.search(r"seed(\d+)", name)
    if not match:
        raise ValueError(f"Could not parse seed from run name: {name}")
    return int(match.group(1))


def load_history(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_index(output_root: Path, train_fraction_pct: int) -> dict[int, dict[str, Path]]:
    result: dict[int, dict[str, Path]] = {}
    for run_dir in output_root.glob(f"*_{train_fraction_pct}pct_*"):
        history_path = run_dir / "history.csv"
        if not history_path.exists():
            continue
        name = run_dir.name
        if "_scratch_" in name:
            mode = "scratch"
        elif "_spatial_finetune_" in name:
            mode = "spatial_finetune"
        else:
            continue
        seed = extract_seed(name)
        result.setdefault(seed, {})[mode] = history_path
    return dict(sorted(result.items()))


def main() -> None:
    args = parse_args()
    histories = build_index(args.output_root, args.train_fraction_pct)
    if args.warmup_root is not None:
        warmup_histories = build_index(args.warmup_root, args.train_fraction_pct)
        for seed, mode_paths in warmup_histories.items():
            if "spatial_finetune" in mode_paths:
                histories.setdefault(seed, {})["spatial_finetune_warmup5"] = mode_paths["spatial_finetune"]
    if args.warmup10_root is not None:
        warmup10_histories = build_index(args.warmup10_root, args.train_fraction_pct)
        for seed, mode_paths in warmup10_histories.items():
            if "spatial_finetune" in mode_paths:
                histories.setdefault(seed, {})["spatial_finetune_warmup10"] = mode_paths["spatial_finetune"]
    if not histories:
        raise SystemExit(f"No matching history.csv found under {args.output_root}")

    fig, axes = plt.subplots(2, len(histories), figsize=(5.2 * len(histories), 7.2), constrained_layout=True)
    if len(histories) == 1:
        axes = axes.reshape(2, 1)

    colors = {
        "scratch": "#1f77b4",
        "spatial_finetune": "#d62728",
        "spatial_finetune_warmup5": "#2ca02c",
        "spatial_finetune_warmup10": "#9467bd",
    }
    labels = {
        "scratch": "Scratch",
        "spatial_finetune": "Spatial Finetune",
        "spatial_finetune_warmup5": "Spatial Warmup5",
        "spatial_finetune_warmup10": "Spatial Warmup10",
    }

    for col, (seed, mode_paths) in enumerate(histories.items()):
        ax_loss = axes[0, col]
        ax_success = axes[1, col]

        for mode in ["scratch", "spatial_finetune", "spatial_finetune_warmup5", "spatial_finetune_warmup10"]:
            if mode not in mode_paths:
                continue
            rows = load_history(mode_paths[mode])
            epochs = [int(r["epoch"]) for r in rows if int(r["epoch"]) <= args.epochs]
            train_loss = [float(r["train_loss"]) for r in rows if int(r["epoch"]) <= args.epochs]
            val_loss = [float(r["val_loss"]) for r in rows if int(r["epoch"]) <= args.epochs]
            val_success = [float(r["val_success_at_0.2"]) for r in rows if int(r["epoch"]) <= args.epochs]

            ax_loss.plot(
                epochs,
                train_loss,
                color=colors[mode],
                linewidth=1.8,
                alpha=0.45,
                linestyle="--",
                label=f"{labels[mode]} train",
            )
            ax_loss.plot(
                epochs,
                val_loss,
                color=colors[mode],
                linewidth=2.0,
                label=f"{labels[mode]} val",
            )
            ax_success.plot(
                epochs,
                val_success,
                color=colors[mode],
                linewidth=2.0,
                label=labels[mode],
            )

        ax_loss.set_title(f"Seed {seed} Loss")
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
        f"Spatial Prior Transfer by Seed ({args.train_fraction_pct}% data, {args.epochs} epochs)",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
