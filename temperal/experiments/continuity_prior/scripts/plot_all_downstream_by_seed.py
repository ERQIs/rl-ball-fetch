from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-seed downstream curves for all main comparison methods."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("d:/projects/rl-ball-fetch"),
    )
    parser.add_argument(
        "--future-warmup-root",
        type=Path,
        default=Path(
            "d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/future_transfer_warmup5_20260326"
        ),
    )
    parser.add_argument(
        "--stacked-root",
        type=Path,
        default=Path(
            "d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/stacked_transfer_888_20260325"
        ),
    )
    parser.add_argument(
        "--stacked-warmup-root",
        type=Path,
        default=Path(
            "d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/stacked_transfer_888_warmup5_20260326"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/downstream_all_methods_by_seed_20260326"
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


def load_history(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def extract_seed(name: str) -> int:
    match = re.search(r"seed(\d+)", name)
    if not match:
        raise ValueError(f"Could not parse seed from run name: {name}")
    return int(match.group(1))


def build_index(root: Path, pattern: str, mode: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for run_dir in root.glob(pattern):
        if not run_dir.is_dir():
            continue
        history_path = run_dir / "history.csv"
        if not history_path.exists():
            continue
        seed = extract_seed(run_dir.name)
        result[seed] = history_path
    if not result:
        raise FileNotFoundError(f"No matching runs found for mode={mode} under {root} with pattern={pattern}")
    return dict(sorted(result.items()))


def epochs_and_metric(rows: list[dict], key: str, max_epochs: int) -> tuple[list[int], list[float]]:
    xs = []
    ys = []
    for row in rows:
        epoch = int(row["epoch"])
        if epoch <= max_epochs:
            xs.append(epoch)
            ys.append(float(row[key]))
    return xs, ys


def gather_paths(args: argparse.Namespace, fraction_pct: int) -> dict[str, dict[int, Path]]:
    temperal_outputs = args.repo_root / "temperal" / "outputs"
    return {
        "scratch": build_index(
            temperal_outputs,
            f"multiscale_transfer_scratch_{fraction_pct}pct_obs50_seed*_20260316_220723",
            "scratch",
        ),
        "future_finetune": build_index(
            temperal_outputs,
            f"multiscale_transfer_finetune_{fraction_pct}pct_obs50_seed*_20260316_220723",
            "future_finetune",
        ),
        "future_warmup5": build_index(
            args.future_warmup_root,
            f"multiscale_transfer_future_warmup5_finetune_{fraction_pct}pct_obs50_seed*_20260326",
            "future_warmup5",
        ),
        "stacked888_finetune": build_index(
            args.stacked_root,
            f"multiscale_transfer_stacked888_finetune_{fraction_pct}pct_obs50_seed*_20260325",
            "stacked888_finetune",
        ),
        "stacked888_warmup5": build_index(
            args.stacked_warmup_root,
            f"multiscale_transfer_stacked888_warmup5_finetune_{fraction_pct}pct_obs50_seed*_20260326",
            "stacked888_warmup5",
        ),
    }


def plot_fraction(args: argparse.Namespace, fraction_pct: int) -> Path:
    mode_paths = gather_paths(args, fraction_pct)
    seeds = sorted(
        set.intersection(*(set(seed_map.keys()) for seed_map in mode_paths.values()))
    )
    if not seeds:
        raise RuntimeError(f"No common seeds found for {fraction_pct}%")

    colors = {
        "scratch": "#1f77b4",
        "future_finetune": "#d62728",
        "future_warmup5": "#ff7f0e",
        "stacked888_finetune": "#059669",
        "stacked888_warmup5": "#7c3aed",
    }
    labels = {
        "scratch": "Scratch",
        "future_finetune": "Future",
        "future_warmup5": "Future Warmup5",
        "stacked888_finetune": "Stacked888",
        "stacked888_warmup5": "Stacked888 Warmup5",
    }
    plot_order = [
        "scratch",
        "future_finetune",
        "future_warmup5",
        "stacked888_finetune",
        "stacked888_warmup5",
    ]

    fig, axes = plt.subplots(2, len(seeds), figsize=(5.4 * len(seeds), 7.4), dpi=180, constrained_layout=True)
    if len(seeds) == 1:
        axes = axes.reshape(2, 1)

    for col, seed in enumerate(seeds):
        ax_loss = axes[0, col]
        ax_success = axes[1, col]

        for mode in plot_order:
            rows = load_history(mode_paths[mode][seed])
            epochs, val_loss = epochs_and_metric(rows, "val_loss", args.epochs)
            _, val_success = epochs_and_metric(rows, "val_success_at_0.2", args.epochs)
            ax_loss.plot(epochs, val_loss, color=colors[mode], linewidth=2.0, label=labels[mode])
            ax_success.plot(epochs, val_success, color=colors[mode], linewidth=2.0, label=labels[mode])

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
        f"Downstream All-Method Comparison by Seed ({fraction_pct}% data, {args.epochs} epochs)",
        fontsize=14,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"all_methods_by_seed_{fraction_pct}pct.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    outputs = [plot_fraction(args, fraction_pct) for fraction_pct in (20, 40, 100)]
    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
