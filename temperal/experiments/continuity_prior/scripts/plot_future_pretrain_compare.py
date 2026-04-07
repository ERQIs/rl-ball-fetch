import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_history(path):
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        raise ValueError(f"Missing epoch column in {path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-history", required=True)
    parser.add_argument("--spatial-history", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    scratch = load_history(args.scratch_history)
    spatial = load_history(args.spatial_history)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    panels = [
        ("train_loss", "Train Total Loss"),
        ("val_loss", "Val Total Loss"),
        ("val_pos", "Val Position Loss"),
        ("val_vel", "Val Velocity Loss"),
    ]

    for ax, (key, title) in zip(axes.flat, panels):
        ax.plot(scratch["epoch"], scratch[key], label="Scratch Future Pretrain", linewidth=2)
        ax.plot(spatial["epoch"], spatial[key], label="Spatial-init Future Pretrain", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)

    axes[0, 1].legend(frameon=False, loc="upper right")
    fig.suptitle("Temporal Pretrain: Scratch vs Spatial-init", fontsize=14)
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
