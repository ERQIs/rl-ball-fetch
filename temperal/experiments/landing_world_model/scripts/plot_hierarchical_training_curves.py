from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Hierarchical Landing Training Curves")
    args = parser.parse_args()

    history_path = Path(args.history)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(history_path)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    epochs = df["epoch"].to_numpy()

    # Loss
    axes[0].plot(epochs, df["train_loss"], label="train_loss", color="#2563eb", linewidth=2)
    axes[0].plot(epochs, df["val_loss"], label="val_loss", color="#dc2626", linewidth=2)
    best_idx = int(df["val_mean_l2"].idxmin())
    best_epoch = int(df.loc[best_idx, "epoch"])
    axes[0].axvline(best_epoch, color="#6b7280", linestyle="--", linewidth=1.5, label=f"best epoch {best_epoch}")
    axes[0].set_ylabel("MSE Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Mean L2
    axes[1].plot(epochs, df["train_mean_l2"], label="train_mean_l2", color="#0891b2", linewidth=2)
    axes[1].plot(epochs, df["val_mean_l2"], label="val_mean_l2", color="#ea580c", linewidth=2)
    axes[1].axvline(best_epoch, color="#6b7280", linestyle="--", linewidth=1.5)
    axes[1].scatter([best_epoch], [df.loc[best_idx, "val_mean_l2"]], color="#ea580c", s=40, zorder=3)
    axes[1].set_ylabel("Mean L2")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    # Success
    axes[2].plot(epochs, df["train_success"], label="train_success@0.2", color="#16a34a", linewidth=2)
    axes[2].plot(epochs, df["val_success"], label="val_success@0.2", color="#9333ea", linewidth=2)
    axes[2].axvline(best_epoch, color="#6b7280", linestyle="--", linewidth=1.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Success@0.2")
    axes[2].set_ylim(bottom=0.0)
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.suptitle(args.title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
