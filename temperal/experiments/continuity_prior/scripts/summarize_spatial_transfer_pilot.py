import csv
import json
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_run_name(name: str):
    parts = name.split("_")
    if "scratch" in name:
        mode = "scratch"
    elif "spatial_finetune" in name:
        mode = "spatial_finetune"
    elif "adapter_finetune" in name:
        mode = "adapter_finetune"
    elif "stacked888_finetune" in name:
        mode = "stacked888_finetune"
    elif "future_warmup5_finetune" in name:
        mode = "future_warmup5_finetune"
    elif "stacked888_warmup5_finetune" in name:
        mode = "stacked888_warmup5_finetune"
    else:
        mode = "unknown"

    fraction = None
    seed = None
    for part in parts:
        if part.endswith("pct") and part[:-3].isdigit():
            fraction = int(part[:-3])
        if part.startswith("seed") and part[4:].isdigit():
            seed = int(part[4:])
    return mode, fraction, seed


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_pilot_20260323",
    )
    args = parser.parse_args()

    root = repo_root()
    output_root = Path(args.output_root)

    temperal_root = root / "temperal"
    if str(temperal_root) not in sys.path:
        sys.path.insert(0, str(temperal_root))

    from scripts.train_multiscale_downstream import evaluate_checkpoint  # pylint: disable=import-error

    run_dirs = sorted([p for p in output_root.iterdir() if p.is_dir() and p.name.startswith("multiscale_transfer_")])
    detailed_rows = []
    grouped = defaultdict(list)

    for run_dir in run_dirs:
        cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        best_ckpt = run_dir / "best.pt"
        test_metrics = evaluate_checkpoint(cfg, best_ckpt, "test")
        save_json(test_metrics, run_dir / "test_metrics.json")

        best_val = load_json(run_dir / "best_val_metrics.json")
        mode, fraction, seed = parse_run_name(run_dir.name)

        row = {
            "run_name": run_dir.name,
            "mode": mode,
            "train_fraction_pct": fraction,
            "seed": seed,
            "best_val_loss": float(best_val["loss"]),
            "best_val_mean_l2": float(best_val["mean_l2"]),
            "best_val_success_at_0.2": float(best_val["success_at_0.2"]),
            "test_loss": float(test_metrics["loss"]),
            "test_mean_l2": float(test_metrics["mean_l2"]),
            "test_success_at_0.2": float(test_metrics["success_at_0.2"]),
            "output_dir": str(run_dir).replace("\\", "/"),
        }
        detailed_rows.append(row)
        grouped[(mode, fraction)].append(row)

    summary_rows = []
    for (mode, fraction), rows in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        test_mean_l2 = np.array([r["test_mean_l2"] for r in rows], dtype=float)
        test_success = np.array([r["test_success_at_0.2"] for r in rows], dtype=float)
        val_mean_l2 = np.array([r["best_val_mean_l2"] for r in rows], dtype=float)
        val_success = np.array([r["best_val_success_at_0.2"] for r in rows], dtype=float)

        summary_rows.append(
            {
                "mode": mode,
                "train_fraction_pct": fraction,
                "num_seeds": len(rows),
                "val_mean_l2_mean": float(val_mean_l2.mean()),
                "val_mean_l2_std": float(val_mean_l2.std(ddof=0)),
                "val_success_at_0.2_mean": float(val_success.mean()),
                "val_success_at_0.2_std": float(val_success.std(ddof=0)),
                "test_mean_l2_mean": float(test_mean_l2.mean()),
                "test_mean_l2_std": float(test_mean_l2.std(ddof=0)),
                "test_success_at_0.2_mean": float(test_success.mean()),
                "test_success_at_0.2_std": float(test_success.std(ddof=0)),
            }
        )

    detailed_path = output_root / "detailed_results.csv"
    with detailed_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detailed_rows)

    summary_path = output_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Saved", detailed_path)
    print("Saved", summary_path)


if __name__ == "__main__":
    main()
