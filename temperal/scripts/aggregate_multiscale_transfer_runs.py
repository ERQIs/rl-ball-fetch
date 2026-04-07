import argparse
import csv
import json
import math
from pathlib import Path


def read_history(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for row in rows:
        parsed.append(
            {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]) if row["val_loss"] else None,
                "val_mean_l2": float(row["val_mean_l2"]) if row["val_mean_l2"] else None,
                "val_median_l2": float(row["val_median_l2"]) if row["val_median_l2"] else None,
                "val_success_at_0.2": float(row["val_success_at_0.2"]) if row["val_success_at_0.2"] else None,
            }
        )
    return parsed


def read_test_metrics(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    return sum(values) / len(values)


def std(values):
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5


def save_csv(rows, path, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_curve_svg(history, path, title, y_label, key_mean, key_std, color):
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
    vals = [row[key_mean] for row in history]
    errs = [row[key_std] for row in history]
    y_min = min(v - e for v, e in zip(vals, errs))
    y_max = max(v + e for v, e in zip(vals, errs))
    if abs(y_max - y_min) < 1e-8:
        y_max = y_min + 1.0

    def x_pos(epoch):
        if len(epochs) == 1:
            return margin_left + plot_w / 2.0
        return margin_left + (epoch - epochs[0]) / (epochs[-1] - epochs[0]) * plot_w

    def y_pos(value):
        return margin_top + (y_max - value) / (y_max - y_min) * plot_h

    mean_points = " ".join(f"{x_pos(row['epoch']):.2f},{y_pos(row[key_mean]):.2f}" for row in history)
    upper = [(x_pos(row["epoch"]), y_pos(row[key_mean] + row[key_std])) for row in history]
    lower = [(x_pos(row["epoch"]), y_pos(row[key_mean] - row[key_std])) for row in reversed(history)]
    band_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in upper + lower)

    x_ticks = sorted(set([epochs[0], epochs[-1], max(1, epochs[-1] // 2)]))
    y_ticks = [y_min + i * (y_max - y_min) / 4.0 for i in range(5)]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<style>text{{font-family:Arial,sans-serif;font-size:12px;fill:#222}} .grid{{stroke:#ddd;stroke-width:1}} .axis{{stroke:#333;stroke-width:1.5}} .band{{fill:{color};fill-opacity:0.18;stroke:none}} .line{{fill:none;stroke:{color};stroke-width:2.5}} .title{{font-size:18px;font-weight:bold}}</style>',
        f'<text x="{width/2:.0f}" y="20" text-anchor="middle" class="title">{title}</text>',
    ]
    for tick in y_ticks:
        y = y_pos(tick)
        lines.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width-margin_right}" y2="{y:.2f}" class="grid" />')
        lines.append(f'<text x="{margin_left-10}" y="{y+4:.2f}" text-anchor="end">{tick:.4f}</text>')
    for tick in x_ticks:
        x = x_pos(tick)
        lines.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height-margin_bottom}" class="grid" />')
        lines.append(f'<text x="{x:.2f}" y="{height-margin_bottom+20}" text-anchor="middle">{tick}</text>')
    lines.extend(
        [
            f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" class="axis" />',
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" class="axis" />',
            f'<text x="{width/2:.0f}" y="{height-15}" text-anchor="middle">Epoch</text>',
            f'<text x="18" y="{height/2:.0f}" text-anchor="middle" transform="rotate(-90 18 {height/2:.0f})">{y_label}</text>',
            f'<polygon points="{band_points}" class="band" />',
            f'<polyline points="{mean_points}" class="line" />',
            f'<text x="{width-margin_right-165}" y="{margin_top+18}">mean ± std across seeds</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    histories = []
    per_seed_rows = []

    for run_dir in args.run_dir:
        run_path = Path(run_dir)
        history = read_history(run_path / "history.csv")
        test_metrics = read_test_metrics(run_path / "test_metrics.json")
        histories.append(history)
        best_row = min(history, key=lambda row: row["val_loss"] if row["val_loss"] is not None else math.inf)
        per_seed_rows.append(
            {
                "run_dir": str(run_path),
                "best_epoch": best_row["epoch"],
                "train_loss": best_row["train_loss"],
                "val_loss": best_row["val_loss"],
                "val_mean_l2": best_row["val_mean_l2"],
                "val_median_l2": best_row["val_median_l2"],
                "val_success_at_0.2": best_row["val_success_at_0.2"],
                "test_loss": test_metrics["loss"],
                "test_mean_l2": test_metrics["mean_l2"],
                "test_median_l2": test_metrics["median_l2"],
                "test_success_at_0.2": test_metrics["success_at_0.2"],
            }
        )

    epoch_count = min(len(history) for history in histories)
    aggregate_history = []
    for idx in range(epoch_count):
        rows = [history[idx] for history in histories]
        aggregate_history.append(
            {
                "epoch": rows[0]["epoch"],
                "train_loss_mean": mean([row["train_loss"] for row in rows]),
                "train_loss_std": std([row["train_loss"] for row in rows]),
                "val_loss_mean": mean([row["val_loss"] for row in rows if row["val_loss"] is not None]),
                "val_loss_std": std([row["val_loss"] for row in rows if row["val_loss"] is not None]),
                "val_mean_l2_mean": mean([row["val_mean_l2"] for row in rows if row["val_mean_l2"] is not None]),
                "val_mean_l2_std": std([row["val_mean_l2"] for row in rows if row["val_mean_l2"] is not None]),
                "val_success_mean": mean([row["val_success_at_0.2"] for row in rows if row["val_success_at_0.2"] is not None]),
                "val_success_std": std([row["val_success_at_0.2"] for row in rows if row["val_success_at_0.2"] is not None]),
            }
        )

    aggregate_summary = [
        {
            "num_seeds": len(per_seed_rows),
            "val_mean_l2_mean": mean([row["val_mean_l2"] for row in per_seed_rows]),
            "val_mean_l2_std": std([row["val_mean_l2"] for row in per_seed_rows]),
            "val_success_mean": mean([row["val_success_at_0.2"] for row in per_seed_rows]),
            "val_success_std": std([row["val_success_at_0.2"] for row in per_seed_rows]),
            "test_mean_l2_mean": mean([row["test_mean_l2"] for row in per_seed_rows]),
            "test_mean_l2_std": std([row["test_mean_l2"] for row in per_seed_rows]),
            "test_success_mean": mean([row["test_success_at_0.2"] for row in per_seed_rows]),
            "test_success_std": std([row["test_success_at_0.2"] for row in per_seed_rows]),
        }
    ]

    save_csv(
        per_seed_rows,
        out_dir / "per_seed_summary.csv",
        [
            "run_dir",
            "best_epoch",
            "train_loss",
            "val_loss",
            "val_mean_l2",
            "val_median_l2",
            "val_success_at_0.2",
            "test_loss",
            "test_mean_l2",
            "test_median_l2",
            "test_success_at_0.2",
        ],
    )
    save_csv(
        aggregate_summary,
        out_dir / "aggregate_summary.csv",
        [
            "num_seeds",
            "val_mean_l2_mean",
            "val_mean_l2_std",
            "val_success_mean",
            "val_success_std",
            "test_mean_l2_mean",
            "test_mean_l2_std",
            "test_success_mean",
            "test_success_std",
        ],
    )
    save_csv(
        aggregate_history,
        out_dir / "aggregate_history.csv",
        [
            "epoch",
            "train_loss_mean",
            "train_loss_std",
            "val_loss_mean",
            "val_loss_std",
            "val_mean_l2_mean",
            "val_mean_l2_std",
            "val_success_mean",
            "val_success_std",
        ],
    )

    save_curve_svg(aggregate_history, out_dir / "aggregate_loss_curve.svg", "Average Loss Curve", "Loss", "val_loss_mean", "val_loss_std", "#dc2626")
    save_curve_svg(aggregate_history, out_dir / "aggregate_success_curve.svg", "Average Success Curve @ 0.2", "Success Rate", "val_success_mean", "val_success_std", "#059669")


if __name__ == "__main__":
    main()
