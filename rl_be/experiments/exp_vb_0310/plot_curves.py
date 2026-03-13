from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "figures"

COLORS = {
    "E1": (31, 119, 180),
    "E2": (255, 127, 14),
    "E3": (44, 160, 44),
    "E4": (214, 39, 40),
}
LABELS = {
    "E1": "E1 Global Scratch",
    "E2": "E2 Spatial Scratch",
    "E3": "E3 Spatial Pretrain Frozen",
    "E4": "E4 Spatial Pretrain Finetune",
    # "E1": "E1",
    # "E2": "E2",
    # "E3": "E3",
    # "E4": "E4",
}


def _load_font(size: int, font_path: str = "") -> ImageFont.ImageFont:
    if font_path:
        return ImageFont.truetype(font_path, size=size)
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _read_csv_series(path: Path) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append((float(row["Step"]), float(row["Value"])))
    return points


def _moving_average(points: List[Tuple[float, float]], window: int) -> List[Tuple[float, float]]:
    if not points:
        return []
    out: List[Tuple[float, float]] = []
    values: List[float] = []
    for step, value in points:
        values.append(value)
        lo = max(0, len(values) - window)
        avg = sum(values[lo:]) / (len(values) - lo)
        out.append((step, avg))
    return out


def _collect(metric_dir: Path, smooth_window: int) -> Dict[str, List[Tuple[float, float]]]:
    series: Dict[str, List[Tuple[float, float]]] = {}
    for path in sorted(metric_dir.glob("*.csv")):
        exp = None
        for key in ("E1", "E2", "E3", "E4"):
            if key in path.name:
                exp = key
                break
        if exp is None:
            continue
        series[exp] = _moving_average(_read_csv_series(path), window=smooth_window)
    return series


def _format_k(x: float) -> str:
    return f"{int(round(x / 1000.0))}k"


def _parse_tick_list(raw: str) -> List[float]:
    if not raw.strip():
        return []
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _piecewise_project(y: float, ticks: List[float], top: int, plot_h: int) -> int:
    """
    Map numeric y to image y using equal visual spacing between adjacent ticks,
    regardless of the numeric distance between those ticks.
    """
    if len(ticks) < 2:
        raise ValueError("Need at least 2 ticks for piecewise projection.")

    ticks_sorted = sorted(ticks)
    y = min(max(y, ticks_sorted[0]), ticks_sorted[-1])
    segments = len(ticks_sorted) - 1
    seg_h = plot_h / segments

    for i in range(segments):
        lo = ticks_sorted[i]
        hi = ticks_sorted[i + 1]
        if lo <= y <= hi or (i == segments - 1 and math.isclose(y, hi)):
            frac = 0.0 if math.isclose(lo, hi) else (y - lo) / (hi - lo)
            # lower numeric y is visually lower on the canvas
            y_from_bottom = i * seg_h + frac * seg_h
            return int(top + plot_h - y_from_bottom)
    return int(top + plot_h)


def _save_png_pdf(img: Image.Image, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_base.with_suffix(".png"))
    img.save(out_base.with_suffix(".pdf"), "PDF", resolution=300.0)


def _draw_plot(
    title: str,
    y_label: str,
    series: Dict[str, List[Tuple[float, float]]],
    out_base: Path,
    args: argparse.Namespace,
    y_min: float | None = None,
    y_max: float | None = None,
    y_ticks: List[float] | None = None,
) -> None:
    width = args.width
    height = args.height
    left = args.margin_left
    right = args.margin_right
    top = args.margin_top
    bottom = args.margin_bottom
    plot_w = width - left - right
    plot_h = height - top - bottom

    title_font = _load_font(args.title_font_size, args.font_path)
    axis_font = _load_font(args.axis_font_size, args.font_path)
    tick_font = _load_font(args.tick_font_size, args.font_path)
    legend_font = _load_font(args.legend_font_size, args.font_path)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    all_x = [x for pts in series.values() for x, _ in pts]
    all_y = [y for pts in series.values() for _, y in pts]
    x_min = min(all_x)
    x_max = args.x_max if args.x_max > 0 else max(all_x)
    if y_ticks:
        y_min = min(y_ticks)
        y_max = max(y_ticks)
    else:
        if y_min is None:
            y_min = min(all_y)
        if y_max is None:
            y_max = max(all_y)
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0

    def to_xy(x: float, y: float) -> Tuple[int, int]:
        x = min(max(x, x_min), x_max)
        px = left + int((x - x_min) / max(1e-8, (x_max - x_min)) * plot_w)
        if y_ticks and args.equal_spacing_y:
            py = _piecewise_project(y, y_ticks, top, plot_h)
        else:
            py = top + int((1.0 - (y - y_min) / max(1e-8, (y_max - y_min))) * plot_h)
        return px, py

    if y_ticks:
        for tick in y_ticks:
            _, y = to_xy(x_min, tick)
            draw.line((left, y, left + plot_w, y), fill=(230, 230, 230), width=1)
    else:
        for i in range(args.grid_ticks_y):
            frac = i / max(1, args.grid_ticks_y - 1)
            y = top + int(frac * plot_h)
            draw.line((left, y, left + plot_w, y), fill=(230, 230, 230), width=1)
    for i in range(args.grid_ticks_x):
        frac = i / max(1, args.grid_ticks_x - 1)
        x = left + int(frac * plot_w)
        draw.line((x, top, x, top + plot_h), fill=(238, 238, 238), width=1)

    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="black", width=args.axis_line_width)
    draw.line((left, top, left, top + plot_h), fill="black", width=args.axis_line_width)

    for i in range(args.tick_count_x):
        frac = i / max(1, args.tick_count_x - 1)
        x_val = x_min + frac * (x_max - x_min)
        x = left + int(frac * plot_w)
        label = _format_k(x_val)
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, top + plot_h + args.tick_label_gap), label, fill="black", font=tick_font)

    if y_ticks:
        for y_val in sorted(y_ticks, reverse=True):
            _, y = to_xy(x_min, y_val)
            label = f"{y_val:.{args.y_tick_decimals}f}"
            bbox = draw.textbbox((0, 0), label, font=tick_font)
            draw.text((left - args.y_tick_offset - (bbox[2] - bbox[0]), y - (bbox[3] - bbox[1]) / 2), label, fill="black", font=tick_font)
    else:
        for i in range(args.tick_count_y):
            frac = i / max(1, args.tick_count_y - 1)
            y_val = y_max - frac * (y_max - y_min)
            y = top + int(frac * plot_h)
            label = f"{y_val:.{args.y_tick_decimals}f}"
            bbox = draw.textbbox((0, 0), label, font=tick_font)
            draw.text((left - args.y_tick_offset - (bbox[2] - bbox[0]), y - (bbox[3] - bbox[1]) / 2), label, fill="black", font=tick_font)

    if y_ticks and args.equal_spacing_y and len(y_ticks) >= 3 and args.show_y_break:
        break_tick = sorted(y_ticks)[1]
        _, break_y = to_xy(x_min, break_tick)
        zig = args.break_marker_size
        gap = args.break_marker_gap
        x0 = left - 2
        draw.line((x0 - zig, break_y - zig, x0, break_y), fill="black", width=args.break_marker_width)
        draw.line((x0, break_y, x0 + zig, break_y + zig), fill="black", width=args.break_marker_width)
        draw.line((x0 - zig, break_y - zig - gap, x0, break_y - gap), fill="black", width=args.break_marker_width)
        draw.line((x0, break_y - gap, x0 + zig, break_y + zig - gap), fill="black", width=args.break_marker_width)

    for exp in ("E1", "E2", "E3", "E4"):
        pts = [(x, y) for x, y in series.get(exp, []) if x <= x_max]
        if len(pts) < 2:
            continue
        poly = [to_xy(x, y) for x, y in pts]
        draw.line(poly, fill=COLORS[exp], width=args.curve_width, joint="curve")

    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_x = args.title_x if args.title_x >= 0 else left + (plot_w - (title_bbox[2] - title_bbox[0])) / 2
    draw.text((title_x, args.title_y), title, fill="black", font=title_font)

    x_label = "Environment Steps"
    x_bbox = draw.textbbox((0, 0), x_label, font=axis_font)
    x_text_x = args.x_label_x if args.x_label_x >= 0 else left + (plot_w - (x_bbox[2] - x_bbox[0])) / 2
    draw.text((x_text_x, height - args.x_label_bottom), x_label, fill="black", font=axis_font)

    draw.text((args.y_label_x, args.y_label_y), y_label, fill="black", font=axis_font)

    legend_x = args.legend_x if args.legend_x >= 0 else width - right - 320
    legend_y = args.legend_y
    for idx, exp in enumerate(("E1", "E2", "E3", "E4")):
        y = legend_y + idx * args.legend_row_gap
        draw.line((legend_x, y + args.legend_line_y, legend_x + args.legend_line_length, y + args.legend_line_y), fill=COLORS[exp], width=args.legend_line_width)
        draw.text((legend_x + args.legend_text_dx, y), LABELS[exp], fill="black", font=legend_font)

    _save_png_pdf(img, out_base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--font-path", default="")
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1100)
    parser.add_argument("--margin-left", type=int, default=170)
    parser.add_argument("--margin-right", type=int, default=90)
    parser.add_argument("--margin-top", type=int, default=120)
    parser.add_argument("--margin-bottom", type=int, default=160)
    parser.add_argument("--title-font-size", type=int, default=36)
    parser.add_argument("--axis-font-size", type=int, default=30)
    parser.add_argument("--tick-font-size", type=int, default=24)
    parser.add_argument("--legend-font-size", type=int, default=26)
    parser.add_argument("--curve-width", type=int, default=6)
    parser.add_argument("--axis-line-width", type=int, default=4)
    parser.add_argument("--legend-line-width", type=int, default=6)
    parser.add_argument("--legend-line-length", type=int, default=52)
    parser.add_argument("--legend-line-y", type=int, default=15)
    parser.add_argument("--legend-text-dx", type=int, default=70)
    parser.add_argument("--legend-row-gap", type=int, default=48)
    parser.add_argument("--legend-x", type=int, default=-1)
    parser.add_argument("--legend-y", type=int, default=150)
    parser.add_argument("--title-x", type=int, default=-1)
    parser.add_argument("--title-y", type=int, default=30)
    parser.add_argument("--x-label-x", type=int, default=-1)
    parser.add_argument("--x-label-bottom", type=int, default=70)
    parser.add_argument("--y-label-x", type=int, default=30)
    parser.add_argument("--y-label-y", type=int, default=30)
    parser.add_argument("--tick-label-gap", type=int, default=18)
    parser.add_argument("--y-tick-offset", type=int, default=18)
    parser.add_argument("--grid-ticks-x", type=int, default=6)
    parser.add_argument("--grid-ticks-y", type=int, default=6)
    parser.add_argument("--tick-count-x", type=int, default=6)
    parser.add_argument("--tick-count-y", type=int, default=6)
    parser.add_argument("--y-tick-decimals", type=int, default=2)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--x-max", type=float, default=0.0, help="Set a positive x max to zoom in.")
    parser.add_argument("--reward-y-ticks", default="", help="Comma-separated custom y ticks for reward plot.")
    parser.add_argument("--success-y-ticks", default="", help="Comma-separated custom y ticks for success plot.")
    parser.add_argument("--equal-spacing-y", action="store_true", help="Use equal visual spacing between adjacent custom y ticks.")
    parser.add_argument("--show-y-break", action="store_true", help="Draw a zigzag marker near the first internal y tick when using equal-spacing y.")
    parser.add_argument("--break-marker-size", type=int, default=10)
    parser.add_argument("--break-marker-gap", type=int, default=10)
    parser.add_argument("--break-marker-width", type=int, default=3)
    args = parser.parse_args()

    reward = _collect(DATA_DIR / "reward", smooth_window=args.smooth_window)
    success = _collect(DATA_DIR / "successRate", smooth_window=args.smooth_window)
    reward_y_ticks = _parse_tick_list(args.reward_y_ticks)
    success_y_ticks = _parse_tick_list(args.success_y_ticks)

    _draw_plot(
        # title="Reward",
        title="",
        y_label="Cumulative Reward",
        series=reward,
        out_base=OUT_DIR / "reward_curves",
        args=args,
        y_ticks=reward_y_ticks,
    )
    _draw_plot(
        # title="Success Rate",
        title="",
        y_label="Success Rate",
        series=success,
        out_base=OUT_DIR / "success_rate_curves",
        args=args,
        y_min=0.0,
        y_max=1.0,
        y_ticks=success_y_ticks,
    )
    print(OUT_DIR / "reward_curves.png")
    print(OUT_DIR / "reward_curves.pdf")
    print(OUT_DIR / "success_rate_curves.png")
    print(OUT_DIR / "success_rate_curves.pdf")


if __name__ == "__main__":
    main()
