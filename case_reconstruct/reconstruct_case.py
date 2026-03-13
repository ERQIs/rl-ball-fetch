from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from tensorboard.backend.event_processing import event_accumulator


KEYS_OF_INTEREST = [
    "Environment/Cumulative Reward",
    "CarCatch/SuccessRate",
    "Environment/Episode Length",
    "Policy/Entropy",
    "Losses/Policy Loss",
    "Losses/Value Loss",
]


def _find_behavior_dir(run_dir: Path) -> Path:
    if (run_dir / "CarCatch").exists():
        return run_dir / "CarCatch"
    for child in run_dir.iterdir():
        if child.is_dir():
            if list(child.glob("events.out.tfevents.*")):
                return child
    raise FileNotFoundError(f"Cannot find behavior dir with tfevents under: {run_dir}")


def _load_scalars(behavior_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
    acc = event_accumulator.EventAccumulator(str(behavior_dir))
    acc.Reload()
    tags = set(acc.Tags().get("scalars", []))
    data: Dict[str, List[Tuple[int, float]]] = {}
    for key in KEYS_OF_INTEREST:
        if key in tags:
            data[key] = [(e.step, float(e.value)) for e in acc.Scalars(key)]
    return data


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _moving_average(values: List[float], window: int) -> List[float]:
    if not values:
        return []
    out: List[float] = []
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= window:
            running -= values[i - window]
        denom = min(i + 1, window)
        out.append(running / denom)
    return out


def _slope_last_n(points: List[Tuple[int, float]], n: int = 30) -> float:
    if len(points) < 2:
        return 0.0
    seg = points[-n:]
    xs = [float(p[0]) for p in seg]
    ys = [float(p[1]) for p in seg]
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


def _extract_scene_params(scene_path: Path) -> Dict[str, float]:
    keys = [
        "precisionK",
        "progressRewardScale",
        "precisionRewardScale",
        "landingProgressRewardScale",
        "controlPenaltyScale",
        "catchReward",
        "missReward",
        "outOfArenaPenalty",
    ]
    text = scene_path.read_text(encoding="utf-8", errors="ignore")
    params: Dict[str, float] = {}
    for key in keys:
        m = re.search(rf"^\s*{re.escape(key)}:\s*(-?\d+(?:\.\d+)?)\s*$", text, re.MULTILINE)
        if m:
            params[key] = float(m.group(1))
    return params


def build_report(run_dir: Path, scene_path: Path | None, ma_window: int) -> Dict:
    behavior_dir = _find_behavior_dir(run_dir)
    scalars = _load_scalars(behavior_dir)
    report: Dict = {
        "run_dir": str(run_dir),
        "behavior_dir": str(behavior_dir),
        "metrics": {},
        "diagnosis": [],
    }

    for key, points in scalars.items():
        vals = [v for _, v in points]
        ma = _moving_average(vals, ma_window)
        report["metrics"][key] = {
            "count": len(vals),
            "first": vals[0] if vals else None,
            "last": vals[-1] if vals else None,
            "best": max(vals) if vals else None,
            "worst": min(vals) if vals else None,
            "last_ma": ma[-1] if ma else None,
            "slope_last": _slope_last_n(points, n=30),
        }

    rew = scalars.get("Environment/Cumulative Reward", [])
    suc = scalars.get("CarCatch/SuccessRate", [])
    if rew and suc:
        rew_last = rew[-1][1]
        suc_last = suc[-1][1]
        report["diagnosis"].append(
            f"Last cumulative reward={rew_last:.3f}, success rate={suc_last:.3f}."
        )
        reward_slope = _slope_last_n(rew, n=30)
        success_slope = _slope_last_n(suc, n=30)
        if abs(success_slope) < 1e-7:
            report["diagnosis"].append("Success curve looks plateaued (near-zero slope in tail).")
        if reward_slope < 0 and suc_last > 0.1:
            report["diagnosis"].append(
                "Reward trend is negative while success is non-trivial; dense penalties may dominate."
            )

    if scene_path is not None and scene_path.exists():
        params = _extract_scene_params(scene_path)
        report["scene_params"] = params
        if "catchReward" in params and "missReward" in params and suc:
            p = suc[-1][1]
            terminal_expect = p * params["catchReward"] + (1.0 - p) * params["missReward"]
            report["terminal_reward_expectation"] = terminal_expect
            if rew:
                gap = terminal_expect - rew[-1][1]
                report["diagnosis"].append(
                    f"Terminal reward expectation={terminal_expect:.3f}, "
                    f"current cumulative reward={rew[-1][1]:.3f}, gap={gap:.3f}."
                )
                if gap > 10:
                    report["diagnosis"].append(
                        "Large positive gap: step penalties or non-success terminations likely dominate."
                    )

    if not report["diagnosis"]:
        report["diagnosis"].append("No obvious anomaly detected from available metrics.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct and diagnose one RL run.")
    parser.add_argument("--run-dir", required=True, help="Path to one ML-Agents run folder.")
    parser.add_argument(
        "--scene",
        default="",
        help="Optional Unity scene file path for reward parameter extraction.",
    )
    parser.add_argument("--ma-window", type=int, default=20)
    parser.add_argument(
        "--out",
        default="",
        help="Optional output json path. Default: <run-dir>/case_reconstruct_report.json",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    scene = Path(args.scene).resolve() if args.scene else None
    report = build_report(run_dir, scene, args.ma_window)

    out_path = Path(args.out).resolve() if args.out else run_dir / "case_reconstruct_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"report: {out_path}")
    for line in report.get("diagnosis", []):
        print(f"- {line}")


if __name__ == "__main__":
    main()
