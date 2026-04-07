from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_TAGS = [
    "CarCatch/SuccessRate",
    "Environment/Cumulative Reward",
    "Environment/Episode Length",
    "Policy/Entropy",
]


def latest_scalar(run_dir: Path, tag: str) -> dict | None:
    event_files = sorted(run_dir.glob("CarCatch/events.out.tfevents.*"))
    if not event_files:
        return None
    acc = EventAccumulator(str(event_files[-1]))
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return None
    events = acc.Scalars(tag)
    if not events:
        return None
    last = events[-1]
    return {"step": int(last.step), "value": float(last.value)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--tags", nargs="*", default=DEFAULT_TAGS)
    args = parser.parse_args()

    summary = {}
    for tag in args.tags:
        summary[tag] = latest_scalar(args.run_dir, tag)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
