import argparse
from pathlib import Path

import yaml

from train_multiscale_downstream import evaluate_checkpoint, load_cfg, save_metrics_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    metrics = evaluate_checkpoint(cfg, args.checkpoint, args.split)
    output_path = Path(args.output) if args.output else Path(cfg["training"]["output_dir"]) / f"{args.split}_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_metrics_json(metrics, output_path)

    print(yaml.safe_dump(metrics, sort_keys=False))
    print("Saved metrics to", output_path)


if __name__ == "__main__":
    main()
