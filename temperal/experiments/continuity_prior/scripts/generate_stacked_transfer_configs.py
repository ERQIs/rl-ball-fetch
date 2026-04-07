import argparse
from pathlib import Path

import yaml


def parse_int_list(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def build_config(
    train_fraction: float,
    seed: int,
    pretrained_checkpoint: str,
    run_name: str,
    output_root: Path,
    epochs: int,
    freeze_backbone_epochs: int,
):
    run_dir = output_root / run_name
    return {
        "dataset": {
            "root": "D:/projects/rl-ball-fetch/ball_fetch/vis_backbone/datasets/manual_capture/20260316_220723",
            "train_split_file": "temperal/data/splits_20260316_220723/train.txt",
            "val_split_file": "temperal/data/splits_20260316_220723/val.txt",
            "test_split_file": "temperal/data/splits_20260316_220723/test.txt",
            "train_fraction": float(train_fraction),
            "observation_length": 8,
            "frame_stride": 2,
            "use_last_n_frames": False,
            "observation_end_fraction": 0.5,
            "sampling_mode": "uniform_visible",
            "img_size": 64,
        },
        "model": {
            "init_mode": "finetune",
            "pretrained_checkpoint": pretrained_checkpoint,
            "c1": 8,
            "c2": 8,
            "c3": 8,
            "s1": 8,
            "s2": 8,
            "s3": 8,
            "head_hidden_dim": 64,
            "pre_head_layernorm": False,
        },
        "training": {
            "seed": int(seed),
            "device": "auto",
            "epochs": int(epochs),
            "batch_size": 8,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "freeze_backbone_epochs": int(freeze_backbone_epochs),
            "log_every_batches": 20,
            "output_dir": str(run_dir).replace("\\", "/"),
            "tensorboard_dir": str((run_dir / "tb")).replace("\\", "/"),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/configs/stacked_transfer_888_20260325",
    )
    parser.add_argument(
        "--output-root",
        default="temperal/experiments/continuity_prior/output/stacked_transfer_888_20260325",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        default="d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_888_formal_seed42_20260325/best.pt",
    )
    parser.add_argument(
        "--fractions",
        default="0.2,0.4,1.0",
    )
    parser.add_argument(
        "--seeds",
        default="42,43,44",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--date-tag",
        default="20260325",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--run-suffix",
        default="",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_root)
    fractions = parse_float_list(args.fractions)
    seeds = parse_int_list(args.seeds)
    suffix = f"_{args.run_suffix}" if args.run_suffix else ""

    for train_fraction in fractions:
        pct = int(train_fraction * 100)
        for seed in seeds:
            run_name = f"multiscale_transfer_stacked888_finetune_{pct}pct_obs50_seed{seed}_{args.date_tag}{suffix}"
            cfg = build_config(
                train_fraction=train_fraction,
                seed=seed,
                pretrained_checkpoint=args.pretrained_checkpoint,
                run_name=run_name,
                output_root=output_root,
                epochs=args.epochs,
                freeze_backbone_epochs=args.freeze_backbone_epochs,
            )
            (output_dir / f"{run_name}.yaml").write_text(
                yaml.safe_dump(cfg, sort_keys=False),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
