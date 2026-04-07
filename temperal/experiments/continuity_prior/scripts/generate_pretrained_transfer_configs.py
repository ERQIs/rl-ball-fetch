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
    c1: int,
    c2: int,
    c3: int,
    s1: int,
    s2: int,
    s3: int,
    backbone_lr: float | None,
    backbone_lr_scale: float | None,
):
    run_dir = output_root / run_name
    cfg = {
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
            "c1": int(c1),
            "c2": int(c2),
            "c3": int(c3),
            "s1": int(s1),
            "s2": int(s2),
            "s3": int(s3),
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
    if backbone_lr is not None:
        cfg["training"]["backbone_lr"] = float(backbone_lr)
    if backbone_lr_scale is not None:
        cfg["training"]["backbone_lr_scale"] = float(backbone_lr_scale)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--method-tag", required=True)
    parser.add_argument("--fractions", default="0.2,0.4,1.0")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--date-tag", default="20260326")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--backbone-lr-scale", type=float, default=None)
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--c1", type=int, default=8)
    parser.add_argument("--c2", type=int, default=8)
    parser.add_argument("--c3", type=int, default=8)
    parser.add_argument("--s1", type=int, default=8)
    parser.add_argument("--s2", type=int, default=8)
    parser.add_argument("--s3", type=int, default=8)
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
            run_name = f"multiscale_transfer_{args.method_tag}_{pct}pct_obs50_seed{seed}_{args.date_tag}{suffix}"
            cfg = build_config(
                train_fraction=train_fraction,
                seed=seed,
                pretrained_checkpoint=args.pretrained_checkpoint,
                run_name=run_name,
                output_root=output_root,
                epochs=args.epochs,
                freeze_backbone_epochs=args.freeze_backbone_epochs,
                c1=args.c1,
                c2=args.c2,
                c3=args.c3,
                s1=args.s1,
                s2=args.s2,
                s3=args.s3,
                backbone_lr=args.backbone_lr,
                backbone_lr_scale=args.backbone_lr_scale,
            )
            (output_dir / f"{run_name}.yaml").write_text(
                yaml.safe_dump(cfg, sort_keys=False),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
