import argparse
from pathlib import Path

import yaml


def build_config(
    train_fraction: float,
    seed: int,
    init_mode: str,
    pretrained_checkpoint: str | None,
    run_name: str,
    output_root: Path,
    epochs: int,
    freeze_backbone_epochs: int,
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
            "init_mode": init_mode,
            "c1": 16,
            "c2": 32,
            "c3": 64,
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
    if pretrained_checkpoint:
        cfg["model"]["pretrained_checkpoint"] = pretrained_checkpoint
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/configs/transfer_pilot_20260323",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        default="d:/projects/rl-ball-fetch/temperal/outputs/multiscale_spatial_prior_train_20260323/best.pt",
    )
    parser.add_argument(
        "--output-root",
        default="temperal/experiments/continuity_prior/output/spatial_prior_transfer_pilot_20260323",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--date-tag",
        default="20260323",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--modes",
        default="scratch,finetune",
        help="Comma-separated subset of {scratch,finetune}.",
    )
    parser.add_argument(
        "--run-suffix",
        default="",
        help="Optional suffix appended to each run name, e.g. warmup5_40ep.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_root)
    requested_modes = {m.strip() for m in args.modes.split(",") if m.strip()}
    suffix = f"_{args.run_suffix}" if args.run_suffix else ""

    for train_fraction in [0.2, 1.0]:
        pct = int(train_fraction * 100)
        for seed in [41, 42, 43]:
            if "scratch" in requested_modes:
                scratch_name = f"multiscale_transfer_scratch_spatialpilot_{pct}pct_obs50_seed{seed}_{args.date_tag}{suffix}"
                scratch_cfg = build_config(
                    train_fraction,
                    seed,
                    "scratch",
                    None,
                    scratch_name,
                    output_root=output_root,
                    epochs=args.epochs,
                    freeze_backbone_epochs=args.freeze_backbone_epochs,
                )
                (out_dir / f"{scratch_name}.yaml").write_text(
                    yaml.safe_dump(scratch_cfg, sort_keys=False),
                    encoding="utf-8",
                )

            if "finetune" in requested_modes:
                finetune_name = f"multiscale_transfer_spatial_finetune_{pct}pct_obs50_seed{seed}_{args.date_tag}{suffix}"
                finetune_cfg = build_config(
                    train_fraction,
                    seed,
                    "finetune",
                    args.pretrained_checkpoint,
                    finetune_name,
                    output_root=output_root,
                    epochs=args.epochs,
                    freeze_backbone_epochs=args.freeze_backbone_epochs,
                )
                (out_dir / f"{finetune_name}.yaml").write_text(
                    yaml.safe_dump(finetune_cfg, sort_keys=False),
                    encoding="utf-8",
                )


if __name__ == "__main__":
    main()
