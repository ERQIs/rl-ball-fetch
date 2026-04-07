import argparse
import os
import subprocess
from pathlib import Path

import yaml


def make_config(
    repo_root: Path,
    mode: str,
    fraction: float,
    seed: int,
    exp_prefix: str,
    pretrained_checkpoint: str,
    pre_head_layernorm: bool,
    freeze_backbone_epochs: int,
):
    frac_tag = f"{int(round(fraction * 100)):02d}pct"
    exp_id = f"{exp_prefix}_{mode}_{frac_tag}_obs50_seed{seed}_20260316_220723"
    output_dir = repo_root / "temperal" / "outputs" / exp_id
    config_dir = repo_root / "temperal" / "configs" / "generated_transfer_grid"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{exp_id}.yaml"

    cfg = {
        "dataset": {
            "root": "D:/projects/rl-ball-fetch/ball_fetch/vis_backbone/datasets/manual_capture/20260316_220723",
            "train_split_file": "temperal/data/splits_20260316_220723/train.txt",
            "val_split_file": "temperal/data/splits_20260316_220723/val.txt",
            "test_split_file": "temperal/data/splits_20260316_220723/test.txt",
            "train_fraction": float(fraction),
            "observation_length": 8,
            "frame_stride": 2,
            "use_last_n_frames": False,
            "observation_end_fraction": 0.5,
            "sampling_mode": "uniform_visible",
            "img_size": 64,
        },
        "model": {
            "init_mode": mode,
            "c1": 8,
            "c2": 8,
            "c3": 8,
            "s1": 8,
            "s2": 8,
            "s3": 8,
            "head_hidden_dim": 64,
            "pre_head_layernorm": bool(pre_head_layernorm),
        },
        "training": {
            "seed": int(seed),
            "device": "auto",
            "epochs": 20,
            "freeze_backbone_epochs": int(freeze_backbone_epochs),
            "batch_size": 8,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "log_every_batches": 20,
            "output_dir": str(output_dir).replace("\\", "/").replace(str(repo_root).replace("\\", "/") + "/", ""),
            "tensorboard_dir": str((output_dir / "tb")).replace("\\", "/").replace(str(repo_root).replace("\\", "/") + "/", ""),
        },
    }
    if mode in ["frozen", "finetune"]:
        cfg["model"]["pretrained_checkpoint"] = pretrained_checkpoint

    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path, output_dir, exp_id


def run_command(cmd, env, cwd):
    print("Running:", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.2, 0.4, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--modes", nargs="+", default=["scratch", "frozen", "finetune"])
    parser.add_argument("--exp-prefix", default="multiscale_transfer")
    parser.add_argument(
        "--pretrained-checkpoint",
        default="temperal/outputs/multiscale_future_formal_seed42_20260316_220723/best.pt",
    )
    parser.add_argument("--pre-head-layernorm", action="store_true", default=False)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "temperal")
    python_exe = repo_root / "rl_be" / ".venv" / "Scripts" / "python.exe"

    aggregate_jobs = {}

    for fraction in args.fractions:
        frac_tag = f"{int(round(fraction * 100)):02d}pct"
        for mode in args.modes:
            aggregate_jobs[(mode, frac_tag)] = []
            for seed in args.seeds:
                config_path, output_dir, exp_id = make_config(
                    repo_root,
                    mode,
                    fraction,
                    seed,
                    args.exp_prefix,
                    args.pretrained_checkpoint,
                    args.pre_head_layernorm,
                    args.freeze_backbone_epochs,
                )
                aggregate_jobs[(mode, frac_tag)].append(output_dir)
                already_done = (output_dir / "test_metrics.json").exists() and (output_dir / "history.csv").exists()
                if args.skip_existing and already_done:
                    print(f"Skipping existing run: {exp_id}")
                    continue
                run_command([str(python_exe), str(repo_root / "temperal" / "scripts" / "train_multiscale_downstream.py"), "--config", str(config_path)], env, str(repo_root))
                run_command(
                    [
                        str(python_exe),
                        str(repo_root / "temperal" / "scripts" / "evaluate_multiscale_downstream.py"),
                        "--config",
                        str(config_path),
                        "--checkpoint",
                        str(output_dir / "best.pt"),
                        "--split",
                        "test",
                    ],
                    env,
                    str(repo_root),
                )

    for (mode, frac_tag), run_dirs in aggregate_jobs.items():
        out_dir = repo_root / "temperal" / "outputs" / f"{args.exp_prefix}_{mode}_{frac_tag}_obs50_3seed_20260316_220723"
        cmd = [
            str(python_exe),
            str(repo_root / "temperal" / "scripts" / "aggregate_multiscale_transfer_runs.py"),
            "--output-dir",
            str(out_dir),
        ]
        for run_dir in run_dirs:
            cmd.extend(["--run-dir", str(run_dir)])
        run_command(cmd, env, str(repo_root))


if __name__ == "__main__":
    main()
