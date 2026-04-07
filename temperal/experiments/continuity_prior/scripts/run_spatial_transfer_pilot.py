import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        default="d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/configs/transfer_pilot_20260323",
    )
    parser.add_argument(
        "--contains",
        default="",
        help="Only run configs whose filename contains this substring.",
    )
    parser.add_argument(
        "--train-script",
        default="",
        help="Optional custom training script. Defaults to temperal/scripts/train_multiscale_downstream.py",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    temperal_root = repo_root / "temperal"
    train_script = Path(args.train_script) if args.train_script else temperal_root / "scripts" / "train_multiscale_downstream.py"
    config_dir = Path(args.config_dir)

    config_paths = sorted(config_dir.glob("*.yaml"))
    if args.contains:
        config_paths = [p for p in config_paths if args.contains in p.name]

    if not config_paths:
        raise SystemExit("No configs selected.")

    for config_path in config_paths:
        cmd = [
            sys.executable,
            str(train_script),
            "--config",
            str(config_path),
        ]
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(temperal_root) if not existing_pythonpath else f"{temperal_root}{os.pathsep}{existing_pythonpath}"
        print(f"\n=== Running {config_path.name} ===", flush=True)
        subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)


if __name__ == "__main__":
    main()
