import argparse
from pathlib import Path
from src.engine.train_loop import train_from_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    cfg = Path(args.config)
    if not cfg.exists():
        raise FileNotFoundError(cfg)
    train_from_config(str(cfg))


if __name__ == '__main__':
    main()
