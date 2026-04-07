import argparse
from pathlib import Path
from src.engine.eval_loop import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    args = p.parse_args()
    cfg = Path(args.config)
    ck = Path(args.checkpoint)
    if not cfg.exists():
        raise FileNotFoundError(cfg)
    if not ck.exists():
        raise FileNotFoundError(ck)
    evaluate(str(cfg), str(ck))


if __name__ == '__main__':
    main()
