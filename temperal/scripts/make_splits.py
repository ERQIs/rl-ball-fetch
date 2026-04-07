import argparse
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--out_dir', default='.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ratios', nargs=3, type=float, default=[0.7,0.15,0.15])
    args = p.parse_args()
    root = Path(args.data_root)
    eps = sorted([p.name for p in root.iterdir() if p.is_dir()])
    random.seed(args.seed)
    random.shuffle(eps)
    n = len(eps)
    a = int(n * args.ratios[0])
    b = int(n * (args.ratios[0] + args.ratios[1]))
    train = eps[:a]
    val = eps[a:b]
    test = eps[b:]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'train.txt').write_text('\n'.join(train))
    (out / 'val.txt').write_text('\n'.join(val))
    (out / 'test.txt').write_text('\n'.join(test))
    print('Wrote splits to', out)


if __name__ == '__main__':
    main()
