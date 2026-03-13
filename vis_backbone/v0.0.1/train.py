from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import StageAConfig
from data import TrajectoryPairDataset
from model import StageABackbone
from train_step import train_step


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _build_random_batch(cfg: StageAConfig, device: torch.device) -> dict[str, torch.Tensor]:
    b, h = cfg.batch_size, cfg.image_size
    return {
        "i_t": torch.rand(b, cfg.input_channels, h, h, device=device),
        "i_t1": torch.rand(b, cfg.input_channels, h, h, device=device),
        "flow_t": torch.zeros(b, 2, h, h, device=device),
    }


def _save_checkpoint(
    model: StageABackbone,
    optimizer: torch.optim.Optimizer,
    cfg: StageAConfig,
    step: int,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "config": cfg.__dict__,
        "model": model.state_dict(),
        "encoder": model.encoder.state_dict(),
        "decoder": model.decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(ckpt, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--in-channels", type=int, default=1, choices=[1, 3])
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fg-weight", type=float, default=6.0)
    parser.add_argument("--fg-thresh", type=float, default=0.55)
    parser.add_argument("--fg-kernel", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default="artifacts")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = StageAConfig(
        batch_size=args.batch_size,
        channels=args.channels,
        input_channels=args.in_channels,
        learning_rate=args.lr,
        foreground_weight=args.fg_weight,
        foreground_dark_threshold=args.fg_thresh,
        foreground_dilate_kernel=args.fg_kernel,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StageABackbone(channels=cfg.channels, input_channels=cfg.input_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    save_dir = Path(args.save_dir)

    if args.dry_run:
        for step in range(1, args.steps + 1):
            metrics = train_step(model, _build_random_batch(cfg, device), optimizer, cfg)
            if step == 1 or step % 10 == 0:
                print(f"step={step:04d} total={metrics['l_total']:.4f} rec={metrics['l_rec']:.4f}")
        _save_checkpoint(model, optimizer, cfg, args.steps, save_dir / "stagea_last.pt")
        print(f"saved: {save_dir / 'stagea_last.pt'}")
        return

    if not args.dataset_root:
        raise ValueError("Please provide --dataset-root or use --dry-run")
    ds = TrajectoryPairDataset(Path(args.dataset_root), input_channels=cfg.input_channels)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    data_iter = iter(loader)

    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = _to_device(batch, device)
        metrics = train_step(model, batch, optimizer, cfg)
        if step == 1 or step % 10 == 0:
            print(
                f"step={step:04d} total={metrics['l_total']:.4f} rec={metrics['l_rec']:.4f} "
                f"trans={metrics['l_trans']:.4f} wd={metrics['l_wd']:.4f}"
            )
        if args.save_every > 0 and step % args.save_every == 0:
            step_ckpt = save_dir / f"stagea_step_{step:04d}.pt"
            _save_checkpoint(model, optimizer, cfg, step, step_ckpt)
            print(f"saved: {step_ckpt}")

    _save_checkpoint(model, optimizer, cfg, args.steps, save_dir / "stagea_last.pt")
    print(f"saved: {save_dir / 'stagea_last.pt'}")


if __name__ == "__main__":
    main()
