from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def load_shared_module():
    temperal_root = Path(__file__).resolve().parents[3]
    shared_path = temperal_root / "scripts" / "train_multiscale_downstream.py"
    spec = importlib.util.spec_from_file_location("shared_train_multiscale_downstream", shared_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load shared module from {shared_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = load_shared_module()


def build_optimizer(model, cfg):
    tr_cfg = cfg["training"]
    base_lr = float(tr_cfg.get("lr", 1e-3))
    weight_decay = float(tr_cfg.get("weight_decay", 0.0))

    backbone_lr = tr_cfg.get("backbone_lr")
    if backbone_lr is None:
        scale = tr_cfg.get("backbone_lr_scale")
        backbone_lr = base_lr if scale is None else base_lr * float(scale)
    backbone_lr = float(backbone_lr)

    backbone_params = [p for module in (model.encoder, model.dynamics) for p in module.parameters() if p.requires_grad]
    head_params = [p for module in (model.pre_head_norm, model.head) for p in module.parameters() if p.requires_grad]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr})
    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimizer.")

    return torch.optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = shared.load_cfg(args.config)
    tr_cfg = cfg["training"]
    out_dir = Path(tr_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(tr_cfg.get("tensorboard_dir", out_dir / "tb"))
    tb_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    device, model_cfg, model = shared.build_runtime(cfg)

    train_ds = shared.build_dataset(cfg, "train")
    val_ds = shared.build_dataset(cfg, "val")
    train_loader = DataLoader(train_ds, batch_size=int(tr_cfg.get("batch_size", 8)), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(tr_cfg.get("batch_size", 8)), shuffle=False, num_workers=0)

    init_mode = cfg["model"].get("init_mode", "scratch")
    freeze_backbone_epochs = int(tr_cfg.get("freeze_backbone_epochs", 0))
    backbone_is_trainable = init_mode != "finetune" or freeze_backbone_epochs <= 0
    if init_mode == "finetune" and freeze_backbone_epochs > 0:
        model.set_backbone_trainable(False)

    optimizer = build_optimizer(model, cfg)
    loss_fn = torch.nn.SmoothL1Loss()
    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("config/yaml", yaml.safe_dump(cfg, sort_keys=False))

    best_val = float("inf")
    history = []
    epochs = int(tr_cfg.get("epochs", 20))
    log_every = int(tr_cfg.get("log_every_batches", 20))

    print(
        "optimizer setup | "
        f"lr={float(tr_cfg.get('lr', 1e-3)):.6f} "
        f"backbone_lr={float(tr_cfg.get('backbone_lr', float(tr_cfg.get('lr', 1e-3)) * float(tr_cfg.get('backbone_lr_scale', 1.0)))):.6f}"
    )

    for epoch in range(1, epochs + 1):
        if init_mode == "finetune" and freeze_backbone_epochs > 0:
            should_train_backbone = epoch > freeze_backbone_epochs
            if should_train_backbone != backbone_is_trainable:
                model.set_backbone_trainable(should_train_backbone)
                optimizer = build_optimizer(model, cfg)
                backbone_is_trainable = should_train_backbone

        model.train()
        losses = []
        total_batches = len(train_loader)
        running = []
        print(f"epoch {epoch:03d} started | batches={total_batches} | device={device} | mode={init_mode}")

        for batch_idx, batch in enumerate(train_loader, start=1):
            frames = batch["frames"].to(device)
            targets = batch["target_xy"].to(device)
            preds = model(frames)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            value = loss.item()
            losses.append(value)
            running.append(value)
            if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches:
                pct = 100.0 * batch_idx / max(total_batches, 1)
                print(
                    f"  epoch {epoch:03d} progress {batch_idx}/{total_batches} ({pct:.1f}%) "
                    f"| recent train_loss={float(np.mean(running)):.4f}"
                )
                running = []

        train_loss = float(np.mean(losses))
        val_stats = shared.evaluate(model, val_loader, device, loss_fn)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_stats["loss"],
            "val_mean_l2": val_stats["mean_l2"],
            "val_median_l2": val_stats["median_l2"],
            "val_success_at_0.2": val_stats["success_at_0.2"],
        }
        history.append(row)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_stats["loss"], epoch)
        writer.add_scalar("metric/val_mean_l2", val_stats["mean_l2"], epoch)
        writer.add_scalar("metric/val_median_l2", val_stats["median_l2"], epoch)
        writer.add_scalar("metric/val_success_at_0.2", val_stats["success_at_0.2"], epoch)

        print(
            f"epoch {epoch:03d} done | train_loss={train_loss:.4f} val_loss={val_stats['loss']:.4f} "
            f"val_mean_l2={val_stats['mean_l2']:.4f} val_success@0.2={val_stats['success_at_0.2']:.4f}"
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "model": model.state_dict(),
            "config": cfg,
            "transfer_config": shared.asdict(model_cfg),
            "epoch": epoch,
            "history": history,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(ckpt, out_dir / "best.pt")
            shared.save_metrics_json(val_stats, out_dir / "best_val_metrics.json")
        shared.save_history(history, out_dir)
        shared.save_curve_svg(
            history,
            out_dir / "loss_curve.svg",
            title="Loss Curve",
            y_label="Loss",
            keys_and_colors=[
                ("train_loss", "#2563eb", "train_loss"),
                ("val_loss", "#dc2626", "val_loss"),
            ],
        )
        shared.save_curve_svg(
            history,
            out_dir / "success_curve.svg",
            title="Success Curve @ 0.2",
            y_label="Success Rate",
            keys_and_colors=[
                ("val_success_at_0.2", "#059669", "val_success@0.2"),
            ],
        )

    writer.flush()
    writer.close()
    print("Best validation loss:", best_val)
    print("Saved to", out_dir)


if __name__ == "__main__":
    main()
