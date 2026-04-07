import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = THIS_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[2]
TEMPERAL_ROOT = REPO_ROOT / "temperal"
if str(TEMPERAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPERAL_ROOT))

from src.datasets.temporal_clip_dataset import TemporalClipDataset
from src.models.multiscale_future_dynamics import ModelConfig, MultiScaleFutureDynamicsModel, compute_losses
from src.utils.seed import set_seed


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(cfg, split):
    ds = cfg["dataset"]
    return TemporalClipDataset(
        root=ds["root"],
        split_file=ds[f"{split}_split_file"],
        seq_len=ds["seq_len"],
        frame_stride=ds.get("frame_stride", 1),
        clip_start_stride=ds.get("clip_start_stride", 1),
        img_size=ds.get("img_size", 64),
        grayscale=ds.get("grayscale", True),
        out_channels=ds.get("out_channels", 3),
        pos_cols=ds.get("pos_cols"),
        vel_cols=ds.get("vel_cols"),
        max_clips=ds.get(f"{split}_max_clips"),
    )


def build_model_cfg(cfg):
    ds = cfg["dataset"]
    mdl = cfg["model"]
    loss = cfg["loss"]
    return ModelConfig(
        image_h=ds.get("img_size", 64),
        image_w=ds.get("img_size", 64),
        in_channels=ds.get("out_channels", 3),
        seq_len=ds["seq_len"],
        history_len=ds["history_len"],
        future_len=ds["future_len"],
        future_recon_len=mdl.get("future_recon_len", ds["future_len"]),
        enable_frame_recon=mdl.get("enable_frame_recon", True),
        pos_dim=len(ds.get("pos_cols", ["ball_px", "ball_py", "ball_pz"])),
        vel_dim=len(ds.get("vel_cols", ["ball_vx", "ball_vy", "ball_vz"])),
        c1=mdl.get("c1", 8),
        c2=mdl.get("c2", 8),
        c3=mdl.get("c3", 8),
        s1=mdl.get("s1", 8),
        s2=mdl.get("s2", 8),
        s3=mdl.get("s3", 8),
        decoder_proj_ch=mdl.get("decoder_proj_ch", 8),
        decoder_hidden_ch=mdl.get("decoder_hidden_ch", 8),
        lambda_frame=loss.get("lambda_frame", 1.0),
        lambda_p=loss.get("lambda_p", 1.0),
        lambda_v=loss.get("lambda_v", 0.2),
    )


def extract_encoder_state_dict(ckpt_obj):
    if not isinstance(ckpt_obj, dict):
        raise ValueError("Checkpoint must be a dict.")
    if "encoder_state_dict" in ckpt_obj:
        return ckpt_obj["encoder_state_dict"]
    if "model_state_dict" in ckpt_obj:
        msd = ckpt_obj["model_state_dict"]
        encoder_keys = {k[len("encoder."):]: v for k, v in msd.items() if k.startswith("encoder.")}
        if encoder_keys:
            return encoder_keys
    raise ValueError("Could not find encoder weights in checkpoint.")


def maybe_load_encoder_init(model, cfg, device):
    ckpt_path = cfg["model"].get("init_encoder_checkpoint")
    if not ckpt_path:
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    enc_state = extract_encoder_state_dict(ckpt)
    missing, unexpected = model.encoder.load_state_dict(
        enc_state,
        strict=bool(cfg["model"].get("init_encoder_strict", True)),
    )
    print("loaded encoder init from", ckpt_path)
    print("encoder missing keys:", list(missing))
    print("encoder unexpected keys:", list(unexpected))
    return ckpt_path


def maybe_resume_training(model, optimizer, out_dir, device, resume):
    if not resume:
        return 1, float("inf"), [], None
    ckpt_path = out_dir / "last.pt"
    if not ckpt_path.exists():
        print("resume requested but no last.pt found in", out_dir)
        return 1, float("inf"), [], None
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    history = ckpt.get("history", [])
    last_epoch = int(ckpt.get("epoch", len(history)))
    best_val = min((row["val_loss"] for row in history), default=float("inf"))
    init_encoder_checkpoint = ckpt.get("init_encoder_checkpoint")
    print("resumed training from", ckpt_path)
    print("resume epoch:", last_epoch)
    print("resume best val:", best_val)
    return last_epoch + 1, best_val, history, init_encoder_checkpoint


@torch.no_grad()
def evaluate(model, loader, model_cfg, device):
    model.eval()
    agg = {"loss": 0.0, "frame": 0.0, "pos": 0.0, "vel": 0.0, "n": 0}
    pos_mae = []
    vel_mae = []
    future_pos_mae = []
    future_vel_mae = []

    for batch in loader:
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        out = model(batch["frames"])
        _, stats = compute_losses(out, batch, model_cfg)
        for key in ["loss", "frame", "pos", "vel"]:
            agg[key] += stats[key]
        agg["n"] += 1

        pos_err = torch.abs(out["pred_pos_all"] - batch["pos_seq"]).mean(dim=(0, 1)).cpu().numpy()
        vel_err = torch.abs(out["pred_vel_all"] - batch["vel_seq"]).mean(dim=(0, 1)).cpu().numpy()
        future_pos_err = torch.abs(
            out["pred_pos_all"][:, model_cfg.history_len:] - batch["pos_seq"][:, model_cfg.history_len:]
        ).mean(dim=(0, 1)).cpu().numpy()
        future_vel_err = torch.abs(
            out["pred_vel_all"][:, model_cfg.history_len:] - batch["vel_seq"][:, model_cfg.history_len:]
        ).mean(dim=(0, 1)).cpu().numpy()
        pos_mae.append(pos_err)
        vel_mae.append(vel_err)
        future_pos_mae.append(future_pos_err)
        future_vel_mae.append(future_vel_err)

    for key in ["loss", "frame", "pos", "vel"]:
        agg[key] /= max(agg["n"], 1)

    agg["pos_mae"] = np.mean(np.stack(pos_mae), axis=0).tolist() if pos_mae else []
    agg["vel_mae"] = np.mean(np.stack(vel_mae), axis=0).tolist() if vel_mae else []
    agg["future_pos_mae"] = np.mean(np.stack(future_pos_mae), axis=0).tolist() if future_pos_mae else []
    agg["future_vel_mae"] = np.mean(np.stack(future_vel_mae), axis=0).tolist() if future_vel_mae else []
    return agg


def save_history(history, out_dir):
    path = out_dir / "history.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "train_frame",
        "train_pos",
        "train_vel",
        "val_loss",
        "val_frame",
        "val_pos",
        "val_vel",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_curve_svg(history, out_path):
    if not history:
        return
    width, height = 800, 480
    ml, mr, mt, mb = 70, 30, 30, 55
    pw, ph = width - ml - mr, height - mt - mb
    epochs = [row["epoch"] for row in history]
    train_vals = [row["train_loss"] for row in history]
    val_vals = [row["val_loss"] for row in history]
    y_min, y_max = min(train_vals + val_vals), max(train_vals + val_vals)
    if abs(y_max - y_min) < 1e-8:
        y_max = y_min + 1.0

    def x_pos(epoch):
        if len(epochs) == 1:
            return ml + pw / 2
        return ml + (epoch - epochs[0]) / (epochs[-1] - epochs[0]) * pw

    def y_pos(value):
        return mt + (y_max - value) / (y_max - y_min) * ph

    def polyline(key):
        return " ".join(f"{x_pos(row['epoch']):.2f},{y_pos(row[key]):.2f}" for row in history)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#333;stroke-width:1.5}.train{fill:none;stroke:#2563eb;stroke-width:2.5}.val{fill:none;stroke:#dc2626;stroke-width:2.5}.title{font-size:18px;font-weight:bold}</style>',
        f'<text x="{width/2:.0f}" y="20" text-anchor="middle" class="title">Future Pretrain Loss Curve</text>',
    ]
    for tick in np.linspace(y_min, y_max, num=5):
        y = y_pos(float(tick))
        svg.append(f'<line x1="{ml}" y1="{y:.2f}" x2="{width-mr}" y2="{y:.2f}" class="grid" />')
        svg.append(f'<text x="{ml-10}" y="{y+4:.2f}" text-anchor="end">{tick:.4f}</text>')
    for tick in sorted(set([epochs[0], epochs[-1], max(1, epochs[-1] // 2)])):
        x = x_pos(tick)
        svg.append(f'<line x1="{x:.2f}" y1="{mt}" x2="{x:.2f}" y2="{height-mb}" class="grid" />')
        svg.append(f'<text x="{x:.2f}" y="{height-mb+20}" text-anchor="middle">{tick}</text>')
    svg.extend(
        [
            f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" class="axis" />',
            f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" class="axis" />',
            f'<text x="{width/2:.0f}" y="{height-15}" text-anchor="middle">Epoch</text>',
            f'<text x="18" y="{height/2:.0f}" text-anchor="middle" transform="rotate(-90 18 {height/2:.0f})">Loss</text>',
            f'<polyline points="{polyline("train_loss")}" class="train" />',
            f'<polyline points="{polyline("val_loss")}" class="val" />',
            f'<line x1="{width-mr-140}" y1="{mt+18}" x2="{width-mr-116}" y2="{mt+18}" class="train" />',
            f'<text x="{width-mr-108}" y="{mt+22}">train_loss</text>',
            f'<line x1="{width-mr-140}" y1="{mt+42}" x2="{width-mr-116}" y2="{mt+42}" class="val" />',
            f'<text x="{width-mr-108}" y="{mt+46}">val_loss</text>',
            "</svg>",
        ]
    )
    out_path.write_text("\n".join(svg), encoding="utf-8")


def main():
    print("started training multiscale future dynamics model with spatial encoder init")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    train_cfg = cfg["training"]
    out_dir = Path(train_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(train_cfg.get("tensorboard_dir", out_dir / "tb"))
    tb_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    set_seed(int(train_cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() and train_cfg.get("device", "auto") == "auto" else "cpu")

    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    train_loader = DataLoader(train_ds, batch_size=int(train_cfg.get("batch_size", 8)), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(train_cfg.get("batch_size", 8)), shuffle=False, num_workers=0)

    model_cfg = build_model_cfg(cfg)
    model = MultiScaleFutureDynamicsModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    start_epoch = 1
    best_val = float("inf")
    history = []
    init_encoder_checkpoint = None
    if args.resume:
        start_epoch, best_val, history, init_encoder_checkpoint = maybe_resume_training(
            model, optimizer, out_dir, device, resume=True
        )
    else:
        init_encoder_checkpoint = maybe_load_encoder_init(model, cfg, device)

    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("config/yaml", yaml.safe_dump(cfg, sort_keys=False))

    epochs = int(train_cfg.get("epochs", 3))
    log_every = int(train_cfg.get("log_every_batches", 20))

    if start_epoch > epochs:
        print(f"nothing to do: start_epoch={start_epoch} > epochs={epochs}")
        writer.flush()
        writer.close()
        return

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        agg = {"loss": 0.0, "frame": 0.0, "pos": 0.0, "vel": 0.0, "n": 0}
        running = {"loss": 0.0, "frame": 0.0, "pos": 0.0, "vel": 0.0, "n": 0}
        total_batches = len(train_loader)
        print(f"epoch {epoch:03d} started | batches={total_batches} | device={device}")
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            out = model(batch["frames"])
            loss, stats = compute_losses(out, batch, model_cfg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for key in ["loss", "frame", "pos", "vel"]:
                agg[key] += stats[key]
                running[key] += stats[key]
            agg["n"] += 1
            running["n"] += 1

            if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches:
                pct = 100.0 * batch_idx / max(total_batches, 1)
                denom = max(running["n"], 1)
                print(
                    f"  epoch {epoch:03d} progress {batch_idx}/{total_batches} ({pct:.1f}%) | "
                    f"recent loss={running['loss']/denom:.4f} "
                    f"frame={running['frame']/denom:.4f} "
                    f"pos={running['pos']/denom:.4f} "
                    f"vel={running['vel']/denom:.4f}"
                )
                running = {"loss": 0.0, "frame": 0.0, "pos": 0.0, "vel": 0.0, "n": 0}

        for key in ["loss", "frame", "pos", "vel"]:
            agg[key] /= max(agg["n"], 1)

        val_stats = evaluate(model, val_loader, model_cfg, device)
        row = {
            "epoch": epoch,
            "train_loss": agg["loss"],
            "train_frame": agg["frame"],
            "train_pos": agg["pos"],
            "train_vel": agg["vel"],
            "val_loss": val_stats["loss"],
            "val_frame": val_stats["frame"],
            "val_pos": val_stats["pos"],
            "val_vel": val_stats["vel"],
        }
        history.append(row)

        writer.add_scalar("loss/train_total", agg["loss"], epoch)
        writer.add_scalar("loss/train_frame", agg["frame"], epoch)
        writer.add_scalar("loss/train_pos", agg["pos"], epoch)
        writer.add_scalar("loss/train_vel", agg["vel"], epoch)
        writer.add_scalar("loss/val_total", val_stats["loss"], epoch)
        writer.add_scalar("loss/val_frame", val_stats["frame"], epoch)
        writer.add_scalar("loss/val_pos", val_stats["pos"], epoch)
        writer.add_scalar("loss/val_vel", val_stats["vel"], epoch)

        for i, value in enumerate(val_stats["pos_mae"]):
            writer.add_scalar(f"mae/val_pos_dim_{i}", value, epoch)
        for i, value in enumerate(val_stats["vel_mae"]):
            writer.add_scalar(f"mae/val_vel_dim_{i}", value, epoch)
        for i, value in enumerate(val_stats["future_pos_mae"]):
            writer.add_scalar(f"mae/val_future_pos_dim_{i}", value, epoch)
        for i, value in enumerate(val_stats["future_vel_mae"]):
            writer.add_scalar(f"mae/val_future_vel_dim_{i}", value, epoch)

        print(
            f"epoch {epoch:03d} | train loss={agg['loss']:.4f} frame={agg['frame']:.4f} pos={agg['pos']:.4f} vel={agg['vel']:.4f} | "
            f"val loss={val_stats['loss']:.4f} frame={val_stats['frame']:.4f} pos={val_stats['pos']:.4f} vel={val_stats['vel']:.4f}"
        )
        print(
            f"           val pos_mae={np.round(val_stats['pos_mae'], 4).tolist()} "
            f"future_pos_mae={np.round(val_stats['future_pos_mae'], 4).tolist()}"
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
            "config": cfg,
            "model_config": asdict(model_cfg),
            "init_encoder_checkpoint": init_encoder_checkpoint,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(ckpt, out_dir / "best.pt")
        save_history(history, out_dir)
        save_curve_svg(history, out_dir / "loss_curve.svg")

    writer.flush()
    writer.close()
    print("Best validation loss:", best_val)
    print("Saved to", out_dir)


if __name__ == "__main__":
    main()
