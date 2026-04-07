from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = THIS_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[2]
TEMPERAL_ROOT = REPO_ROOT / "temperal"


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


future_module = load_module(
    "temperal_multiscale_future_dynamics",
    TEMPERAL_ROOT / "src" / "models" / "multiscale_future_dynamics.py",
)
dataset_module = load_module(
    "temperal_temporal_clip_dataset",
    TEMPERAL_ROOT / "src" / "datasets" / "temporal_clip_dataset.py",
)
seed_module = load_module(
    "temperal_seed_utils",
    TEMPERAL_ROOT / "src" / "utils" / "seed.py",
)
spatial_module = load_module(
    "continuity_prior_multiscale_spatial_prior",
    EXPERIMENT_ROOT / "src" / "models" / "multiscale_spatial_prior.py",
)

TemporalClipDataset = dataset_module.TemporalClipDataset
ModelConfig = future_module.ModelConfig
MultiScaleEncoder = future_module.MultiScaleEncoder
MultiScaleDynamics = future_module.MultiScaleDynamics
FuturePyramidDecoder = future_module.PyramidDecoder
compute_future_losses = future_module.compute_losses
set_seed = seed_module.set_seed
F3Decoder = spatial_module.F3Decoder
MultiScaleSpatialPriorConfig = spatial_module.MultiScaleSpatialPriorConfig
SpatialPyramidDecoder = spatial_module.PyramidDecoder
compute_spatial_prior_losses = spatial_module.compute_spatial_prior_losses
warp_feature_map = spatial_module.warp_feature_map


class JointFutureSpatialModel(nn.Module):
    def __init__(self, future_cfg: ModelConfig, spatial_cfg: MultiScaleSpatialPriorConfig):
        super().__init__()
        self.future_cfg = future_cfg
        self.spatial_cfg = spatial_cfg
        self.encoder = MultiScaleEncoder(future_cfg)
        self.dynamics = MultiScaleDynamics(future_cfg)
        self.future_decoder = FuturePyramidDecoder(future_cfg) if future_cfg.enable_frame_recon else None
        self.pos_head = nn.Linear(future_cfg.s3, future_cfg.pos_dim)
        self.vel_head = nn.Linear(future_cfg.s3, future_cfg.vel_dim)
        if spatial_cfg.decoder_mode == "pyramid":
            self.spatial_decoder = SpatialPyramidDecoder(spatial_cfg)
        else:
            self.spatial_decoder = F3Decoder(spatial_cfg.c3, spatial_cfg.in_channels)

    def _probe(self, h3: torch.Tensor):
        g = h3.mean(dim=(2, 3))
        return self.pos_head(g), self.vel_head(g)

    def _decode_spatial(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor) -> torch.Tensor:
        if isinstance(self.spatial_decoder, SpatialPyramidDecoder):
            return self.spatial_decoder(f1, f2, f3)
        return self.spatial_decoder(f3)

    def forward(self, frames: torch.Tensor):
        b, t, _, _, _ = frames.shape
        assert t == self.future_cfg.seq_len
        assert self.future_cfg.history_len + self.future_cfg.future_len == self.future_cfg.seq_len

        f1_seq, f2_seq, f3_seq = [], [], []
        for i in range(t):
            f1, f2, f3 = self.encoder(frames[:, i])
            f1_seq.append(f1)
            f2_seq.append(f2)
            f3_seq.append(f3)

        f1_seq = torch.stack(f1_seq, dim=1)
        f2_seq = torch.stack(f2_seq, dim=1)
        f3_seq = torch.stack(f3_seq, dim=1)

        h1, h2, h3 = self.dynamics.init_states(b, f1_seq[:, 0], f2_seq[:, 0], f3_seq[:, 0])
        pred_pos, pred_vel, pred_future_frames = [], [], []

        for i in range(self.future_cfg.history_len):
            h1, h2, h3 = self.dynamics.update_with_observation(
                f1_seq[:, i], f2_seq[:, i], f3_seq[:, i], h1, h2, h3
            )
            p, v = self._probe(h3)
            pred_pos.append(p)
            pred_vel.append(v)

        rh1, rh2, rh3 = h1, h2, h3
        for _ in range(self.future_cfg.future_len):
            rh1, rh2, rh3 = self.dynamics.rollout_one_step(rh1, rh2, rh3)
            if self.future_decoder is not None:
                pred_future_frames.append(self.future_decoder(rh1, rh2, rh3))
            p, v = self._probe(rh3)
            pred_pos.append(p)
            pred_vel.append(v)

        return {
            "pred_pos_all": torch.stack(pred_pos, dim=1),
            "pred_vel_all": torch.stack(pred_vel, dim=1),
            "pred_future_frames": torch.stack(pred_future_frames, dim=1) if pred_future_frames else None,
            "f1_seq": f1_seq,
            "f2_seq": f2_seq,
            "f3_seq": f3_seq,
        }


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


def build_future_cfg(cfg):
    ds = cfg["dataset"]
    mdl = cfg["model"]
    loss = cfg["loss"]["future"]
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


def build_spatial_cfg(cfg):
    ds = cfg["dataset"]
    mdl = cfg["model"]
    loss = cfg["loss"]["spatial"]
    return MultiScaleSpatialPriorConfig(
        image_h=ds.get("img_size", 64),
        image_w=ds.get("img_size", 64),
        in_channels=ds.get("out_channels", 3),
        c1=mdl.get("c1", 8),
        c2=mdl.get("c2", 8),
        c3=mdl.get("c3", 8),
        decoder_mode=mdl.get("spatial_decoder_mode", "f3"),
        lambda_rec=loss.get("lambda_rec", 1.0),
        lambda_trans_f2=loss.get("lambda_trans_f2", 0.15),
        lambda_trans=loss.get("lambda_trans", 1.0),
        lambda_wd=loss.get("lambda_wd", 1.0),
        lambda_nb=loss.get("lambda_nb", 0.1),
        foreground_weight=loss.get("foreground_weight", 6.0),
        foreground_dark_threshold=loss.get("foreground_dark_threshold", 0.55),
        foreground_dilate_kernel=loss.get("foreground_dilate_kernel", 5),
    )


def extract_state_sections(ckpt_obj):
    if not isinstance(ckpt_obj, dict):
        raise ValueError("Checkpoint must be a dict.")
    if "model_state_dict" in ckpt_obj:
        state = ckpt_obj["model_state_dict"]
    elif "model" in ckpt_obj:
        state = ckpt_obj["model"]
    else:
        state = ckpt_obj
    encoder_state = {k.replace("encoder.", "", 1): v for k, v in state.items() if k.startswith("encoder.")}
    decoder_state = {k.replace("decoder.", "", 1): v for k, v in state.items() if k.startswith("decoder.")}
    return encoder_state, decoder_state


def maybe_load_spatial_init(model, cfg, device):
    ckpt_path = cfg["model"].get("init_spatial_checkpoint")
    if not ckpt_path:
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder_state, decoder_state = extract_state_sections(ckpt)
    strict = bool(cfg["model"].get("init_spatial_strict", True))
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=strict)
    print("loaded spatial encoder init from", ckpt_path)
    print("encoder missing keys:", list(missing))
    print("encoder unexpected keys:", list(unexpected))
    if decoder_state:
        dec_missing, dec_unexpected = model.spatial_decoder.load_state_dict(decoder_state, strict=False)
        print("loaded spatial decoder init from", ckpt_path)
        print("spatial decoder missing keys:", list(dec_missing))
        print("spatial decoder unexpected keys:", list(dec_unexpected))
    return ckpt_path


def pair_slice_count(cfg) -> tuple[int, int]:
    mode = cfg["loss"]["spatial"].get("pairs_mode", "all")
    history_len = int(cfg["dataset"]["history_len"])
    seq_len = int(cfg["dataset"]["seq_len"])
    if mode == "history":
        return 0, history_len
    return 0, seq_len


def compute_joint_losses(outputs, batch, future_cfg, spatial_cfg, cfg):
    future_loss, future_stats = compute_future_losses(outputs, batch, future_cfg)

    start_idx, end_idx = pair_slice_count(cfg)
    frames = batch["frames"][:, start_idx:end_idx]
    f1_seq = outputs["f1_seq"][:, start_idx:end_idx]
    f2_seq = outputs["f2_seq"][:, start_idx:end_idx]
    f3_seq = outputs["f3_seq"][:, start_idx:end_idx]

    if frames.shape[1] < 2:
        spatial_raw_loss = torch.tensor(0.0, device=frames.device)
        spatial_raw_stats = {
            "l_total": 0.0,
            "l_rec": 0.0,
            "l_trans_f2": 0.0,
            "l_trans": 0.0,
            "l_wd": 0.0,
            "l_nb": 0.0,
        }
    else:
        i_t = frames[:, :-1].reshape(-1, frames.shape[2], frames.shape[3], frames.shape[4])
        i_t1 = frames[:, 1:].reshape(-1, frames.shape[2], frames.shape[3], frames.shape[4])
        f1_t = f1_seq[:, :-1].reshape(-1, f1_seq.shape[2], f1_seq.shape[3], f1_seq.shape[4])
        f2_t = f2_seq[:, :-1].reshape(-1, f2_seq.shape[2], f2_seq.shape[3], f2_seq.shape[4])
        f3_t = f3_seq[:, :-1].reshape(-1, f3_seq.shape[2], f3_seq.shape[3], f3_seq.shape[4])
        f1_t1 = f1_seq[:, 1:].reshape(-1, f1_seq.shape[2], f1_seq.shape[3], f1_seq.shape[4])
        f2_t1 = f2_seq[:, 1:].reshape(-1, f2_seq.shape[2], f2_seq.shape[3], f2_seq.shape[4])
        f3_t1 = f3_seq[:, 1:].reshape(-1, f3_seq.shape[2], f3_seq.shape[3], f3_seq.shape[4])
        zero_flow = torch.zeros(i_t.shape[0], 2, i_t.shape[2], i_t.shape[3], device=i_t.device, dtype=i_t.dtype)
        f2_warp = warp_feature_map(f2_t, zero_flow)
        f3_warp = warp_feature_map(f3_t, zero_flow)
        i_hat_t = outputs["spatial_decode"](f1_t, f2_t, f3_t)
        i_hat_t1_from_warp = outputs["spatial_decode"](f1_t1, f2_t1, f3_warp)
        spatial_outputs = {
            "f2_warp": f2_warp,
            "f3_warp": f3_warp,
            "f2_t1": f2_t1,
            "f3_t1": f3_t1,
            "f3_t": f3_t,
            "i_hat_t": i_hat_t,
            "i_hat_t1_from_warp": i_hat_t1_from_warp,
        }
        spatial_raw_loss, spatial_raw_stats = compute_spatial_prior_losses(spatial_outputs, i_t, i_t1, spatial_cfg)

    lambda_spatial_total = float(cfg["loss"]["spatial"].get("lambda_total", 0.2))
    total_loss = future_loss + lambda_spatial_total * spatial_raw_loss

    stats = {
        "loss": float(total_loss.item()),
        "future_loss": float(future_loss.item()),
        "future_frame": float(future_stats["frame"]),
        "future_pos": float(future_stats["pos"]),
        "future_vel": float(future_stats["vel"]),
        "spatial_raw": float(spatial_raw_loss.item()),
        "spatial_scaled": float((lambda_spatial_total * spatial_raw_loss).item()),
        "spatial_rec": float(spatial_raw_stats["l_rec"]),
        "spatial_trans_f2": float(spatial_raw_stats["l_trans_f2"]),
        "spatial_trans": float(spatial_raw_stats["l_trans"]),
        "spatial_wd": float(spatial_raw_stats["l_wd"]),
        "spatial_nb": float(spatial_raw_stats["l_nb"]),
    }
    return total_loss, stats


@torch.no_grad()
def evaluate(model, loader, future_cfg, spatial_cfg, cfg, device):
    model.eval()
    agg = {
        "loss": 0.0,
        "future_loss": 0.0,
        "future_frame": 0.0,
        "future_pos": 0.0,
        "future_vel": 0.0,
        "spatial_raw": 0.0,
        "spatial_scaled": 0.0,
        "n": 0,
    }
    for batch in loader:
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        outputs = model(batch["frames"])
        outputs["spatial_decode"] = model._decode_spatial  # bound method reuse
        _, stats = compute_joint_losses(outputs, batch, future_cfg, spatial_cfg, cfg)
        for key in ["loss", "future_loss", "future_frame", "future_pos", "future_vel", "spatial_raw", "spatial_scaled"]:
            agg[key] += stats[key]
        agg["n"] += 1
    for key in ["loss", "future_loss", "future_frame", "future_pos", "future_vel", "spatial_raw", "spatial_scaled"]:
        agg[key] /= max(agg["n"], 1)
    return agg


def save_history(history, out_dir):
    path = out_dir / "history.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "train_future_loss",
        "train_future_frame",
        "train_future_pos",
        "train_future_vel",
        "train_spatial_raw",
        "train_spatial_scaled",
        "val_loss",
        "val_future_loss",
        "val_future_frame",
        "val_future_pos",
        "val_future_vel",
        "val_spatial_raw",
        "val_spatial_scaled",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_metrics_json(metrics, path):
    serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in metrics.items()}
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def save_curve_svg(history, out_path):
    if not history:
        return
    width, height = 820, 500
    ml, mr, mt, mb = 70, 30, 30, 55
    pw, ph = width - ml - mr, height - mt - mb
    epochs = [row["epoch"] for row in history]
    y_values = [row["train_loss"] for row in history] + [row["val_loss"] for row in history]
    y_min, y_max = min(y_values), max(y_values)
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
        f'<text x="{width/2:.0f}" y="20" text-anchor="middle" class="title">Joint Future + Spatial Loss Curve</text>',
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
            f'<line x1="{width-mr-180}" y1="{mt+18}" x2="{width-mr-156}" y2="{mt+18}" class="train" />',
            f'<text x="{width-mr-148}" y="{mt+22}">train_loss</text>',
            f'<line x1="{width-mr-180}" y1="{mt+42}" x2="{width-mr-156}" y2="{mt+42}" class="val" />',
            f'<text x="{width-mr-148}" y="{mt+46}">val_loss</text>',
            "</svg>",
        ]
    )
    out_path.write_text("\n".join(svg), encoding="utf-8")


def maybe_resume_training(model, optimizer, out_dir, device, resume):
    if not resume:
        return 1, float("inf"), [], None
    ckpt_path = out_dir / "last.pt"
    if not ckpt_path.exists():
        print("resume requested but no last.pt found in", out_dir)
        return 1, float("inf"), [], None
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    else:
        print("warning: resume checkpoint has no optimizer_state_dict, continuing with fresh optimizer state")
    history = ckpt.get("history", [])
    last_epoch = int(ckpt.get("epoch", len(history)))
    best_val = min((row["val_loss"] for row in history), default=float("inf"))
    init_spatial_checkpoint = ckpt.get("init_spatial_checkpoint")
    print("resumed training from", ckpt_path)
    print("resume epoch:", last_epoch)
    print("resume best val:", best_val)
    return last_epoch + 1, best_val, history, init_spatial_checkpoint


def main():
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

    future_cfg = build_future_cfg(cfg)
    spatial_cfg = build_spatial_cfg(cfg)
    model = JointFutureSpatialModel(future_cfg, spatial_cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("config/yaml", yaml.safe_dump(cfg, sort_keys=False))

    start_epoch = 1
    history = []
    best_val = float("inf")
    init_spatial_checkpoint = None
    if args.resume:
        start_epoch, best_val, history, init_spatial_checkpoint = maybe_resume_training(
            model, optimizer, out_dir, device, resume=True
        )
    else:
        init_spatial_checkpoint = maybe_load_spatial_init(model, cfg, device)
    epochs = int(train_cfg.get("epochs", 3))
    log_every = int(train_cfg.get("log_every_batches", 20))

    if start_epoch > epochs:
        print(f"nothing to do: start_epoch={start_epoch} > epochs={epochs}")
        writer.flush()
        writer.close()
        return

    print("started joint future+spatial pretraining")
    print("device:", device)
    print("train clips:", len(train_ds), "| val clips:", len(val_ds))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        agg = {
            "loss": 0.0,
            "future_loss": 0.0,
            "future_frame": 0.0,
            "future_pos": 0.0,
            "future_vel": 0.0,
            "spatial_raw": 0.0,
            "spatial_scaled": 0.0,
            "n": 0,
        }
        tracked_keys = [k for k in agg.keys() if k != "n"]
        running = {k: 0.0 for k in agg}
        total_batches = len(train_loader)
        print(f"epoch {epoch:03d} started | batches={total_batches} | device={device}")

        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            outputs = model(batch["frames"])
            outputs["spatial_decode"] = model._decode_spatial
            loss, stats = compute_joint_losses(outputs, batch, future_cfg, spatial_cfg, cfg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for key in tracked_keys:
                agg[key] += stats[key]
                running[key] += stats[key]
            agg["n"] += 1
            running["n"] += 1

            if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches:
                pct = 100.0 * batch_idx / max(total_batches, 1)
                denom = max(running["n"], 1)
                print(
                    f"  epoch {epoch:03d} progress {batch_idx}/{total_batches} ({pct:.1f}%) | "
                    f"recent total={running['loss']/denom:.4f} "
                    f"future={running['future_loss']/denom:.4f} "
                    f"spatial={running['spatial_scaled']/denom:.4f}"
                )
                running = {k: 0.0 for k in agg}

        train_stats = {k: agg[k] / max(agg["n"], 1) for k in agg if k != "n"}
        val_stats = evaluate(model, val_loader, future_cfg, spatial_cfg, cfg, device)

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_future_loss": train_stats["future_loss"],
            "train_future_frame": train_stats["future_frame"],
            "train_future_pos": train_stats["future_pos"],
            "train_future_vel": train_stats["future_vel"],
            "train_spatial_raw": train_stats["spatial_raw"],
            "train_spatial_scaled": train_stats["spatial_scaled"],
            "val_loss": val_stats["loss"],
            "val_future_loss": val_stats["future_loss"],
            "val_future_frame": val_stats["future_frame"],
            "val_future_pos": val_stats["future_pos"],
            "val_future_vel": val_stats["future_vel"],
            "val_spatial_raw": val_stats["spatial_raw"],
            "val_spatial_scaled": val_stats["spatial_scaled"],
        }
        history.append(row)

        for key, value in row.items():
            if key != "epoch":
                writer.add_scalar(key, value, epoch)

        print(
            f"epoch {epoch:03d} done | train_total={row['train_loss']:.4f} val_total={row['val_loss']:.4f} "
            f"val_future={row['val_future_loss']:.4f} val_spatial={row['val_spatial_scaled']:.4f}"
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "model": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "future_config": asdict(future_cfg),
            "spatial_config": asdict(spatial_cfg),
            "epoch": epoch,
            "history": history,
            "init_spatial_checkpoint": init_spatial_checkpoint,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save(ckpt, out_dir / "best.pt")
            save_metrics_json(val_stats, out_dir / "best_val_metrics.json")
        save_history(history, out_dir)
        save_curve_svg(history, out_dir / "loss_curve.svg")

    writer.flush()
    writer.close()
    print("Best validation loss:", best_val)
    print("Saved to", out_dir)


if __name__ == "__main__":
    main()
