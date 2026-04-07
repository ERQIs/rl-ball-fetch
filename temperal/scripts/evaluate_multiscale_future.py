import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets.temporal_clip_dataset import TemporalClipDataset
from src.models.multiscale_future_dynamics import ModelConfig, MultiScaleFutureDynamicsModel, compute_losses


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
        pos_mae.append(torch.abs(out["pred_pos_all"] - batch["pos_seq"]).mean(dim=(0, 1)).cpu().numpy())
        vel_mae.append(torch.abs(out["pred_vel_all"] - batch["vel_seq"]).mean(dim=(0, 1)).cpu().numpy())
        future_pos_mae.append(torch.abs(out["pred_pos_all"][:, model_cfg.history_len:] - batch["pos_seq"][:, model_cfg.history_len:]).mean(dim=(0, 1)).cpu().numpy())
        future_vel_mae.append(torch.abs(out["pred_vel_all"][:, model_cfg.history_len:] - batch["vel_seq"][:, model_cfg.history_len:]).mean(dim=(0, 1)).cpu().numpy())

    for key in ["loss", "frame", "pos", "vel"]:
        agg[key] /= max(agg["n"], 1)
    agg["pos_mae"] = np.mean(np.stack(pos_mae), axis=0).tolist()
    agg["vel_mae"] = np.mean(np.stack(vel_mae), axis=0).tolist()
    agg["future_pos_mae"] = np.mean(np.stack(future_pos_mae), axis=0).tolist()
    agg["future_vel_mae"] = np.mean(np.stack(future_vel_mae), axis=0).tolist()
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val")
    args = p.parse_args()

    cfg = load_cfg(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = ModelConfig(**ckpt["model_config"])
    model = MultiScaleFutureDynamicsModel(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["training"].get("device", "auto") == "auto" else "cpu")
    model.to(device)
    ds = build_dataset(cfg, args.split)
    loader = DataLoader(ds, batch_size=int(cfg["training"].get("batch_size", 8)), shuffle=False, num_workers=0)
    metrics = evaluate(model, loader, model_cfg, device)
    print("Evaluation metrics:", metrics)


if __name__ == "__main__":
    main()
