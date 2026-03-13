from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from data import TrajectoryPairDataset
from model import StageABackbone


def _to_u8_image(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(0.0, 1.0)
    if x.shape[0] == 1:
        img = (x[0].numpy() * 255.0).astype(np.uint8)
        return np.stack([img, img, img], axis=-1)
    img = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return img


def _load_model(ckpt_path: Path, device: torch.device, default_channels: int, default_in_channels: int) -> StageABackbone:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    channels = int(cfg.get("channels", default_channels))
    in_channels = int(cfg.get("input_channels", default_in_channels))
    model = StageABackbone(channels=channels, input_channels=in_channels).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--in-channels", type=int, default=1, choices=[1, 3])
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--output", type=str, default="artifacts/recon_grid.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(Path(args.checkpoint), device, default_channels=64, default_in_channels=args.in_channels)
    dataset = TrajectoryPairDataset(args.dataset_root, input_channels=args.in_channels)

    num_samples = max(1, min(args.num_samples, len(dataset)))
    rows = []
    col_names = ["I_t", "I_hat_t", "I_t1", "I_hat_t1_from_warp"]

    with torch.no_grad():
        for i in range(num_samples):
            item = dataset[i]
            i_t = item["i_t"].unsqueeze(0).to(device)
            i_t1 = item["i_t1"].unsqueeze(0).to(device)
            flow_t = item["flow_t"].unsqueeze(0).to(device)
            out = model(i_t, i_t1, flow_t)
            row = [
                _to_u8_image(i_t[0]),
                _to_u8_image(out["i_hat_t"][0]),
                _to_u8_image(i_t1[0]),
                _to_u8_image(out["i_hat_t1_from_warp"][0]),
            ]
            rows.append(row)

    h, w, _ = rows[0][0].shape
    pad = 6
    header_h = 24
    canvas_h = header_h + num_samples * h + (num_samples + 1) * pad
    canvas_w = len(col_names) * w + (len(col_names) + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    for c, name in enumerate(col_names):
        x = pad + c * (w + pad)
        draw.text((x + 2, 4), name, fill=(235, 235, 235))

    for r, row in enumerate(rows):
        y = header_h + pad + r * (h + pad)
        for c, arr in enumerate(row):
            x = pad + c * (w + pad)
            canvas.paste(Image.fromarray(arr), (x, y))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
