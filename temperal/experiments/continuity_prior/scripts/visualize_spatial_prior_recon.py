from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = THIS_DIR.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.datasets.frame_pair_dataset import FramePairDataset
from src.models.multiscale_spatial_prior import MultiScaleSpatialPriorConfig, MultiScaleSpatialPriorModel


def _to_u8_image(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(0.0, 1.0)
    if x.shape[0] == 1:
        img = (x[0].numpy() * 255.0).astype(np.uint8)
        return np.stack([img, img, img], axis=-1)
    img = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return img


def _load_model(ckpt_path: Path, device: torch.device) -> MultiScaleSpatialPriorModel:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("spatial_prior_config") or {}
    cfg = MultiScaleSpatialPriorConfig(**cfg_dict)
    model = MultiScaleSpatialPriorModel(cfg).to(device)
    state_dict = ckpt.get("model_state_dict") or ckpt.get("model")
    if state_dict is None:
        raise RuntimeError(f"checkpoint {ckpt_path} does not contain model weights")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _pick_episode_indices(dataset: FramePairDataset, episode_id: str | None, num_samples: int) -> tuple[str, list[int]]:
    groups = {}
    for idx, (ep_id, p0, _p1) in enumerate(dataset.pairs):
        groups.setdefault(ep_id, []).append((idx, p0.name))

    if not groups:
        raise RuntimeError("dataset has no frame pairs")

    chosen_episode = episode_id if episode_id is not None else next(iter(groups.keys()))
    if chosen_episode not in groups:
        raise RuntimeError(f"episode_id {chosen_episode} not found in split")

    ordered = sorted(groups[chosen_episode], key=lambda x: x[1])
    ordered_indices = [idx for idx, _ in ordered]
    if len(ordered_indices) <= num_samples:
        return chosen_episode, ordered_indices

    pick = np.linspace(0, len(ordered_indices) - 1, num=num_samples, dtype=int)
    return chosen_episode, [ordered_indices[i] for i in pick.tolist()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--split-file", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--out-channels", type=int, default=3)
    parser.add_argument("--pair-step", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--episode-id", type=str, default="")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(Path(args.checkpoint), device)
    dataset = FramePairDataset(
        root=args.dataset_root,
        split_file=args.split_file,
        img_size=args.img_size,
        grayscale=args.grayscale,
        out_channels=args.out_channels,
        pair_step=args.pair_step,
    )

    episode_id = args.episode_id.strip() or None
    chosen_episode, sample_indices = _pick_episode_indices(dataset, episode_id, max(1, args.num_samples))
    rows = []
    col_names = ["I_t", "I_hat_t", "I_t1", "I_hat_t1_from_warp"]

    with torch.no_grad():
        for i in sample_indices:
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
    num_rows = len(rows)
    pad = 6
    header_h = 42
    canvas_h = header_h + num_rows * h + (num_rows + 1) * pad
    canvas_w = len(col_names) * w + (len(col_names) + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad + 2, 4), f"episode: {chosen_episode}", fill=(235, 235, 235))

    for c, name in enumerate(col_names):
        x = pad + c * (w + pad)
        draw.text((x + 2, 22), name, fill=(235, 235, 235))

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
