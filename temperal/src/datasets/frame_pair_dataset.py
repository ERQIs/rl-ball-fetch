from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class FramePairDataset(Dataset):
    def __init__(
        self,
        root,
        split_file,
        img_size=64,
        grayscale=True,
        out_channels=3,
        pair_step=1,
        max_pairs=None,
    ):
        self.root = Path(root)
        self.img_size = int(img_size)
        self.grayscale = bool(grayscale)
        self.out_channels = int(out_channels)
        self.pair_step = max(1, int(pair_step))
        self.max_pairs = None if max_pairs is None else int(max_pairs)

        with open(split_file, "r", encoding="utf-8") as f:
            episode_ids = [line.strip() for line in f if line.strip()]

        self.pairs = []
        for episode_id in episode_ids:
            ep_dir = self.root / episode_id
            frames_dir = ep_dir / "frames"
            if not frames_dir.exists():
                continue
            frame_files = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
            if len(frame_files) <= self.pair_step:
                continue
            for idx in range(len(frame_files) - self.pair_step):
                self.pairs.append((episode_id, frame_files[idx], frame_files[idx + self.pair_step]))

        if self.max_pairs is not None and len(self.pairs) > self.max_pairs:
            pick = np.linspace(0, len(self.pairs) - 1, num=self.max_pairs, dtype=int)
            self.pairs = [self.pairs[i] for i in pick.tolist()]

    def __len__(self):
        return len(self.pairs)

    def _load_frame(self, path: Path):
        if self.grayscale:
            image = Image.open(path).convert("L")
        else:
            image = Image.open(path).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[..., None]
        if self.grayscale:
            if self.out_channels == 1:
                arr = arr[..., :1]
            else:
                arr = np.repeat(arr[..., :1], self.out_channels, axis=2)
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        episode_id, p0, p1 = self.pairs[idx]
        i_t = self._load_frame(p0)
        i_t1 = self._load_frame(p1)
        flow_t = torch.zeros(2, self.img_size, self.img_size, dtype=i_t.dtype)
        return {
            "i_t": i_t,
            "i_t1": i_t1,
            "flow_t": flow_t,
            "episode_id": episode_id,
            "frame_t": p0.name,
            "frame_t1": p1.name,
        }
