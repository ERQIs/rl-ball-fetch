from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class TemporalClipDataset(Dataset):
    def __init__(
        self,
        root,
        split_file,
        seq_len=12,
        frame_stride=2,
        clip_start_stride=6,
        img_size=64,
        grayscale=True,
        out_channels=3,
        pos_cols=None,
        vel_cols=None,
        max_clips=None,
    ):
        self.root = Path(root)
        self.seq_len = int(seq_len)
        self.frame_stride = int(frame_stride)
        self.clip_start_stride = int(clip_start_stride)
        self.img_size = int(img_size)
        self.grayscale = bool(grayscale)
        self.out_channels = int(out_channels)
        self.pos_cols = pos_cols or ["ball_px", "ball_py", "ball_pz"]
        self.vel_cols = vel_cols or ["ball_vx", "ball_vy", "ball_vz"]
        self.max_clips = None if max_clips is None else int(max_clips)

        with open(split_file, "r", encoding="utf-8") as f:
            episode_ids = [line.strip() for line in f if line.strip()]

        self.episodes = []
        self.clips = []
        clip_span = (self.seq_len - 1) * self.frame_stride + 1

        for episode_id in episode_ids:
            ep_dir = self.root / episode_id
            frames_dir = ep_dir / "frames"
            csv_path = ep_dir / "frames.csv"
            if not frames_dir.exists() or not csv_path.exists():
                continue

            frame_files = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
            if not frame_files:
                continue

            df = pd.read_csv(csv_path)
            missing_cols = [c for c in self.pos_cols + self.vel_cols if c not in df.columns]
            if missing_cols:
                raise RuntimeError(f"Missing columns in {csv_path}: {missing_cols}")

            usable = min(len(frame_files), len(df))
            if usable < clip_span:
                continue

            episode = {
                "episode_id": episode_id,
                "frame_files": frame_files[:usable],
                "pos": df[self.pos_cols].to_numpy(dtype=np.float32)[:usable],
                "vel": df[self.vel_cols].to_numpy(dtype=np.float32)[:usable],
            }
            ep_idx = len(self.episodes)
            self.episodes.append(episode)

            max_start = usable - clip_span
            for start in range(0, max_start + 1, self.clip_start_stride):
                self.clips.append((ep_idx, start))

        if self.max_clips is not None and len(self.clips) > self.max_clips:
            pick = np.linspace(0, len(self.clips) - 1, num=self.max_clips, dtype=int)
            self.clips = [self.clips[i] for i in pick.tolist()]

    def __len__(self):
        return len(self.clips)

    def _load_frame(self, path):
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
        ep_idx, start = self.clips[idx]
        episode = self.episodes[ep_idx]
        indices = [start + i * self.frame_stride for i in range(self.seq_len)]
        frames = torch.stack([self._load_frame(episode["frame_files"][i]) for i in indices], dim=0)
        pos_seq = torch.from_numpy(episode["pos"][indices])
        vel_seq = torch.from_numpy(episode["vel"][indices])

        return {
            "frames": frames,
            "pos_seq": pos_seq,
            "vel_seq": vel_seq,
            "episode_id": episode["episode_id"],
            "frame_indices": indices,
        }
