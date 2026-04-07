from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class HierarchicalMotionWindowsDataset(Dataset):
    def __init__(
        self,
        root: str,
        split_file: str,
        img_size: int = 64,
        observation_end_fraction: float = 0.5,
        frame_stride: int = 2,
        window_length: int = 8,
        window_hop: int = 4,
        grayscale: bool = True,
        out_channels: int = 3,
        max_episodes: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.img_size = int(img_size)
        self.observation_end_fraction = float(observation_end_fraction)
        self.frame_stride = int(frame_stride)
        self.window_length = int(window_length)
        self.window_hop = int(window_hop)
        self.grayscale = bool(grayscale)
        self.out_channels = int(out_channels)
        self.max_episodes = None if max_episodes is None else int(max_episodes)

        with open(split_file, "r", encoding="utf-8") as f:
            episode_ids = [line.strip() for line in f if line.strip()]

        self.episodes: list[dict] = []
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
            required_cols = ["landing_px", "landing_py", "landing_pz"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise RuntimeError(f"Missing required columns in {csv_path}: {missing}")

            usable = min(len(frame_files), len(df))
            if usable < self.window_length:
                continue

            df = df.iloc[:usable].reset_index(drop=True)
            visible_total = max(1, min(usable, int(np.ceil(usable * self.observation_end_fraction))))
            sampled_indices = list(range(0, visible_total, self.frame_stride))
            if len(sampled_indices) < self.window_length:
                sampled_indices = sampled_indices + [sampled_indices[-1]] * (self.window_length - len(sampled_indices))

            window_indices = self._make_windows(sampled_indices)
            self.episodes.append(
                {
                    "episode_id": episode_id,
                    "frame_files": frame_files[:usable],
                    "landing_xyz": df.loc[0, ["landing_px", "landing_py", "landing_pz"]].to_numpy(dtype=np.float32),
                    "window_indices": window_indices,
                    "sampled_indices": sampled_indices,
                    "visible_total": visible_total,
                    "num_total_frames": usable,
                }
            )

        if self.max_episodes is not None and len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[: self.max_episodes]

    def _make_windows(self, sampled_indices: list[int]) -> list[list[int]]:
        if len(sampled_indices) <= self.window_length:
            base = sampled_indices[: self.window_length]
            if len(base) < self.window_length:
                base = base + [base[-1]] * (self.window_length - len(base))
            return [base]

        starts = list(range(0, len(sampled_indices) - self.window_length + 1, self.window_hop))
        last_start = len(sampled_indices) - self.window_length
        if starts[-1] != last_start:
            starts.append(last_start)
        return [sampled_indices[s : s + self.window_length] for s in starts]

    def __len__(self) -> int:
        return len(self.episodes)

    def _load_frame(self, path: Path) -> torch.Tensor:
        if self.grayscale:
            image = Image.open(path).convert("L")
        else:
            image = Image.open(path).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[..., None]
        if self.grayscale and self.out_channels > 1:
            arr = np.repeat(arr[..., :1], self.out_channels, axis=2)
        elif self.grayscale:
            arr = arr[..., :1]
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> dict:
        episode = self.episodes[idx]
        windows = []
        for frame_ids in episode["window_indices"]:
            frames = torch.stack([self._load_frame(episode["frame_files"][i]) for i in frame_ids], dim=0)
            windows.append(frames)
        window_tensor = torch.stack(windows, dim=0)
        landing_xyz = torch.from_numpy(episode["landing_xyz"])
        landing_xz = torch.tensor([float(landing_xyz[0]), float(landing_xyz[2])], dtype=torch.float32)
        return {
            "episode_id": episode["episode_id"],
            "windows": window_tensor,
            "num_windows": int(window_tensor.shape[0]),
            "landing_xyz": landing_xyz,
            "target_xz": landing_xz,
            "visible_total": int(episode["visible_total"]),
            "num_total_frames": int(episode["num_total_frames"]),
        }


def hierarchical_motion_collate(batch: list[dict]) -> dict:
    max_windows = max(item["num_windows"] for item in batch)
    window_length = batch[0]["windows"].shape[1]
    channels = batch[0]["windows"].shape[2]
    height = batch[0]["windows"].shape[3]
    width = batch[0]["windows"].shape[4]

    padded = torch.zeros(len(batch), max_windows, window_length, channels, height, width, dtype=batch[0]["windows"].dtype)
    num_windows = []
    landing_xyz = []
    target_xz = []
    episode_ids = []
    visible_total = []
    num_total_frames = []

    for i, item in enumerate(batch):
        n = item["num_windows"]
        padded[i, :n] = item["windows"]
        num_windows.append(n)
        landing_xyz.append(item["landing_xyz"])
        target_xz.append(item["target_xz"])
        episode_ids.append(item["episode_id"])
        visible_total.append(item["visible_total"])
        num_total_frames.append(item["num_total_frames"])

    return {
        "windows": padded,
        "num_windows": torch.tensor(num_windows, dtype=torch.long),
        "landing_xyz": torch.stack(landing_xyz, dim=0),
        "target_xz": torch.stack(target_xz, dim=0),
        "episode_id": episode_ids,
        "visible_total": torch.tensor(visible_total, dtype=torch.long),
        "num_total_frames": torch.tensor(num_total_frames, dtype=torch.long),
    }
