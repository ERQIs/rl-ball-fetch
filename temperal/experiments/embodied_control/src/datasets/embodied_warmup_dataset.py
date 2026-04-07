from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class EmbodiedWarmupDataset(Dataset):
    def __init__(
        self,
        root: str,
        split_file: str,
        img_size: int = 64,
        window_length: int = 8,
        window_stride: int = 4,
        grayscale: bool = True,
        out_channels: int = 3,
        forward_speed_scale: float = 8.0,
        lateral_speed_scale: float = 8.0,
        max_episodes: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.img_size = int(img_size)
        self.window_length = int(window_length)
        self.window_stride = int(window_stride)
        self.grayscale = bool(grayscale)
        self.out_channels = int(out_channels)
        self.forward_speed_scale = float(forward_speed_scale)
        self.lateral_speed_scale = float(lateral_speed_scale)
        self.max_episodes = None if max_episodes is None else int(max_episodes)

        with open(split_file, "r", encoding="utf-8") as f:
            episode_ids = [line.strip() for line in f if line.strip()]

        self.episodes: list[dict] = []
        for episode_id in episode_ids:
            episode = self._load_episode(episode_id)
            if episode is not None:
                self.episodes.append(episode)

        if self.max_episodes is not None and len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[: self.max_episodes]

    def _load_episode(self, episode_id: str) -> dict | None:
        ep_dir = self.root / episode_id
        frames_dir = ep_dir / "frames"
        csv_path = ep_dir / "frames.csv"
        if not frames_dir.exists() or not csv_path.exists():
            return None

        frame_files = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        if not frame_files:
            return None

        df = pd.read_csv(csv_path)
        required_cols = [
            "car_px",
            "car_pz",
            "car_vx",
            "car_vz",
            "landing_px",
            "landing_pz",
            "predicted_t",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing required columns in {csv_path}: {missing}")

        usable = min(len(frame_files), len(df))
        if usable < self.window_length:
            return None

        df = df.iloc[:usable].reset_index(drop=True)
        starts = list(range(0, usable - self.window_length + 1, self.window_stride))
        last_start = usable - self.window_length
        if starts[-1] != last_start:
            starts.append(last_start)

        window_indices = [list(range(start, start + self.window_length)) for start in starts]
        end_indices = np.array([indices[-1] for indices in window_indices], dtype=np.int64)

        sim_time = df["sim_time"].to_numpy(dtype=np.float32) if "sim_time" in df.columns else np.arange(usable, dtype=np.float32) * 0.02
        rel_time = sim_time - float(sim_time[0])
        predicted_t = df["predicted_t"].to_numpy(dtype=np.float32)
        time_to_land = np.maximum(0.0, predicted_t[end_indices] - rel_time[end_indices])

        car_px = df["car_px"].to_numpy(dtype=np.float32)
        car_pz = df["car_pz"].to_numpy(dtype=np.float32)
        car_vx = df["car_vx"].to_numpy(dtype=np.float32)
        car_vz = df["car_vz"].to_numpy(dtype=np.float32)
        landing_px = df["landing_px"].to_numpy(dtype=np.float32)
        landing_pz = df["landing_pz"].to_numpy(dtype=np.float32)

        forward_scale = self.forward_speed_scale if abs(self.forward_speed_scale) > 1e-6 else 1.0
        lateral_scale = self.lateral_speed_scale if abs(self.lateral_speed_scale) > 1e-6 else 1.0

        self_state = np.stack(
            [
                car_vz[end_indices] / forward_scale,
                car_vx[end_indices] / lateral_scale,
            ],
            axis=1,
        ).astype(np.float32)

        landing_offset = np.stack(
            [
                landing_px[end_indices] - car_px[end_indices],
                landing_pz[end_indices] - car_pz[end_indices],
            ],
            axis=1,
        ).astype(np.float32)

        return {
            "episode_id": episode_id,
            "frame_files": frame_files[:usable],
            "window_indices": window_indices,
            "self_state": self_state,
            "landing_offset": landing_offset,
            "time_to_land": time_to_land.astype(np.float32),
            "landing_xz": np.array([landing_px[0], landing_pz[0]], dtype=np.float32),
            "num_total_frames": usable,
        }

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

        windows_tensor = torch.stack(windows, dim=0)
        return {
            "episode_id": episode["episode_id"],
            "windows": windows_tensor,
            "num_windows": int(windows_tensor.shape[0]),
            "self_state": torch.from_numpy(episode["self_state"]),
            "landing_offset": torch.from_numpy(episode["landing_offset"]),
            "time_to_land": torch.from_numpy(episode["time_to_land"]),
            "landing_xz": torch.from_numpy(episode["landing_xz"]),
            "num_total_frames": int(episode["num_total_frames"]),
        }


def embodied_warmup_collate(batch: list[dict]) -> dict:
    max_windows = max(item["num_windows"] for item in batch)
    window_length = batch[0]["windows"].shape[1]
    channels = batch[0]["windows"].shape[2]
    height = batch[0]["windows"].shape[3]
    width = batch[0]["windows"].shape[4]

    padded_windows = torch.zeros(len(batch), max_windows, window_length, channels, height, width, dtype=batch[0]["windows"].dtype)
    padded_self_state = torch.zeros(len(batch), max_windows, batch[0]["self_state"].shape[-1], dtype=batch[0]["self_state"].dtype)
    padded_landing_offset = torch.zeros(
        len(batch), max_windows, batch[0]["landing_offset"].shape[-1], dtype=batch[0]["landing_offset"].dtype
    )
    padded_time_to_land = torch.zeros(len(batch), max_windows, dtype=batch[0]["time_to_land"].dtype)

    num_windows = []
    landing_xz = []
    episode_ids = []
    num_total_frames = []

    for i, item in enumerate(batch):
        n = item["num_windows"]
        padded_windows[i, :n] = item["windows"]
        padded_self_state[i, :n] = item["self_state"]
        padded_landing_offset[i, :n] = item["landing_offset"]
        padded_time_to_land[i, :n] = item["time_to_land"]
        num_windows.append(n)
        landing_xz.append(item["landing_xz"])
        episode_ids.append(item["episode_id"])
        num_total_frames.append(item["num_total_frames"])

    return {
        "windows": padded_windows,
        "self_state": padded_self_state,
        "landing_offset": padded_landing_offset,
        "time_to_land": padded_time_to_land,
        "num_windows": torch.tensor(num_windows, dtype=torch.long),
        "landing_xz": torch.stack(landing_xz, dim=0),
        "episode_id": episode_ids,
        "num_total_frames": torch.tensor(num_total_frames, dtype=torch.long),
    }
