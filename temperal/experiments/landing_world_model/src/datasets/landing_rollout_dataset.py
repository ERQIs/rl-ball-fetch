from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class LandingRolloutDataset(Dataset):
    def __init__(
        self,
        root: str,
        split_file: str,
        img_size: int = 64,
        observation_length: int = 8,
        observation_end_fraction: float = 0.5,
        sampling_mode: str = "uniform_visible",
        grayscale: bool = True,
        out_channels: int = 3,
        max_episodes: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.img_size = int(img_size)
        self.observation_length = int(observation_length)
        self.observation_end_fraction = float(observation_end_fraction)
        self.sampling_mode = str(sampling_mode)
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
            required_cols = [
                "sim_time",
                "ball_px",
                "ball_py",
                "ball_pz",
                "ball_vx",
                "ball_vy",
                "ball_vz",
                "landing_px",
                "landing_py",
                "landing_pz",
            ]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise RuntimeError(f"Missing required columns in {csv_path}: {missing}")

            usable = min(len(frame_files), len(df))
            if usable < self.observation_length:
                continue

            df = df.iloc[:usable].reset_index(drop=True)
            observation_indices = self._sample_observation_indices(usable)
            last_obs_idx = int(observation_indices[-1])
            self.episodes.append(
                {
                    "episode_id": episode_id,
                    "frame_files": frame_files[:usable],
                    "df": df,
                    "observation_indices": observation_indices,
                    "last_obs_idx": last_obs_idx,
                    "landing_xyz": df.loc[0, ["landing_px", "landing_py", "landing_pz"]].to_numpy(dtype=np.float32),
                }
            )

        if self.max_episodes is not None and len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[: self.max_episodes]

    def _sample_observation_indices(self, total_frames: int) -> list[int]:
        visible_total = max(1, min(total_frames, int(np.ceil(total_frames * self.observation_end_fraction))))
        if self.sampling_mode == "uniform_visible":
            if visible_total <= 1:
                return [0] * self.observation_length
            positions = np.linspace(0, visible_total - 1, num=self.observation_length)
            return [min(visible_total - 1, int(round(p))) for p in positions]

        step = max(1, visible_total // max(self.observation_length, 1))
        indices = list(range(0, visible_total, step))[: self.observation_length]
        if len(indices) < self.observation_length:
            indices = indices + [indices[-1]] * (self.observation_length - len(indices))
        return indices

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
        obs_indices = episode["observation_indices"]
        frames = torch.stack([self._load_frame(episode["frame_files"][i]) for i in obs_indices], dim=0)
        df = episode["df"]

        observed_pos = torch.from_numpy(df.loc[obs_indices, ["ball_px", "ball_py", "ball_pz"]].to_numpy(dtype=np.float32))
        observed_vel = torch.from_numpy(df.loc[obs_indices, ["ball_vx", "ball_vy", "ball_vz"]].to_numpy(dtype=np.float32))

        sim_times = df["sim_time"].to_numpy(dtype=np.float32)
        time_delta = float(np.median(np.diff(sim_times))) if len(sim_times) > 1 else 0.02

        return {
            "episode_id": episode["episode_id"],
            "frames": frames,
            "observation_indices": torch.tensor(obs_indices, dtype=torch.long),
            "last_obs_idx": int(episode["last_obs_idx"]),
            "landing_xyz": torch.from_numpy(episode["landing_xyz"]),
            "observed_pos": observed_pos,
            "observed_vel": observed_vel,
            "sim_time_delta": time_delta,
            "num_total_frames": int(len(df)),
        }
