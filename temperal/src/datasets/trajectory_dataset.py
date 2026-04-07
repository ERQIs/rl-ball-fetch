import os
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch


class TrajectoryDataset(Dataset):
    """Dataset for episodes exported from Unity.

    Expected folder structure per episode:
      episode_xxx/
        frames/
          000000.png
          000001.png
        trajectory.csv

    The dataset returns a dict with keys: frames (T,1,H,W), target_xy (2,), episode_id, frame_indices
    """

    def __init__(self, root, split_file=None, img_size=64, observation_length=8, frame_stride=1,
                 use_last_n_frames=True, transforms=None, trajectory_csv_name="trajectory.csv",
                 observation_end_fraction=1.0, sampling_mode="tail"):
        self.root = Path(root)
        self.episodes = []
        if split_file:
            with open(split_file, 'r') as f:
                ids = [l.strip() for l in f if l.strip()]
            for eid in ids:
                ep = self.root / eid
                if ep.exists():
                    self.episodes.append(ep)
        else:
            for p in sorted(self.root.iterdir()):
                if p.is_dir():
                    self.episodes.append(p)

        self.img_size = img_size
        self.observation_length = observation_length
        self.frame_stride = frame_stride
        self.use_last_n_frames = use_last_n_frames
        self.observation_end_fraction = float(observation_end_fraction)
        self.sampling_mode = sampling_mode
        self.transforms = transforms or transforms_fn(img_size)
        self.trajectory_csv_name = trajectory_csv_name

        # build index of episodes -> frame file list
        self.episode_index = []
        for ep in self.episodes:
            frames_dir = ep / "frames"
            if not frames_dir.exists():
                continue
            files = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg']])
            if len(files) == 0:
                continue
            self.episode_index.append({
                'ep': ep,
                'frames': files,
                'traj': ep / self.trajectory_csv_name
            })

    def __len__(self):
        return len(self.episode_index)

    def _read_landing_point(self, traj_path: Path):
        # try to read csv with pandas and infer landing point as last row ball position
        if not traj_path.exists():
            raise FileNotFoundError(f"trajectory csv not found: {traj_path}")
        df = pd.read_csv(traj_path)
        # heuristics: find columns with ball and x,y
        cols = df.columns.str.lower()
        xcol = None
        ycol = None
        for c in cols:
            if 'ball' in c and ('_x' in c or 'x' == c or '.x' in c or 'pos_x' in c or 'position_x' in c or 'px' in c):
                xcol = c
            if 'ball' in c and ('_y' in c or 'y' == c or '.y' in c or 'pos_y' in c or 'position_y' in c or 'py' in c):
                ycol = c
        # fallback: look for any two x/y columns
        if xcol is None or ycol is None:
            for c in cols:
                if c.endswith('_x') or c.endswith('.x') or c.endswith('x'):
                    if xcol is None:
                        xcol = c
                if c.endswith('_y') or c.endswith('.y') or c.endswith('y'):
                    if ycol is None:
                        ycol = c

        if xcol is None or ycol is None:
            # as last resort, try columns named x,y
            if 'x' in cols and 'y' in cols:
                xcol = 'x'; ycol = 'y'

        if xcol is None or ycol is None:
            raise RuntimeError(f"Could not find ball x/y columns in {traj_path}. Columns: {list(df.columns)}")

        last = df.iloc[-1]
        return float(last[xcol]), float(last[ycol])

    def __getitem__(self, idx):
        info = self.episode_index[idx]
        ep = info['ep']
        frames = info['frames']
        traj = info['traj']
        if not traj.exists():
            # try to find any csv under the episode folder as fallback
            cands = list(ep.glob('*.csv')) + list(ep.glob('**/*.csv'))
            if len(cands) > 0:
                traj = cands[0]
            else:
                raise FileNotFoundError(f"trajectory csv not found: {info['traj']}")

        total = len(frames)
        stride = max(1, int(self.frame_stride))
        L = self.observation_length
        visible_total = max(1, min(total, int(np.ceil(total * self.observation_end_fraction))))

        if self.sampling_mode == "uniform_visible":
            end = visible_total
            if end <= 1:
                indices = [0] * L
            else:
                positions = np.linspace(0, end - 1, num=L)
                indices = [min(end - 1, int(round(p))) for p in positions]
        elif self.use_last_n_frames:
            # choose the last L*stride frames from the visible prefix
            end = visible_total
            start = max(0, end - L * stride)
            indices = list(range(start, end, stride))[-L:]
        else:
            # choose the first L*stride frames from the visible prefix
            indices = list(range(0, min(visible_total, L * stride), stride))[:L]

        if len(indices) < L:
            pad = [indices[-1]] * (L - len(indices)) if indices else [0] * L
            indices = indices + pad

        imgs = []
        timestamps = []
        for i in indices:
            p = frames[i]
            im = Image.open(p).convert('L')
            im = self.transforms(im)
            imgs.append(im)
            # optional: try to parse timestamp from filename or csv later
            timestamps.append(0.0)

        frames_tensor = torch.stack(imgs)  # (T, C, H, W)

        target_x, target_y = self._read_landing_point(traj)
        target = torch.tensor([target_x, target_y], dtype=torch.float32)

        return {
            'frames': frames_tensor,
            'target_xy': target,
            'episode_id': str(ep.name),
            'frame_indices': indices,
            'timestamps': np.array(timestamps),
            'meta': {'traj_path': str(traj)}
        }


def transforms_fn(img_size=64):
    def _fn(pil_img):
        img = pil_img.resize((img_size, img_size))
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[None, ...]
        else:
            arr = arr.mean(axis=2)[None, ...]
        arr = (arr - 0.5) / 0.5
        return torch.from_numpy(arr)
    return _fn
