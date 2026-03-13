from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Pillow is required for loading PNG frames.") from exc


def _load_image_tensor(path: Path, input_channels: int) -> torch.Tensor:
    if input_channels == 1:
        img = Image.open(path).convert("L")
    elif input_channels == 3:
        img = Image.open(path).convert("RGB")
    else:
        raise ValueError("input_channels must be 1 or 3")
    x = torch.from_numpy(np.array(img)).float() / 255.0
    if input_channels == 1:
        return x.unsqueeze(0)
    return x.permute(2, 0, 1)


class ZeroFlowProvider:
    def __call__(self, i_t: torch.Tensor, i_t1: torch.Tensor) -> torch.Tensor:
        _, h, w = i_t.shape
        return torch.zeros(2, h, w, dtype=i_t.dtype)


class TrajectoryPairDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        flow_provider: Callable | None = None,
        input_channels: int = 1,
    ) -> None:
        self.root = Path(root_dir)
        self.flow_provider = flow_provider or ZeroFlowProvider()
        self.input_channels = input_channels
        self.pairs: list[tuple[Path, Path]] = []
        self._build_index()
        if not self.pairs:
            raise RuntimeError(f"No frame pairs found under {self.root}")

    def _build_index(self) -> None:
        for traj in sorted(self.root.glob("traj_*")):
            frames = sorted((traj / "frames").glob("*.png"))
            for i in range(len(frames) - 1):
                self.pairs.append((frames[i], frames[i + 1]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        p0, p1 = self.pairs[idx]
        i_t = _load_image_tensor(p0, self.input_channels)
        i_t1 = _load_image_tensor(p1, self.input_channels)
        flow_t = self.flow_provider(i_t, i_t1)
        return {"i_t": i_t, "i_t1": i_t1, "flow_t": flow_t}
