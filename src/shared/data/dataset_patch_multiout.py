from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.shared.data.concentration import physical_to_log10


class GroundwaterPatchDatasetMultiOut(Dataset):
    """Return a conductivity patch and all requested concentration timesteps."""

    def __init__(
        self,
        file_list,
        stats,
        patch_size=320,
        use_logK=True,
        eps_k=1e-6,
        eps_c=1e-12,
        timesteps: Optional[Sequence[int]] = None,
        crop_mode: str = "random",
        fixed_crop_seed: int = 0,
        require_source_in_crop: bool = False,
        source_row: int = 399,
        source_col: int = 199,
        crop_key: str = "absolute_legacy",
    ):
        if crop_key not in {"absolute_legacy", "relative"}:
            raise ValueError("crop_key must be absolute_legacy or relative")
        self.crop_key = crop_key
        self.files = list(file_list)
        self.stats = stats
        self.patch = int(patch_size)
        self.use_logK = bool(use_logK)
        self.eps_k = float(eps_k)
        self.eps_c = float(eps_c)
        self.timesteps = self._normalize_timesteps(timesteps)
        if crop_mode not in {"random", "fixed", "full"}:
            raise ValueError("crop_mode must be one of: random, fixed, full")
        self.crop_mode = crop_mode
        self.fixed_crop_seed = int(fixed_crop_seed)
        self.require_source_in_crop = bool(require_source_in_crop)
        self.source_row = int(source_row)
        self.source_col = int(source_col)

        self.k_mean = float(stats["k_mean"])
        self.k_std = float(stats["k_std"])

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _normalize_timesteps(timesteps: Optional[Sequence[int]]) -> Optional[np.ndarray]:
        if timesteps is None:
            return None
        ts = np.array(list(timesteps), dtype=np.int64)
        if ts.ndim != 1 or ts.size == 0:
            raise ValueError("timesteps must be a non-empty 1D sequence of indices.")
        if np.any(ts < 0):
            raise ValueError("timesteps must contain non-negative indices.")
        return ts

    def _normalize_K(self, conductivity):
        values = conductivity.astype(np.float64)
        if self.use_logK:
            values = np.log(np.clip(values, self.eps_k, None))
        values = (values - self.k_mean) / (self.k_std + 1e-8)
        return values.astype(np.float32)

    def _logC(self, concentration):
        return physical_to_log10(concentration, self.eps_c, clip_for_ssim=False)

    def _fixed_crop_origin(self, path: str, height: int, width: int) -> tuple[int, int]:
        """Choose a stable crop without Python's process-randomized hash."""
        max_y = max(height - self.patch, 0)
        max_x = max(width - self.patch, 0)
        key = (Path(path).as_posix() if self.crop_key == "absolute_legacy"
               else f"{Path(path).parent.name}/{Path(path).name}")
        digest = hashlib.sha256(f"{key}|{self.fixed_crop_seed}".encode("utf-8")).digest()
        y0 = int.from_bytes(digest[:8], "little") % (max_y + 1)
        x0 = int.from_bytes(digest[8:16], "little") % (max_x + 1)

        if self.require_source_in_crop:
            y_low = max(0, self.source_row - self.patch + 1)
            y_high = min(self.source_row, max_y)
            x_low = max(0, self.source_col - self.patch + 1)
            x_high = min(self.source_col, max_x)
            if y_low > y_high or x_low > x_high:
                raise ValueError(
                    "Requested source-containing crop is impossible for "
                    f"shape={(height, width)}, patch={self.patch}, "
                    f"source={(self.source_row, self.source_col)}"
                )
            y0 = y_low + (y0 % (y_high - y_low + 1))
            x0 = x_low + (x0 % (x_high - x_low + 1))
        return y0, x0

    def __getitem__(self, idx):
        path = self.files[idx]
        with np.load(path) as data:
            conductivity = data["K"].astype(np.float32)
            concentration = data["C"].astype(np.float32)

        conductivity = self._normalize_K(conductivity)
        concentration_log = self._logC(concentration)

        if self.timesteps is not None:
            max_t = int(self.timesteps.max())
            if max_t >= concentration_log.shape[0]:
                raise IndexError(
                    f"Requested timestep index {max_t} exceeds available range "
                    f"[0, {concentration_log.shape[0] - 1}] in file: {path}"
                )
            concentration_log = concentration_log[self.timesteps]

        height, width = conductivity.shape
        patch = self.patch
        if self.crop_mode == "full":
            y0 = x0 = 0
            patch_h, patch_w = height, width
        elif self.crop_mode == "fixed":
            y0, x0 = self._fixed_crop_origin(path, height, width)
            patch_h = patch_w = patch
        else:
            y0 = np.random.randint(0, height - patch + 1) if height > patch else 0
            x0 = np.random.randint(0, width - patch + 1) if width > patch else 0
            patch_h = patch_w = patch

        conductivity_patch = conductivity[y0 : y0 + patch_h, x0 : x0 + patch_w]
        concentration_patch = concentration_log[:, y0 : y0 + patch_h, x0 : x0 + patch_w]

        return {
            "K": torch.from_numpy(conductivity_patch[None, ...]),
            "C_log": torch.from_numpy(concentration_patch),
            "path": str(path),
            "crop_y0": int(y0),
            "crop_x0": int(x0),
        }
