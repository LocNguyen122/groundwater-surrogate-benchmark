from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from src.shared.data.concentration import physical_to_log10


def normalize_K(K: np.ndarray, stats: Dict[str, float], use_logK: bool, eps_k: float) -> np.ndarray:
    """Normalize a conductivity field using the training-split statistics."""
    values = K.astype(np.float64)
    if use_logK:
        values = np.log(np.clip(values, eps_k, None))
    values = (values - float(stats["k_mean"])) / (float(stats["k_std"]) + 1e-8)
    return values.astype(np.float32)


def log10_C(C: np.ndarray, eps_c: float) -> np.ndarray:
    """Convert nonnegative concentration to the manuscript log scale."""
    return physical_to_log10(C, eps_c, clip_for_ssim=False)


class GroundwaterTimeCondPatchDatasetLogC(Dataset):
    """Deterministic per-index time-conditioned patch dataset.

    Each item contains a normalized conductivity channel, a spatially constant
    normalized-time channel, and the log-concentration target for one sampled
    timestep. Random choices are reproducible from ``seed`` and ``idx``.
    """

    def __init__(
        self,
        files: List[str],
        stats: Dict[str, float],
        patch_size: int = 256,
        use_logK: bool = True,
        eps_k: float = 1e-6,
        eps_c: float = 1e-12,
        seed: int = 0,
        t_index_mode: str = "uniform",
        mid_bias_sigma: float = 0.18,
    ) -> None:
        self.files = files
        self.stats = stats
        self.patch = int(patch_size)
        self.use_logK = bool(use_logK)
        self.eps_k = float(eps_k)
        self.eps_c = float(eps_c)
        self.seed = int(seed)
        self.t_index_mode = str(t_index_mode)
        self.mid_bias_sigma = float(mid_bias_sigma)

    def __len__(self) -> int:
        return len(self.files)

    def _rng_for_index(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + 1_000_003 * idx)

    def _choose_timestep(self, rng: np.random.Generator, n_steps: int) -> int:
        if self.t_index_mode == "uniform":
            return int(rng.integers(0, n_steps))
        if self.t_index_mode == "mid_bias":
            x = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)
            sigma = max(self.mid_bias_sigma, 1e-6)
            probabilities = np.exp(-0.5 * ((x - 0.5) / sigma) ** 2)
            probabilities /= probabilities.sum()
            return int(rng.choice(np.arange(n_steps), p=probabilities))
        raise ValueError(f"Unknown t_index_mode={self.t_index_mode!r}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx]
        rng = self._rng_for_index(idx)
        with np.load(path) as data:
            conductivity = data["K"].astype(np.float32)
            concentration = data["C"].astype(np.float32)
            times = data["times"].astype(np.float32)

        height, width = conductivity.shape
        timestep = self._choose_timestep(rng, concentration.shape[0])
        t_norm = float((times[timestep] - times.min()) / (times.max() - times.min() + 1e-12))

        patch = self.patch
        if height < patch or width < patch:
            raise ValueError(f"Patch {patch} is too large for field {height}x{width} in {path}")
        y0 = int(rng.integers(0, height - patch + 1))
        x0 = int(rng.integers(0, width - patch + 1))

        k_patch = conductivity[y0 : y0 + patch, x0 : x0 + patch]
        c_patch = concentration[timestep, y0 : y0 + patch, x0 : x0 + patch]
        k_norm = normalize_K(k_patch, self.stats, self.use_logK, self.eps_k)
        target = log10_C(c_patch, self.eps_c)
        time_channel = np.full((patch, patch), t_norm, dtype=np.float32)

        return {
            "X": torch.from_numpy(np.stack([k_norm, time_channel], axis=0)),
            "Y_log": torch.from_numpy(target[None, ...]),
            "C_raw": torch.from_numpy(c_patch[None, ...]),
            "t_norm": torch.tensor([t_norm], dtype=torch.float32),
            "t_idx": torch.tensor([timestep], dtype=torch.int64),
        }
