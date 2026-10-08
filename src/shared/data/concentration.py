"""Shared concentration transforms for training and evaluation.

Generic helpers: ``physical_to_log10`` computes ``log10(max(C, 0) + eps)`` (optionally clipped for SSIM) and
``log10_to_physical`` inverts it. Historical (pre-v5) trainers used the unclipped transform as target.

Confirmatory protocol (v5 onwards, the manuscript): targets are clipped to [LOG10_MIN, LOG10_MAX] = [-12, 2],
the network output is mapped into the same interval by ``bounded_log10_tensor`` (a sigmoid), SSIM uses that
bounded view with data_range 14, and physical metrics use the inverse of the bounded prediction. The canonical
evaluation is ``src.shared.eval.eval_transport_unified_logc_global_plumemask``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


EPS_C = 1e-12
LOG10_MIN = -12.0
LOG10_MAX = 2.0
SSIM_DATA_RANGE = LOG10_MAX - LOG10_MIN


def physical_to_log10(
    concentration: np.ndarray,
    eps_c: float = EPS_C,
    *,
    clip_for_ssim: bool = False,
    bounds: Tuple[float, float] = (LOG10_MIN, LOG10_MAX),
) -> np.ndarray:
    """Convert physical concentration to log10 space with explicit clipping."""
    values = np.log10(np.clip(np.asarray(concentration), 0.0, None) + float(eps_c))
    if clip_for_ssim:
        values = np.clip(values, float(bounds[0]), float(bounds[1]))
    return values.astype(np.float32)


def clip_log10_for_ssim(
    values: np.ndarray,
    bounds: Tuple[float, float] = (LOG10_MIN, LOG10_MAX),
) -> np.ndarray:
    """Return the symmetric, fixed-range SSIM view of log10 values."""
    return np.clip(np.asarray(values), float(bounds[0]), float(bounds[1])).astype(np.float32)


def log10_to_physical(values: np.ndarray, eps_c: float = EPS_C) -> np.ndarray:
    """Invert finite log10 values to nonnegative physical units."""
    return np.clip(np.power(10.0, np.asarray(values)) - float(eps_c), 0.0, None).astype(np.float32)


def bounded_log10_tensor(
    raw_values: torch.Tensor,
    bounds: Tuple[float, float] = (LOG10_MIN, LOG10_MAX),
) -> torch.Tensor:
    """Map raw model outputs smoothly into the declared log-concentration range."""
    lower, upper = float(bounds[0]), float(bounds[1])
    if not lower < upper:
        raise ValueError(f"Invalid log10 bounds: {bounds}")
    return lower + (upper - lower) * torch.sigmoid(raw_values)


def clip_log10_tensor_for_loss(
    values: torch.Tensor,
    bounds: Tuple[float, float] = (LOG10_MIN, LOG10_MAX),
) -> torch.Tensor:
    """Apply the same declared bounds to a tensor used by a bounded loss."""
    return torch.clamp(values, min=float(bounds[0]), max=float(bounds[1]))


def clipping_counts(
    values: np.ndarray,
    bounds: Tuple[float, float] = (LOG10_MIN, LOG10_MAX),
) -> dict[str, int]:
    """Count finite values below and above the declared SSIM bounds."""
    array = np.asarray(values)
    finite = np.isfinite(array)
    return {
        "finite": int(finite.sum()),
        "below": int(np.logical_and(finite, array < float(bounds[0])).sum()),
        "above": int(np.logical_and(finite, array > float(bounds[1])).sum()),
    }
