"""Full-field concentration-weighted centroid metric used by the manuscript."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def concentration_weighted_centroid(values: np.ndarray) -> Tuple[float, float] | None:
    """Return `(row, column)` using all nonnegative physical concentrations."""
    mass = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    total = float(mass.sum())
    if total <= 0.0:
        return None
    rows, columns = np.indices(mass.shape, dtype=np.float64)
    return float((mass * rows).sum() / total), float((mass * columns).sum() / total)


def compute_concentration_centroid_error(gt_phys: np.ndarray, pred_phys: np.ndarray) -> Tuple[float, int]:
    """Return full-field centroid distance and an eligibility indicator."""
    gt_centroid = concentration_weighted_centroid(gt_phys)
    pred_centroid = concentration_weighted_centroid(pred_phys)
    if gt_centroid is None or pred_centroid is None:
        return float("nan"), 0
    error = float(np.hypot(gt_centroid[0] - pred_centroid[0], gt_centroid[1] - pred_centroid[1]))
    return error, 1


def compute_plume_centroid_error(
    gt_phys: np.ndarray,
    pred_phys: np.ndarray,
    thresh: float | None = None,
    min_pixels: int | None = None,
) -> Tuple[float, int]:
    """Backward-compatible alias for the documented full-field metric.

    `thresh` and `min_pixels` are accepted only for compatibility and do not
    alter the calculation. The manuscript metric is not mask-thresholded.
    """
    del thresh, min_pixels
    return compute_concentration_centroid_error(gt_phys, pred_phys)
