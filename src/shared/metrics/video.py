"""Temporal and plume-specific metrics for concentration sequences.

All functions operate on NumPy arrays. Batch and temporal dimensions are
explicit, so callers should loop over seeds or timesteps as needed.

Metric design notes:
- Global SSIM and plume SSIM delegate to the plume-SSIM utility.
- Optional feature-distribution metrics require an explicitly documented feature extractor.
- Physical-unit centroid errors require a pixel-size argument.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "compute_temporal_mask_iou",
    "compute_dice",
    "compute_boundary_f_score",
    "compute_centroid_error_px",
    "compute_plume_area_error",
    "compute_brier_score",
    "compute_ece",
    "compute_coverage_width",
    "compute_uncertainty_error_correlation",
    "VideoMetrics",
    "aggregate_video_metrics",
]

# -- binary mask helpers --------------------------------------------------------

def _as_bool(x: np.ndarray) -> np.ndarray:
    return x.astype(bool)


def _intersection_union(
    pred_mask: np.ndarray, gt_mask: np.ndarray
) -> Tuple[int, int]:
    p, g = _as_bool(pred_mask), _as_bool(gt_mask)
    return int((p & g).sum()), int((p | g).sum())


# -- frame-level metrics --------------------------------------------------------

def compute_temporal_mask_iou(
    pred_masks: np.ndarray,    # (T, H, W) binary float
    gt_masks: np.ndarray,      # (T, H, W) binary float
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Compute per-frame and mean mask IoU over T future frames."""
    T = pred_masks.shape[0]
    per_frame = []
    for t in range(T):
        inter, union = _intersection_union(pred_masks[t], gt_masks[t])
        per_frame.append(inter / (union + eps))
    return {"mean_iou": float(np.mean(per_frame)), "per_frame_iou": per_frame}


def compute_dice(
    pred_masks: np.ndarray,    # (T, H, W)
    gt_masks: np.ndarray,      # (T, H, W)
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Dice coefficient averaged over T frames."""
    T = pred_masks.shape[0]
    per_frame = []
    for t in range(T):
        p, g = _as_bool(pred_masks[t]), _as_bool(gt_masks[t])
        inter = int((p & g).sum())
        denom = int(p.sum()) + int(g.sum())
        per_frame.append(2 * inter / (denom + eps))
    return {"mean_dice": float(np.mean(per_frame)), "per_frame_dice": per_frame}


def compute_boundary_f_score(
    pred_masks: np.ndarray,    # (T, H, W)
    gt_masks: np.ndarray,      # (T, H, W)
    dilation: int = 3,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Boundary F-score (and simplified Hausdorff-lite distance) over T frames.

    Uses binary dilation of the boundary for tolerance matching.
    Hausdorff-lite is the max of directed Hausdorff distances.
    """
    from scipy.ndimage import binary_dilation, distance_transform_edt

    T = pred_masks.shape[0]
    f_scores, hausdorff = [], []

    for t in range(T):
        p_b = _boundary(pred_masks[t])
        g_b = _boundary(gt_masks[t])

        p_dilated = binary_dilation(p_b, iterations=dilation)
        g_dilated = binary_dilation(g_b, iterations=dilation)

        tp = int((g_b & p_dilated).sum())
        fp = int((p_b & ~g_dilated).sum())
        fn = int((g_b & ~p_dilated).sum())
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f = 2 * prec * rec / (prec + rec + eps)
        f_scores.append(f)

        # Hausdorff-lite via distance transform
        if p_b.any() and g_b.any():
            dist_p = distance_transform_edt(~p_b)
            dist_g = distance_transform_edt(~g_b)
            h = max(dist_p[g_b].max(), dist_g[p_b].max())
        else:
            h = float("nan")
        hausdorff.append(float(h))

    return {
        "mean_boundary_f": float(np.nanmean(f_scores)),
        "mean_hausdorff": float(np.nanmean(hausdorff)),
        "per_frame_f": f_scores,
    }


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Extract binary boundary pixels (4-connectivity erosion difference)."""
    from scipy.ndimage import binary_erosion
    b = _as_bool(mask)
    return b & ~binary_erosion(b)


def compute_centroid_error_px(
    pred_masks: np.ndarray,    # (T, H, W)
    gt_masks: np.ndarray,      # (T, H, W)
    pixel_size_m: Optional[float] = None,
) -> Dict[str, float]:
    """Centroid L2 error in pixels (and optionally meters) per frame."""
    T = pred_masks.shape[0]
    per_frame_px = []
    for t in range(T):
        c_pred = _centroid(pred_masks[t])
        c_gt = _centroid(gt_masks[t])
        if c_pred is None or c_gt is None:
            per_frame_px.append(float("nan"))
        else:
            per_frame_px.append(float(np.linalg.norm(np.array(c_pred) - np.array(c_gt))))
    out = {"mean_centroid_err_px": float(np.nanmean(per_frame_px)),
           "per_frame_centroid_err_px": per_frame_px}
    if pixel_size_m is not None:
        out["mean_centroid_err_m"] = out["mean_centroid_err_px"] * pixel_size_m
    return out


def _centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    b = _as_bool(mask)
    if not b.any():
        return None
    rows, cols = np.where(b)
    return float(rows.mean()), float(cols.mean())


def compute_plume_area_error(
    pred_masks: np.ndarray,    # (T, H, W)
    gt_masks: np.ndarray,      # (T, H, W)
) -> Dict[str, float]:
    """Signed and absolute plume area error (pixel count) per frame."""
    T = pred_masks.shape[0]
    signed, absolute = [], []
    for t in range(T):
        a_pred = int(_as_bool(pred_masks[t]).sum())
        a_gt = int(_as_bool(gt_masks[t]).sum())
        signed.append(float(a_pred - a_gt))
        absolute.append(float(abs(a_pred - a_gt)))
    return {
        "mean_area_err": float(np.mean(signed)),
        "mean_abs_area_err": float(np.mean(absolute)),
    }


# -- probabilistic metrics ------------------------------------------------------

def compute_brier_score(
    prob_map: np.ndarray,   # (T, H, W) predicted probability in [0,1]
    gt_binary: np.ndarray,  # (T, H, W) ground-truth binary {0,1}
) -> float:
    """Mean Brier score over all pixels and timesteps."""
    return float(np.mean((prob_map - gt_binary.astype(float)) ** 2))


def compute_ece(
    prob_map: np.ndarray,   # (N,) or (T*H*W,) predicted probabilities
    gt_binary: np.ndarray,  # (N,) ground-truth labels
    n_bins: int = 10,
) -> float:
    """Expected calibration error (pixel-wise, uniform-mass bins)."""
    probs = prob_map.ravel()
    labels = gt_binary.ravel().astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(probs)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        idx = (probs >= lo) & (probs < hi)
        if not idx.any():
            continue
        mean_conf = probs[idx].mean()
        mean_acc = labels[idx].mean()
        ece += idx.sum() * abs(mean_conf - mean_acc)
    return float(ece / max(n_total, 1))


def compute_coverage_width(
    lower: np.ndarray,   # (T, H, W) lower bound of prediction interval
    upper: np.ndarray,   # (T, H, W) upper bound
    gt: np.ndarray,      # (T, H, W) ground-truth
) -> Dict[str, float]:
    """Empirical coverage and mean interval width over all pixels/timesteps."""
    covered = ((gt >= lower) & (gt <= upper)).astype(float)
    width = (upper - lower).astype(float)
    return {
        "empirical_coverage": float(covered.mean()),
        "mean_width": float(width.mean()),
    }


def compute_uncertainty_error_correlation(
    uncertainty: np.ndarray,   # (T, H, W) or (N,) uncertainty estimate
    error: np.ndarray,         # (T, H, W) or (N,) abs prediction error
) -> float:
    """Pearson correlation between pixel-wise uncertainty and absolute error."""
    u = uncertainty.ravel().astype(float)
    e = error.ravel().astype(float)
    if len(u) < 2:
        return float("nan")
    return float(np.corrcoef(u, e)[0, 1])


# -- aggregate container --------------------------------------------------------

@dataclass
class VideoMetrics:
    """Container for all metrics for a single predicted clip."""

    mean_iou: float = float("nan")
    mean_dice: float = float("nan")
    mean_boundary_f: float = float("nan")
    mean_hausdorff: float = float("nan")
    mean_centroid_err_px: float = float("nan")
    mean_area_err: float = float("nan")
    brier_mask: float = float("nan")
    brier_exceedance: float = float("nan")
    ece: float = float("nan")
    empirical_coverage: float = float("nan")
    mean_width: float = float("nan")
    unc_err_corr: float = float("nan")
    global_ssim: float = float("nan")
    plume_ssim: float = float("nan")
    per_frame_iou: List[float] = field(default_factory=list)


def aggregate_video_metrics(metrics: Sequence[VideoMetrics]) -> Dict[str, float]:
    """Compute mean ± std across a list of VideoMetrics (e.g., multiple seeds)."""
    keys = [
        "mean_iou", "mean_dice", "mean_boundary_f", "mean_hausdorff",
        "mean_centroid_err_px", "brier_mask", "brier_exceedance",
        "ece", "empirical_coverage", "mean_width", "unc_err_corr",
        "global_ssim", "plume_ssim",
    ]
    result = {}
    for k in keys:
        vals = [getattr(m, k) for m in metrics if not np.isnan(getattr(m, k))]
        if vals:
            result[f"{k}_mean"] = float(np.mean(vals))
            result[f"{k}_std"] = float(np.std(vals))
        else:
            result[f"{k}_mean"] = float("nan")
            result[f"{k}_std"] = float("nan")
    return result
