from typing import Tuple

import numpy as np
from skimage.metrics import structural_similarity as ssim

from src.shared.data.concentration import (
    LOG10_MAX,
    LOG10_MIN,
    SSIM_DATA_RANGE,
    clip_log10_for_ssim,
    physical_to_log10,
)


PLUME_THRESH = 1e-8
PLUME_PAD = 8
PLUME_MIN_PIXELS = 64
LOG10_SSIM_DATA_RANGE = SSIM_DATA_RANGE
PLUME_SSIM_DATA_RANGE = LOG10_SSIM_DATA_RANGE


def compute_global_ssim(
    gt_log: np.ndarray,
    pred_log: np.ndarray,
    data_range: float = LOG10_SSIM_DATA_RANGE,
) -> float:
    gt_view = clip_log10_for_ssim(gt_log, (LOG10_MIN, LOG10_MAX))
    pred_view = clip_log10_for_ssim(pred_log, (LOG10_MIN, LOG10_MAX))
    return float(ssim(gt_view, pred_view, data_range=float(data_range)))


def compute_plume_ssim(
    gt_phys: np.ndarray,
    pred_log: np.ndarray,
    eps_c: float,
    thresh: float = PLUME_THRESH,
    pad: int = PLUME_PAD,
    min_pixels: int = PLUME_MIN_PIXELS,
    data_range: float = PLUME_SSIM_DATA_RANGE,
) -> Tuple[float, int]:
    plume_mask = np.asarray(gt_phys) > float(thresh)
    valid_count = int(plume_mask.sum())
    if valid_count < int(min_pixels):
        return float("nan"), valid_count

    rows, cols = np.where(plume_mask)
    if rows.size == 0 or cols.size == 0:
        return float("nan"), valid_count

    height, width = gt_phys.shape
    r0 = max(0, int(rows.min()) - int(pad))
    r1 = min(height, int(rows.max()) + int(pad) + 1)
    c0 = max(0, int(cols.min()) - int(pad))
    c1 = min(width, int(cols.max()) + int(pad) + 1)

    gt_crop = physical_to_log10(
        gt_phys[r0:r1, c0:c1], eps_c, clip_for_ssim=True
    )
    pred_crop = clip_log10_for_ssim(pred_log[r0:r1, c0:c1])
    return float(ssim(gt_crop, pred_crop, data_range=float(data_range))), valid_count
