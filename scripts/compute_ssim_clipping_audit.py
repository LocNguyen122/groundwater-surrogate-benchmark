"""Audit focal SSIM values under raw and symmetrically clipped log arrays.

This script requires the withheld simulation fields and prediction caches. It
does not run a model. For each cached seed it reports whole-domain and padded
ground-truth-ROI SSIM with the historical raw arrays and with both target and
prediction clipped to [-12, 2]. The comparison quantifies the sensitivity of
the conclusions to the fixed SSIM dynamic-range convention.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from skimage.metrics import structural_similarity


LOG_MIN = -12.0
LOG_MAX = 2.0
DATA_RANGE = LOG_MAX - LOG_MIN
EPS_C = 1e-12
TAU = 1e-8
PAD = 8
MIN_PIXELS = 64


@dataclass(frozen=True)
class Family:
    name: str
    directory: str
    seeds: tuple[int, ...]


def default_families(pred_root: Path) -> list[Family]:
    return [
        Family("CTA-UNet+FFL TC", str(pred_root / "cta_unet_ffl_transport_conditioned"), tuple(range(10))),
        Family("MS-TMO-UNet TC", str(pred_root / "ms_tmo_transport_conditioned"), tuple(range(5))),
        Family("MS-TMO-UNet TC", str(pred_root / "ms_tmo_transport_conditioned_recovery"), tuple(range(5, 10))),
        Family("CTA-UNet+FFL K-only", str(pred_root / "cta_unet_ffl_konly"), tuple(range(1, 10))),
    ]


def _bbox(mask: np.ndarray) -> tuple[slice, slice] | None:
    ys, xs = np.nonzero(mask)
    if len(ys) < MIN_PIXELS:
        return None
    y0 = max(0, int(ys.min()) - PAD)
    y1 = min(mask.shape[0], int(ys.max()) + PAD + 1)
    x0 = max(0, int(xs.min()) - PAD)
    x1 = min(mask.shape[1], int(xs.max()) + PAD + 1)
    return slice(y0, y1), slice(x0, x1)


def _one_file(task: tuple[str, str]) -> dict[str, float | int]:
    pred_path, data_root = task
    with np.load(pred_path) as cache:
        pred = cache["pred_log"].astype(np.float32)
        source = os.path.basename(str(cache["source_file"]))
    stem = os.path.basename(pred_path).replace(".pred.npz", "")
    param_folder = stem.split("__", 1)[0]
    with np.load(os.path.join(data_root, param_folder, source)) as simulation:
        concentration = simulation["C"].astype(np.float32)
    gt = np.log10(np.clip(concentration, 0.0, None) + EPS_C)

    sums = {
        "global_raw_sum": 0.0,
        "global_clipped_sum": 0.0,
        "gt_roi_raw_sum": 0.0,
        "gt_roi_clipped_sum": 0.0,
        "n_global": 0,
        "n_gt_roi": 0,
        "target_below": int(np.count_nonzero(gt < LOG_MIN)),
        "target_above": int(np.count_nonzero(gt > LOG_MAX)),
        "prediction_below": int(np.count_nonzero(pred < LOG_MIN)),
        "prediction_above": int(np.count_nonzero(pred > LOG_MAX)),
        "n_values": int(gt.size),
    }
    for timestep in range(gt.shape[0]):
        gt_t = gt[timestep]
        pred_t = pred[timestep]
        gt_clip = np.clip(gt_t, LOG_MIN, LOG_MAX)
        pred_clip = np.clip(pred_t, LOG_MIN, LOG_MAX)
        sums["global_raw_sum"] += float(structural_similarity(gt_t, pred_t, data_range=DATA_RANGE))
        sums["global_clipped_sum"] += float(structural_similarity(gt_clip, pred_clip, data_range=DATA_RANGE))
        sums["n_global"] += 1

        roi = _bbox(concentration[timestep] > TAU)
        if roi is None:
            continue
        sums["gt_roi_raw_sum"] += float(structural_similarity(gt_t[roi], pred_t[roi], data_range=DATA_RANGE))
        sums["gt_roi_clipped_sum"] += float(structural_similarity(gt_clip[roi], pred_clip[roi], data_range=DATA_RANGE))
        sums["n_gt_roi"] += 1
    return sums


def _sum_dicts(parts: Iterable[dict[str, float | int]]) -> dict[str, float]:
    total: dict[str, float] = {}
    for part in parts:
        for key, value in part.items():
            total[key] = total.get(key, 0.0) + float(value)
    return total


def audit_seed(family: Family, seed: int, data_root: Path, workers: int) -> dict[str, object]:
    seed_dir = Path(family.directory) / f"seed{seed}"
    paths = sorted(glob.glob(str(seed_dir / "*.pred.npz")))
    if len(paths) != 170:
        raise FileNotFoundError(f"Expected 170 cache files in {seed_dir}, found {len(paths)}")
    tasks = [(path, str(data_root)) for path in paths]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        total = _sum_dicts(pool.map(_one_file, tasks, chunksize=2))
    n_global = int(total["n_global"])
    n_roi = int(total["n_gt_roi"])
    return {
        "family": family.name,
        "seed": seed,
        "n_files": len(paths),
        "n_global_pairs": n_global,
        "n_gt_roi_pairs": n_roi,
        "roi_pad_pixels": PAD,
        "roi_min_pixels": MIN_PIXELS,
        "global_ssim_raw": total["global_raw_sum"] / n_global,
        "global_ssim_clipped": total["global_clipped_sum"] / n_global,
        "global_ssim_delta": (total["global_clipped_sum"] - total["global_raw_sum"]) / n_global,
        "gt_roi_ssim_raw": total["gt_roi_raw_sum"] / n_roi,
        "gt_roi_ssim_clipped": total["gt_roi_clipped_sum"] / n_roi,
        "gt_roi_ssim_delta": (total["gt_roi_clipped_sum"] - total["gt_roi_raw_sum"]) / n_roi,
        "target_fraction_below": total["target_below"] / total["n_values"],
        "target_fraction_above": total["target_above"] / total["n_values"],
        "prediction_fraction_below": total["prediction_below"] / total["n_values"],
        "prediction_fraction_above": total["prediction_above"] / total["n_values"],
    }


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    """Write family-level means and sample standard deviations."""
    metrics = (
        "global_ssim_raw",
        "global_ssim_clipped",
        "global_ssim_delta",
        "gt_roi_ssim_raw",
        "gt_roi_ssim_clipped",
        "gt_roi_ssim_delta",
        "target_fraction_below",
        "target_fraction_above",
        "prediction_fraction_below",
        "prediction_fraction_above",
    )
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["family"]), []).append(row)
    summary_rows: list[dict[str, object]] = []
    for family, family_rows in grouped.items():
        summary: dict[str, object] = {"family": family, "n_seeds": len(family_rows)}
        for metric in metrics:
            values = [float(row[metric]) for row in family_rows]
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--summary-out",
        type=Path,
        help="Optional family-level summary CSV (default: *_summary.csv next to --out).",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--family", action="append", help="Optional exact family label; may be repeated.")
    parser.add_argument("--seed", type=int, action="append", help="Optional seed restriction; may be repeated.")
    args = parser.parse_args()

    families = default_families(args.pred_root)
    if args.family:
        wanted = set(args.family)
        families = [family for family in families if family.name in wanted]
    seed_filter = set(args.seed or [])
    rows: list[dict[str, object]] = []
    for family in families:
        for seed in family.seeds:
            if seed_filter and seed not in seed_filter:
                continue
            print(f"Auditing {family.name}, seed {seed}", flush=True)
            rows.append(audit_seed(family, seed, args.data_root, max(1, args.workers)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")
    summary_out = args.summary_out or args.out.with_name(args.out.stem.replace("_per_seed", "_summary") + args.out.suffix)
    write_summary(rows, summary_out)
    print(f"Wrote family summary to {summary_out}")


if __name__ == "__main__":
    main()
