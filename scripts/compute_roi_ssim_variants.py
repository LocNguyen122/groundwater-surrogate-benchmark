"""Recompute GT-, predicted-, and union-ROI SSIM from external caches.

Without external artifacts, the script validates the included precomputed CSVs.
With `--data-root` and `--pred-root`, it recomputes all focal seeds using an
8-pixel padded bounding box and a 64-pixel minimum defining mask.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[1]
EPS_C = 1e-12
TAU = 1e-8
PAD = 8
MIN_PIXELS = 64
DATA_RANGE = 14.0


def family_layout(pred_root: Path):
    return [
        ("CTA-UNet+FFL TC", pred_root / "cta_unet_ffl_transport_conditioned", range(10)),
        ("MS-TMO-UNet TC", pred_root / "ms_tmo_transport_conditioned", range(5)),
        ("MS-TMO-UNet TC", pred_root / "ms_tmo_transport_conditioned_recovery", range(5, 10)),
        ("CTA-UNet+FFL K-only", pred_root / "cta_unet_ffl_konly", range(1, 10)),
    ]


def roi_ssim(gt_log: np.ndarray, pred_log: np.ndarray, mask: np.ndarray) -> float | None:
    ys, xs = np.nonzero(mask)
    if len(ys) < MIN_PIXELS:
        return None
    y0 = max(0, int(ys.min()) - PAD)
    y1 = min(mask.shape[0], int(ys.max()) + PAD + 1)
    x0 = max(0, int(xs.min()) - PAD)
    x1 = min(mask.shape[1], int(xs.max()) + PAD + 1)
    return float(structural_similarity(gt_log[y0:y1, x0:x1], pred_log[y0:y1, x0:x1], data_range=DATA_RANGE))


def process_seed(family: str, seed: int, seed_dir: Path, data_root: Path, clip_ssim: bool) -> dict[str, object]:
    paths = sorted(glob.glob(str(seed_dir / "*.pred.npz")))
    if len(paths) != 170:
        raise FileNotFoundError(f"Expected 170 cache files in {seed_dir}, found {len(paths)}")
    totals = {name: 0.0 for name in ("gt", "pred", "union")}
    counts = {name: 0 for name in totals}
    for pred_path in paths:
        with np.load(pred_path) as cache:
            pred_log = cache["pred_log"].astype(np.float32)
            source = os.path.basename(str(cache["source_file"]))
        param_folder = Path(pred_path).name.split("__", 1)[0]
        with np.load(data_root / param_folder / source) as simulation:
            concentration = simulation["C"].astype(np.float32)
        gt_log = np.log10(np.clip(concentration, 0.0, None) + EPS_C)
        if clip_ssim:
            gt_log = np.clip(gt_log, -12.0, 2.0)
            pred_log = np.clip(pred_log, -12.0, 2.0)
        pred_phys = np.clip(np.power(10.0, pred_log) - EPS_C, 0.0, None)
        for timestep in range(concentration.shape[0]):
            gt_mask = concentration[timestep] > TAU
            pred_mask = pred_phys[timestep] > TAU
            masks = {
                "gt": gt_mask,
                "pred": pred_mask,
                "union": np.logical_or(gt_mask, pred_mask),
            }
            for name, mask in masks.items():
                value = roi_ssim(gt_log[timestep], pred_log[timestep], mask)
                if value is not None:
                    totals[name] += value
                    counts[name] += 1
    return {
        "family": family,
        "seed": seed,
        "gt_roi_ssim": totals["gt"] / counts["gt"],
        "pred_roi_ssim": totals["pred"] / counts["pred"],
        "union_roi_ssim": totals["union"] / counts["union"],
        "n_gt_roi_pairs": counts["gt"],
        "n_pred_roi_pairs": counts["pred"],
        "n_union_roi_pairs": counts["union"],
        "roi_pad_pixels": PAD,
        "roi_min_pixels": MIN_PIXELS,
        "ssim_array_policy": "symmetric_clip_-12_2" if clip_ssim else "legacy_raw_log",
    }


def validate_precomputed(tables_dir: Path) -> None:
    summary = pd.read_csv(tables_dir / "roi_ssim_gt_pred_union_summary.csv")
    per_seed = pd.read_csv(tables_dir / "roi_ssim_gt_pred_union_per_seed.csv")
    if summary.empty or per_seed.empty:
        raise ValueError("Precomputed ROI SSIM CSVs must not be empty.")
    print({"status": "precomputed-ok", "summary_rows": len(summary), "per_seed_rows": len(per_seed)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pred-root", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--clip-ssim", action="store_true")
    args = parser.parse_args()
    if args.data_root is None or args.pred_root is None:
        validate_precomputed(args.tables_dir)
        print("Full recomputation requires --data-root and --pred-root.")
        return

    rows = []
    for family, directory, seeds in family_layout(args.pred_root):
        for seed in seeds:
            print(f"Computing {family}, seed {seed}", flush=True)
            rows.append(process_seed(family, seed, directory / f"seed{seed}", args.data_root, args.clip_ssim))
    out_dir = args.out_dir or args.tables_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = out_dir / "roi_ssim_gt_pred_union_per_seed.csv"
    with per_seed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    frame = pd.DataFrame(rows)
    summary = frame.groupby("family", sort=False).agg(
        n_seeds=("seed", "count"),
        gt_roi_ssim_mean=("gt_roi_ssim", "mean"),
        gt_roi_ssim_std=("gt_roi_ssim", "std"),
        pred_roi_ssim_mean=("pred_roi_ssim", "mean"),
        pred_roi_ssim_std=("pred_roi_ssim", "std"),
        union_roi_ssim_mean=("union_roi_ssim", "mean"),
        union_roi_ssim_std=("union_roi_ssim", "std"),
        n_gt_roi_pairs=("n_gt_roi_pairs", "sum"),
        n_pred_roi_pairs=("n_pred_roi_pairs", "sum"),
        n_union_roi_pairs=("n_union_roi_pairs", "sum"),
    ).reset_index()
    summary.to_csv(out_dir / "roi_ssim_gt_pred_union_summary.csv", index=False)
    print(f"Wrote {len(rows)} per-seed rows to {out_dir}")


if __name__ == "__main__":
    main()
