"""Print the GT-, predicted-, and union-ROI SSIM table from included CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--csv", type=Path, default=None, help="Optional explicit ROI SSIM summary CSV.")
    args = parser.parse_args()
    path = args.csv or args.tables_dir / "roi_ssim_gt_pred_union_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing ROI SSIM summary CSV: {path}")
    df = pd.read_csv(path)
    required = {
        "family",
        "n_seeds",
        "gt_roi_ssim_mean",
        "gt_roi_ssim_std",
        "pred_roi_ssim_mean",
        "pred_roi_ssim_std",
        "union_roi_ssim_mean",
        "union_roi_ssim_std",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"ROI SSIM CSV missing columns: {missing}")
    print("family,GT-ROI,pred-ROI,union-ROI,seeds")
    for _, row in df.iterrows():
        print(
            ",".join(
                [
                    str(row["family"]),
                    fmt(row["gt_roi_ssim_mean"], row["gt_roi_ssim_std"]),
                    fmt(row["pred_roi_ssim_mean"], row["pred_roi_ssim_std"]),
                    fmt(row["union_roi_ssim_mean"], row["union_roi_ssim_std"]),
                    str(int(row["n_seeds"])),
                ]
            )
        )


if __name__ == "__main__":
    main()
