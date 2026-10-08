"""Print the threshold-sensitivity table from included CSV artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--csv", type=Path, default=None, help="Optional explicit threshold summary CSV.")
    args = parser.parse_args()
    path = args.csv or args.tables_dir / "threshold_sensitivity_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing threshold-sensitivity summary CSV: {path}")
    df = pd.read_csv(path)
    required = {
        "family",
        "threshold",
        "iou_mean_mean",
        "iou_mean_std",
        "dice_mean_mean",
        "dice_mean_std",
        "n_seeds",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Threshold CSV missing columns: {missing}")
    print("family,threshold,IoU,Dice,seeds")
    for _, row in df.sort_values(["family", "threshold"]).iterrows():
        print(
            f"{row['family']},{float(row['threshold']):.0e},"
            f"{row['iou_mean_mean']:.4f} +/- {row['iou_mean_std']:.4f},"
            f"{row['dice_mean_mean']:.4f} +/- {row['dice_mean_std']:.4f},"
            f"{int(row['n_seeds'])}"
        )


if __name__ == "__main__":
    main()
