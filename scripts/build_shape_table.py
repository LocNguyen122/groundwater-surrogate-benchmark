"""Print the six-family plume-shape table from included CSV artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def fmt(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--csv", type=Path, default=None, help="Optional explicit six-family shape summary CSV.")
    args = parser.parse_args()
    path = args.csv or args.tables_dir / "shape_metrics_six_family_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing six-family shape summary CSV: {path}")
    df = pd.read_csv(path)
    required = {
        "family",
        "iou_mean_mean",
        "iou_mean_std",
        "dice_mean_mean",
        "dice_mean_std",
        "fp_area_frac_mean_mean",
        "fp_area_frac_mean_std",
        "fn_area_frac_mean_mean",
        "fn_area_frac_mean_std",
        "abs_plume_area_rel_err_mean_mean",
        "abs_plume_area_rel_err_mean_std",
        "centroid_err_mean_px_mean",
        "centroid_err_mean_px_std",
        "mass_abs_mean_mean",
        "mass_abs_mean_std",
        "n_seeds",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Shape CSV missing columns: {missing}")
    order = [
        "CTA-UNet+FFL TC",
        "MS-TMO-UNet TC",
        "Pix2Pix TC",
        "FNO TC",
        "CNN/U-Net TC",
        "CTA-UNet+FFL K-only",
    ]
    df["_order"] = df["family"].map({name: i for i, name in enumerate(order)})
    df = df.sort_values("_order")
    print("family,iou,dice,fp_area_frac,fn_area_frac,area_rel_err,centroid_px,ICE/t,seeds")
    for _, row in df.iterrows():
        print(
            ",".join(
                [
                    str(row["family"]),
                    fmt(row["iou_mean_mean"], row["iou_mean_std"], 3),
                    fmt(row["dice_mean_mean"], row["dice_mean_std"], 3),
                    fmt(row["fp_area_frac_mean_mean"], row["fp_area_frac_mean_std"], 5),
                    fmt(row["fn_area_frac_mean_mean"], row["fn_area_frac_mean_std"], 5),
                    fmt(row["abs_plume_area_rel_err_mean_mean"], row["abs_plume_area_rel_err_mean_std"], 3),
                    fmt(row["centroid_err_mean_px_mean"], row["centroid_err_mean_px_std"], 2),
                    fmt(row["mass_abs_mean_mean"], row["mass_abs_mean_std"], 1),
                    str(int(row["n_seeds"])),
                ]
            )
        )


if __name__ == "__main__":
    main()
