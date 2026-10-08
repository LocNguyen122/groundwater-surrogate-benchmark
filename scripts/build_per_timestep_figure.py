"""Rebuild the historical per-timestep panel in a compact portrait layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("CTA-UNet+FFL TC", "MS-TMO-UNet TC", "CTA-UNet+FFL K-only")
COLORS = ("#1f77b4", "#4c566a", "#d17a00")


def build(source: Path, output: Path) -> None:
    frame = pd.read_csv(source)
    frame = frame[frame["family"].isin(FAMILIES)].copy()
    metrics = (
        ("iou_mean", "IoU", "higher is better"),
        ("dice_mean", "Dice", "higher is better"),
        ("centroid_err_mean_px", "Full-field concentration-weighted centroid error (px)", "lower is better"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(6.3, 8.2), sharex=True)
    for axis, (column, label, direction) in zip(axes, metrics):
        for family, color in zip(FAMILIES, COLORS):
            subset = frame[frame["family"] == family]
            summary = subset.groupby("timestep")[column].agg(["mean", "std"]).reset_index()
            x = summary["timestep"].to_numpy()
            mean = summary["mean"].to_numpy()
            std = summary["std"].fillna(0).to_numpy()
            axis.plot(x, mean, color=color, linewidth=1.8, label=family)
            axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
        axis.set_ylabel(label, fontsize=8)
        axis.text(0.99, 0.05, direction, transform=axis.transAxes, ha="right", va="bottom", fontsize=7)
        axis.grid(alpha=0.25, linewidth=0.6)
    axes[-1].set_xlabel("Timestep index")
    axes[0].legend(loc="best", frameon=False, fontsize=7)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "results" / "tables" / "shape_metrics_by_timestep.csv",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "figures" / "fig6_per_timestep.pdf")
    args = parser.parse_args()
    build(args.source, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
