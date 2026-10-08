"""Build the focal scalar-comparison figures from released per-seed values."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    ("CTA-UNet+FFL", "TC"): "#277da1",
    ("MS-TMO-UNet", "TC"): "#577590",
    ("CTA-UNet+FFL", "K-only"): "#f8961e",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def values(rows: list[dict[str, str]], family: str, regime: str, metric: str) -> np.ndarray:
    return np.asarray(
        [float(row[metric]) for row in rows if row["family"] == family and row["regime"] == regime],
        dtype=np.float64,
    )


def build_bar(rows: list[dict[str, str]], output: Path) -> None:
    metrics = (
        ("global_ssim", "Global SSIM", "higher is better"),
        ("plume_ssim", "Plume-region SSIM", "higher is better"),
        ("mass_err", "ICE/t", "lower is better"),
        ("physics_l1", "Concentration MAE", "lower is better"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.55))
    for axis, (metric, title, direction) in zip(axes, metrics):
        groups = [
            values(rows, "CTA-UNet+FFL", "K-only", metric),
            values(rows, "CTA-UNet+FFL", "TC", metric),
        ]
        means = [group.mean() for group in groups]
        sds = [group.std(ddof=1) for group in groups]
        bars = axis.bar(
            [0, 1], means, yerr=sds, capsize=3,
            color=[COLORS[("CTA-UNet+FFL", "K-only")], COLORS[("CTA-UNet+FFL", "TC")]],
            edgecolor="black", linewidth=0.5,
        )
        axis.set_xticks([0, 1], ["K-only", "TC"])
        axis.set_title(f"{title}\n{direction}", fontsize=9)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        for bar, mean in zip(bars, means):
            label = f"{mean:.4f}" if mean < 1 else f"{mean:.0f}"
            axis.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7)
    fig.tight_layout(w_pad=1.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def build_tradeoff(rows: list[dict[str, str]], output: Path) -> None:
    groups = (
        ("CTA-UNet+FFL", "TC", "CTA-UNet+FFL TC", "o"),
        ("MS-TMO-UNet", "TC", "MS-TMO-UNet TC", "s"),
        ("CTA-UNet+FFL", "K-only", "CTA-UNet+FFL K-only", "^"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.05), sharey=True)
    for axis, x_metric, x_label in (
        (axes[0], "mass_err", "ICE/t (lower is better)"),
        (axes[1], "physics_l1", "Concentration MAE (lower is better)"),
    ):
        for family, regime, label, marker in groups:
            xs = values(rows, family, regime, x_metric)
            ys = values(rows, family, regime, "plume_ssim")
            color = COLORS[(family, regime)]
            axis.scatter(xs, ys, s=22, marker=marker, color=color, alpha=0.72,
                         edgecolor="white", linewidth=0.35, label=label)
            axis.scatter([xs.mean()], [ys.mean()], s=125, marker="*", color=color,
                         edgecolor="black", linewidth=0.55, zorder=4)
        axis.set_xlabel(x_label)
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Plume-region SSIM (higher is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "tables" / "scalar_benchmark_per_seed.csv")
    parser.add_argument("--bar-out", type=Path, default=ROOT / "figures" / "fig2_konly_vs_tc_scalar.pdf")
    parser.add_argument("--tradeoff-out", type=Path, default=ROOT / "figures" / "fig5_scalar_tradeoff.pdf")
    args = parser.parse_args()
    rows = load_rows(args.input)
    build_bar(rows, args.bar_out)
    build_tradeoff(rows, args.tradeoff_out)
    print(args.bar_out)
    print(args.tradeoff_out)


if __name__ == "__main__":
    main()
