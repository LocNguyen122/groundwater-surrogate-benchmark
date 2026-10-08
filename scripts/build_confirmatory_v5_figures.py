"""Build compact confirmatory v5 figures from lightweight CSV summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FAMILY_ORDER = (
    "cta_ffl_k_only",
    "cta_ffl_matched",
    "cta_ffl_permuted",
    "cta_ffl_constant_placebo",
    "cta_ffl_independent_nuisance",
    "ms_tmo_matched",
)

FAMILY_LABELS = (
    "$K$-only",
    "Matched\ncodes",
    "Cyclic\nlabels",
    "Const.\nplacebo",
    "Nuis.\nplacebo",
    "MS-TMO\nmatched",
)

CODE_TAGS = (
    "fixed_alphaL_0p62_ratio_0p1",
    "fixed_alphaL_0p62_ratio_1p0",
    "fixed_alphaL_6p2_ratio_0p1",
    "fixed_alphaL_6p2_ratio_1p0",
    "fixed_alphaL_62p0_ratio_0p1",
    "fixed_alphaL_62p0_ratio_1p0",
)

CODE_LABELS = (
    "0.62, 0.1",
    "0.62, 1.0",
    "6.2, 0.1",
    "6.2, 1.0",
    "62, 0.1",
    "62, 1.0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--counterfactual-matrix", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_csv(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_condition_figure(rows: list[dict[str, str]], out_path: Path) -> None:
    by_family = {row["family"]: row for row in rows}
    colors = ["#607D8B", "#00796B", "#26A69A", "#9E9E9E", "#B0BEC5", "#5E35B1"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("global_ssim", "plume_ssim"),
        ("Global SSIM", "GT-ROI plume SSIM"),
    ):
        means = [float(by_family[family][f"{metric}_mean"]) for family in FAMILY_ORDER]
        errors = [float(by_family[family][f"{metric}_sd"]) for family in FAMILY_ORDER]
        x = np.arange(len(FAMILY_ORDER))
        axis.bar(x, means, yerr=errors, capsize=2.5, color=colors, edgecolor="white", linewidth=0.7)
        axis.set_xticks(x, FAMILY_LABELS)
        axis.set_ylabel(title)
        axis.set_ylim(0.62 if metric == "plume_ssim" else 0.82, 1.0)
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", labelsize=6.5, pad=2)
        axis.tick_params(axis="y", labelsize=8)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_counterfactual_figure(rows: list[dict[str, str]], out_path: Path) -> None:
    lookup = {(row["true_code"], row["supplied_code"]): row for row in rows}
    matrix = np.array(
        [
            [float(lookup[(true_tag, supplied_tag)]["plume_ssim"]) for supplied_tag in CODE_TAGS]
            for true_tag in CODE_TAGS
        ]
    )
    fig, axis = plt.subplots(figsize=(5.2, 4.25), constrained_layout=True)
    image = axis.imshow(matrix, cmap="viridis", vmin=0.45, vmax=1.0, aspect="equal")
    axis.set_xticks(np.arange(6), CODE_LABELS, rotation=35, ha="right")
    axis.set_yticks(np.arange(6), CODE_LABELS)
    axis.set_xlabel(r"Supplied $(\alpha_L,\alpha_T/\alpha_L)$ code")
    axis.set_ylabel(r"Simulator $(\alpha_L,\alpha_T/\alpha_L)$ setting")
    axis.tick_params(labelsize=8)
    for row in range(6):
        for column in range(6):
            value = matrix[row, column]
            color = "black" if value > 0.78 else "white"
            axis.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=7.2, color=color)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("GT-ROI plume SSIM")
    colorbar.ax.tick_params(labelsize=8)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_condition_figure(read_csv(args.summary), out_dir / "fig_confirmatory_v5_conditions.pdf")
    build_counterfactual_figure(
        read_csv(args.counterfactual_matrix), out_dir / "fig_confirmatory_v5_counterfactual_matrix.pdf"
    )
    print(f"Wrote confirmatory v5 figures to {out_dir}")


if __name__ == "__main__":
    main()
