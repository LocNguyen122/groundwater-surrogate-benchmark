"""Build the signed standardized-effect figure from its released CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "results" / "tables" / "signed_metric_advantage_data.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "figures" / "fig4_signed_metric_advantage.pdf")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    frame["label"] = frame["label"].replace(
        {
            "Mass error": "ICE/t",
            "Physics L1": "Concentration MAE",  # backward-compatible legacy label
            "Mass abs. per timestep": "ICE/t (shape cache)",
        }
    )
    values = frame["signed_advantage_d_z"].astype(float)
    colors = ["#D55E00" if value > 0 else "#0072B2" for value in values]

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(7.1, 4.8))
    bars = ax.barh(frame["label"], values, color=colors, edgecolor="black", linewidth=0.45)
    for bar, value, clear in zip(bars, values, frame["clear_at_alpha_0.05"]):
        hatch = "" if str(clear).upper() == "YES" else "///"
        bar.set_hatch(hatch)
        offset = 0.10 if value >= 0 else -0.10
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Signed paired standardized effect, $d_z$")
    ax.text(0.01, 1.02, r"MS-TMO-UNet TC favored  $\leftarrow$", transform=ax.transAxes, color="#0072B2", ha="left", fontweight="bold")
    ax.text(0.99, 1.02, r"$\rightarrow$  CTA-UNet+FFL TC favored", transform=ax.transAxes, color="#D55E00", ha="right", fontweight="bold")
    ax.grid(axis="x", alpha=0.22, linewidth=0.6)
    ax.invert_yaxis()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
