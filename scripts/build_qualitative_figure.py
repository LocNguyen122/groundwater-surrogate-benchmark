"""Build the deterministic four-column qualitative cache panel.

The selected case is the lexicographically first file shared by the three
seed-1 caches. The stored arrays use the vertically flipped learning
orientation. For presentation only, every panel is flipped back to the
physical model-grid convention in which the injection well is at row 399.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EPS_C = 1e-12


def display_physical(values: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(values, 0.0, None) + EPS_C)


def load_prediction(path: Path) -> np.ndarray:
    with np.load(path) as cache:
        return cache["pred_log"].astype(np.float32)


def physical_grid_orientation(values: np.ndarray) -> np.ndarray:
    """Undo the vertical preprocessing flip for visual presentation only."""
    return np.flip(values, axis=-2)


def shared_case(pred_root: Path, seed: int) -> str:
    directories = [
        pred_root / "cta_unet_ffl_transport_conditioned" / f"seed{seed}",
        pred_root / "ms_tmo_transport_conditioned" / f"seed{seed}",
        pred_root / "cta_unet_ffl_konly" / f"seed{seed}",
    ]
    names = [set(path.name for path in directory.glob("*.pred.npz")) for directory in directories]
    common = sorted(set.intersection(*names))
    if not common:
        raise FileNotFoundError("No prediction-cache case is shared by all displayed families.")
    return common[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    case = shared_case(args.pred_root, args.seed)
    param_folder, realization = case.replace(".pred.npz", "").split("__", 1)
    with np.load(args.data_root / param_folder / f"{realization}.npz") as simulation:
        ground_truth = simulation["C"].astype(np.float32)
    columns = [
        ("Ground truth", physical_grid_orientation(display_physical(ground_truth))),
        ("CTA-UNet+FFL TC", physical_grid_orientation(load_prediction(args.pred_root / "cta_unet_ffl_transport_conditioned" / f"seed{args.seed}" / case))),
        ("MS-TMO-UNet TC", physical_grid_orientation(load_prediction(args.pred_root / "ms_tmo_transport_conditioned" / f"seed{args.seed}" / case))),
        ("CTA-UNet+FFL K-only", physical_grid_orientation(load_prediction(args.pred_root / "cta_unet_ffl_konly" / f"seed{args.seed}" / case))),
    ]
    timesteps = [0, 15, 19, 24]
    days = [1, 90, 180, 365]

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(4, 4, figsize=(7.5, 9.4))
    image = None
    for row, (timestep, day) in enumerate(zip(timesteps, days)):
        for col, (label, values) in enumerate(columns):
            ax = axes[row, col]
            image = ax.imshow(values[timestep], cmap="viridis", vmin=-10.0, vmax=0.3, origin="upper")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(label, pad=7)
            if col == 0:
                ax.set_ylabel(f"Day {day}", labelpad=8)
    fig.subplots_adjust(left=0.055, right=0.89, top=0.965, bottom=0.075, wspace=0.08, hspace=0.055)
    color_axis = fig.add_axes([0.91, 0.24, 0.018, 0.46])
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_label(r"$\log_{10}(C+\varepsilon_c)$")
    fig.text(
        0.5,
        0.025,
        f"Lexicographically first shared test case: {param_folder}/{realization}, seed {args.seed}. Displayed in physical grid orientation.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"{args.out} ({param_folder}/{realization}, seed {args.seed})")


if __name__ == "__main__":
    main()
