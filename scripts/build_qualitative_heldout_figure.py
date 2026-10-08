#!/usr/bin/env python3
"""Qualitative figure: zero-shot prediction on a WITHHELD transport pair, GT vs V0 vs V1.

Case selection is pre-specified, not chosen by outcome: the lexicographically first test
configuration carrying the withheld pair (alpha_L=6.2, alpha_T/alpha_L=1.0) and its first
realization. This mirrors the selection rule already used for the v5 qualitative figure.

Both models were trained on the compositional split, so neither ever saw this transport pair.
Fields are shown in the paper's bounded log10 concentration domain [-12, 2], with one shared
colour scale across all panels so the columns are directly comparable.

Usage:
    python build_qualitative_heldout_figure.py <figure_assets_dir> [--out FILE.pdf]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

LOG_MIN, LOG_MAX = -12.0, 2.0
# T=25 acquisition days for this dataset family
DAYS = [1, 2, 3, 4, 5, 7, 10, 14, 15, 20, 30, 40, 50, 60, 75, 90, 100, 120, 150, 180, 210, 240,
        270, 300, 365]
TIMESTEPS = [5, 13, 24]  # day 7 (early), day 60 (mid), day 365 (late)


def bounded_log10(phys: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(np.log10(np.clip(phys.astype(np.float64), 0.0, None) + eps), LOG_MIN, LOG_MAX)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("assets")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    A = a.assets
    out = a.out or os.path.join(A, "fig_qualitative_heldout_pair.pdf")

    name = "param_021__real_001.pred.npz"
    z0 = np.load(os.path.join(A, "pred_v0", name))
    z1 = np.load(os.path.join(A, "pred_v1", name))
    eps = float(z0["eps_c"])

    gt = bounded_log10(np.asarray(z0["gt_phys"]), eps)
    p0 = np.clip(np.asarray(z0["pred_log"], dtype=np.float64), LOG_MIN, LOG_MAX)
    p1 = np.clip(np.asarray(z1["pred_log"], dtype=np.float64), LOG_MIN, LOG_MAX)
    assert gt.shape == p0.shape == p1.shape, (gt.shape, p0.shape, p1.shape)

    m0 = json.load(open(os.path.join(A, "v0_fig.json"), encoding="utf-8"))
    m1 = json.load(open(os.path.join(A, "v1_fig.json"), encoding="utf-8"))
    s0 = m0["plume"]["mean_plume_ssim_log_bbox"]
    s1 = m1["plume"]["mean_plume_ssim_log_bbox"]

    cols = [
        ("Ground truth", gt, ""),
        ("Raw constant channels", p0, f"plume SSIM {s0:.4f}"),
        ("Factorized conditioning", p1, f"plume SSIM {s1:.4f}"),
    ]

    # Crop to the union of *meaningful* support so the plume fills the panel. A floor-epsilon test
    # is useless here because the predictions sit marginally above -12 almost everywhere; we use a
    # concentration level well above the floor instead.
    SUPPORT_LEVEL = LOG_MIN + 4.0  # log10 C > -8
    sup = (gt > SUPPORT_LEVEL) | (p0 > SUPPORT_LEVEL) | (p1 > SUPPORT_LEVEL)
    rows_any, cols_any = np.where(sup.any(axis=0))
    pad = 12
    r0, r1 = max(0, rows_any.min() - pad), min(gt.shape[1], rows_any.max() + pad + 1)
    c0, c1 = max(0, cols_any.min() - pad), min(gt.shape[2], cols_any.max() + pad + 1)

    vmin = float(min(gt[:, r0:r1, c0:c1].min(), p0[:, r0:r1, c0:c1].min(), p1[:, r0:r1, c0:c1].min()))
    vmax = float(max(gt[:, r0:r1, c0:c1].max(), p0[:, r0:r1, c0:c1].max(), p1[:, r0:r1, c0:c1].max()))

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType, required by many publishers
    })
    fig = plt.figure(figsize=(6.6, 5.4))
    gs = GridSpec(len(TIMESTEPS), 4, figure=fig, width_ratios=[1, 1, 1, 0.05],
                  wspace=0.06, hspace=0.10)

    im = None
    for r, t in enumerate(TIMESTEPS):
        for c, (title, arr, sub) in enumerate(cols):
            ax = fig.add_subplot(gs[r, c])
            # Stored learning arrays are row-0-at-top; the physical model grid is the
            # vertical mirror of that. Flip on display so this panel matches the
            # orientation used by build_dataset_contact_sheet.py and the other plume
            # figures. Without the flip the plume renders upside down relative to
            # every other figure in the paper.
            im = ax.imshow(np.flip(arr[t, r0:r1, c0:c1], axis=0), vmin=vmin, vmax=vmax,
                           cmap="viridis", interpolation="nearest", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title + (f"\n{sub}" if sub else "\n"), pad=4)
            if c == 0:
                ax.set_ylabel(f"day {DAYS[t]}", labelpad=3)

    cax = fig.add_subplot(gs[:, 3])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\log_{10}$ concentration (bounded to $[-12,2]$)", labelpad=6)

    fig.savefig(out, bbox_inches="tight", dpi=600)
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"wrote {out}")
    print(f"case: param_021 real_001 (withheld pair alpha_L=6.2, alpha_T/alpha_L=1.0)")
    print(f"timesteps shown: {[DAYS[t] for t in TIMESTEPS]} days")
    print(f"colour range: [{vmin:.3f}, {vmax:.3f}]  crop rows {r0}:{r1} cols {c0}:{c1}")
    print(f"plume SSIM  V0 raw={s0:.4f}   V1 factorized={s1:.4f}")


if __name__ == "__main__":
    main()
