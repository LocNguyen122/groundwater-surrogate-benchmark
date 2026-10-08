"""Counterfactual code-sensitivity figure for the v7 manuscript (grouped split, matched CTA-UNet, 10 seeds).

Two panels from the same per-case sweeps: global SSIM (penalizes structure anywhere in the domain) and plume SSIM
(ground-truth region; oracle-assisted). Axes are categorical code assignments; all six settings were in training.
Also writes the cell-level and seed-level consistency of the diagonal (correct code highest in its row):
for every held-out geological cell (true geology, registry v7) and every seed, the correct-code mean is compared
with the best wrong-code mean over the realizations of that cell and setting.

Inputs: <run_root>/seed*/counterfactual_sweep/<code>_per_case.csv (one file per supplied code, all test cases).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODES = [("0p62", "0p1"), ("0p62", "1p0"), ("6p2", "0p1"), ("6p2", "1p0"), ("62p0", "0p1"), ("62p0", "1p0")]
LABEL = {c: f"({c[0].replace('p', '.').replace('.0', '') if c[0] == '62p0' else c[0].replace('p', '.')}, "
            f"{c[1].replace('p', '.')})" for c in CODES}
METRICS = (("global_ssim", "(a) Global SSIM"), ("plume_ssim", "(b) Plume SSIM (ground-truth region)"))


def tag(c: tuple[str, str]) -> str:
    return f"fixed_alphaL_{c[0]}_ratio_{c[1]}"


def to_code(alpha_l: float, ratio: float) -> tuple[str, str]:
    a = {0.62: "0p62", 6.2: "6p2", 62.0: "62p0"}[round(alpha_l, 2)]
    return a, {0.1: "0p1", 1.0: "1p0"}[round(ratio, 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True, help="runs_v5_confirmatory_20260715/cta_ffl_matched")
    ap.add_argument("--registry_v7", required=True)
    ap.add_argument("--out", required=True, help="output PDF; a JSON with the same stem is written alongside")
    a = ap.parse_args()
    reg = {r["param_folder"]: r for r in csv.DictReader(open(a.registry_v7, encoding="utf-8"))}
    frames = []
    for sd in sorted(Path(a.run_root).glob("seed*")):
        for c in CODES:
            df = pd.read_csv(sd / "counterfactual_sweep" / f"{tag(c)}_per_case.csv")
            frames.append(df.assign(seed=int(sd.name[4:]), supplied=tag(c)))
    d = pd.concat(frames, ignore_index=True)
    d["true"] = [tag(to_code(float(reg[p]["long_dispersivity"]), float(reg[p]["trans_ratio"]))) for p in d.param_folder]
    d["cell"] = [(reg[p]["sigma2Y"], reg[p]["correlation_length"], reg[p]["anisotropy"]) for p in d.param_folder]
    n_seeds, n_cells = d.seed.nunique(), d.cell.nunique()

    out = {"n_seeds": int(n_seeds), "n_cells": int(n_cells), "n_cases_per_seed": int(len(d) // (n_seeds * len(CODES))),
           "matrix": {}, "consistency": {}}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.1), constrained_layout=True)
    order = [tag(c) for c in CODES]
    for ax, (metric, title) in zip(axes, METRICS):
        mat = d.pivot_table(index="true", columns="supplied", values=metric, aggfunc="mean").loc[order, order]
        out["matrix"][metric] = {r: {c: float(mat.loc[r, c]) for c in order} for r in order}
        im = ax.imshow(mat.to_numpy(), cmap="viridis", vmin=0.45, vmax=1.0)
        for i in range(6):
            for j in range(6):
                v = mat.iloc[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.2,
                        color="black" if v > 0.8 else "white", fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(6), [LABEL[c] for c in CODES], rotation=35, ha="right", fontsize=7.5)
        ax.set_yticks(range(6), [LABEL[c] for c in CODES], fontsize=7.5)
        ax.set_xlabel(r"supplied code $(\alpha_L, \alpha_T/\alpha_L)$ (categorical)", fontsize=9)
        ax.set_ylabel(r"simulated setting $(\alpha_L, \alpha_T/\alpha_L)$", fontsize=9)
        ax.set_title(title, fontsize=10)
        # consistency: per (cell, true row) and per (seed, true row), correct minus best wrong code
        unit = d.groupby(["cell", "true", "supplied"])[metric].mean().unstack("supplied")[order]
        per_seed = d.groupby(["seed", "true", "supplied"])[metric].mean().unstack("supplied")[order]
        def margins(tab):
            res = []
            for idx, row in tab.iterrows():
                t = idx[1]
                res.append(float(row[t] - row.drop(t).max()))
            return np.array(res)
        mc, ms = margins(unit), margins(per_seed)
        rows = {r: float(mat.loc[r, r] - mat.loc[r].drop(r).max()) for r in order}
        out["consistency"][metric] = {
            "pooled_row_margin": rows,
            "cell_rows_correct_highest": int((mc > 0).sum()), "cell_rows_total": int(mc.size),
            "cell_row_margin_min": float(mc.min()),
            "seed_rows_correct_highest": int((ms > 0).sum()), "seed_rows_total": int(ms.size)}
    fig.colorbar(im, ax=axes, shrink=0.75, label="SSIM")
    fig.savefig(a.out, bbox_inches="tight")
    Path(a.out).with_suffix(".json").write_text(json.dumps(out, indent=1))
    for k, v in out["consistency"].items():
        print(k, {x: v[x] for x in v if x != "pooled_row_margin"},
              {r[-13:]: round(m, 3) for r, m in v["pooled_row_margin"].items()})


if __name__ == "__main__":
    main()
