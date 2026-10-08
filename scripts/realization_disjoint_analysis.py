#!/usr/bin/env python3
"""Analyse the realization-disjoint sensitivity study (v6).

The random-field generator reuses one seed for every geological cell, so under the grouped split
every test conductivity field has a training twin that is an exact amplitude rescaling of it
(standardized log-K correlation 1.000). Different realization IDs are independent, so training on
realizations 1-3 and testing on realizations 4-5 is the only evaluation in this dataset with
genuinely unseen geology (max test-to-train correlation 0.087).

This script reports, under that split:
  * absolute plume SSIM for K-only, raw channels (V0) and factorized levels (V1), separately for
    the five transport pairs seen in training and for the withheld pair (6.2, 1.0);
  * cell-clustered paired contrasts (three-level cluster bootstrap over seeds, geological cells and
    realizations within cell; cell-level paired t-test), using the same machinery as
    cell_clustered_inference.py.

Usage:
    python realization_disjoint_analysis.py <rd_root> <registry_csv> [--out FILE]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

import numpy as np
from scipy import stats

from cell_clustered_inference import load_registry, three_level_bootstrap

WITHHELD = ("6.2", "1.0")
VARIANTS = ("k_only", "v0_raw_channels", "v1_factorized_levels")
CONTRASTS = [
    ("v0_raw_channels", "k_only", False, "codes vs K-only, trained pairs"),
    ("v0_raw_channels", "k_only", True, "codes vs K-only, withheld pair"),
    ("v1_factorized_levels", "v0_raw_channels", False, "V1 vs V0, trained pairs (in-design)"),
    ("v1_factorized_levels", "v0_raw_channels", True, "V1 vs V0, withheld pair (compositional)"),
    ("v1_factorized_levels", "k_only", True, "V1 vs K-only, withheld pair"),
]


def load_variant(root: Path, combo):
    out = {}
    for seed_dir in sorted(root.glob("seed*")):
        f = seed_dir / "per_case_metrics.csv"
        if not f.is_file():
            continue
        seed = int(seed_dir.name.replace("seed", ""))
        with f.open(encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                out[(seed, r["param_folder"], r["realization"])] = {
                    "plume_ssim": float(r["plume_ssim"]),
                    "global_ssim": float(r["global_ssim"]),
                    "withheld": combo[r["param_folder"]] == WITHHELD,
                }
    return out


def absolute(data, metric, withheld):
    per_seed = {}
    for (seed, _, _), v in data.items():
        if v["withheld"] == withheld:
            per_seed.setdefault(seed, []).append(v[metric])
    means = [st.mean(x) for x in per_seed.values()]
    return {"mean": st.mean(means), "sd": st.stdev(means), "n_seeds": len(means)}


def contrast(left, right, cell_of, metric, withheld, rng, replicates):
    acc = {}
    for key in sorted(set(left) & set(right)):
        seed, pf, rz = key
        if left[key]["withheld"] != withheld:
            continue
        d = left[key][metric] - right[key][metric]
        acc.setdefault(cell_of[pf], {}).setdefault(rz, {}).setdefault(seed, []).append(d)
    table = {c: {rz: {s: float(np.mean(v)) for s, v in b.items()} for rz, b in r.items()}
             for c, r in acc.items()}
    cells = sorted(table)
    cm = np.array([np.mean([np.mean(list(b.values())) for b in table[c].values()]) for c in cells])
    seeds = sorted({s for c in cells for b in table[c].values() for s in b})
    draws = three_level_bootstrap(cells, table, seeds, replicates, rng)
    lo, hi = np.nanquantile(draws, [0.025, 0.975])
    return {
        "mean_difference": float(cm.mean()),
        "ci95": [float(lo), float(hi)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "p_cell_ttest": float(stats.ttest_1samp(cm, 0.0).pvalue),
        "n_cells": int(len(cm)),
        "n_cells_positive": int((cm > 0).sum()),
        "n_seeds": len(seeds),
        "cell_level_means": [float(x) for x in cm],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rd_root", help=".../realization_disjoint_results/compositional")
    ap.add_argument("registry")
    ap.add_argument("--replicates", type=int, default=10000)
    ap.add_argument("--rng-seed", type=int, default=20260922)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    combo, cell_of = load_registry(Path(a.registry))
    root = Path(a.rd_root)
    data = {v: load_variant(root / v, combo) for v in VARIANTS}
    rng = np.random.default_rng(a.rng_seed)

    result = {"absolute": {}, "contrasts": []}
    for v in VARIANTS:
        result["absolute"][v] = {
            metric: {"trained_pairs": absolute(data[v], metric, False),
                     "withheld_pair": absolute(data[v], metric, True)}
            for metric in ("plume_ssim", "global_ssim")
        }
    for left, right, withheld, label in CONTRASTS:
        entry = {"label": label, "left": left, "right": right, "withheld_pair_only": withheld}
        for metric in ("plume_ssim", "global_ssim"):
            entry[metric] = contrast(data[left], data[right], cell_of, metric, withheld,
                                     rng, a.replicates)
        result["contrasts"].append(entry)

    payload = {
        "analysis": "realization-disjoint sensitivity (train realizations 1-3, test 4-5)",
        "withheld_pair": {"alpha_L": WITHHELD[0], "alpha_T_ratio": WITHHELD[1]},
        "inference": "three-level cluster bootstrap + cell-level paired t-test; 6 test cells",
        **result,
    }
    text = json.dumps(payload, indent=2)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")

    print("ABSOLUTE plume SSIM (mean +/- SD over seeds)")
    for v in VARIANTS:
        t = result["absolute"][v]["plume_ssim"]
        print(f"  {v:22s} trained pairs {t['trained_pairs']['mean']:.4f} +/- {t['trained_pairs']['sd']:.4f}"
              f"   withheld {t['withheld_pair']['mean']:.4f} +/- {t['withheld_pair']['sd']:.4f}")
    print("PAIRED CONTRASTS, plume SSIM, cell-clustered (positive favours first)")
    for c in result["contrasts"]:
        m = c["plume_ssim"]
        print(f"  {c['label']:42s} {m['mean_difference']:+.4f} "
              f"CI[{m['ci95'][0]:+.4f},{m['ci95'][1]:+.4f}] t-p={m['p_cell_ttest']:.4f} "
              f"cells {m['n_cells_positive']}/{m['n_cells']}")


if __name__ == "__main__":
    main()
