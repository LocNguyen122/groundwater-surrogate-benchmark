#!/usr/bin/env python3
"""Cell-clustered inference for the whole-test-set confirmatory contrasts (v6).

Replaces the v5 analysis for the main confirmatory table, which had two defects identified in
the v5 pre-submission review:

  * the hierarchical bootstrap resampled 29 (geological cell x realization) blocks as if they
    were independent, although the test partition contains six geological cells;
  * the physical metrics (ICE per timestep, concentration MAE) were computed against raw ground
    truth that contains three solver-corrupted realizations. Those three realizations hold
    0.00365% of test voxels but 99.99% of test concentration mass (raw maximum 8.3e8 against a
    physical source concentration of 1.0), so the unscreened metrics measure the corruption,
    not the surrogates.

Here every case is first averaged over the six transport settings inside its
(geological cell, realization) unit, units are averaged within cells, and inference is carried
out over the six cells: a three-level cluster bootstrap (seed x cell x realization-within-cell)
for intervals, and an exact cell-level sign-flip test plus a cell-level paired t-test for
p-values. Physical metrics are reported after excluding the corrupted realizations; SSIM is
reported both ways because the fixed [-12, 2] SSIM range already clips the corrupted voxels.

Usage:
    python confirmatory_cell_clustered.py <runs_root> <registry_csv> [--out FILE]
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

CORRUPTED = {("param_033", "real_003"), ("param_114", "real_003"), ("param_122", "real_004")}
METRICS = {
    "global_ssim": 1.0,
    "plume_ssim": 1.0,
    "ice_per_timestep": -1.0,
    "concentration_mae": -1.0,
}
PHYSICAL = {"ice_per_timestep", "concentration_mae"}
CONTRASTS = [
    ("cta_ffl_matched", "cta_ffl_k_only"),
    ("cta_ffl_matched", "cta_ffl_permuted"),
    ("cta_ffl_matched", "cta_ffl_constant_placebo"),
    ("cta_ffl_matched", "cta_ffl_independent_nuisance"),
    ("cta_ffl_matched", "ms_tmo_matched"),
]


def load_cells(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {r["param_folder"]: (r["sigma2Y"], r["correlation_length"], r["anisotropy"]) for r in rows}


def read_condition(root: Path):
    out = {}
    for seed_dir in sorted(root.glob("seed*")):
        f = seed_dir / "per_case_metrics.csv"
        if not f.is_file():
            continue
        seed = int(seed_dir.name.replace("seed", ""))
        with f.open(encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                out[(seed, r["param_folder"], r["realization"])] = {m: float(r[m]) for m in METRICS}
    return out


def unit_table(left, right, cell_of, metric, direction, screen):
    """{cell: {realization: {seed: mean over transport settings of the paired difference}}}."""
    acc: dict = {}
    for key in sorted(set(left) & set(right)):
        seed, pf, rz = key
        if screen and (pf, rz) in CORRUPTED:
            continue
        d = direction * (left[key][metric] - right[key][metric])
        if np.isfinite(d):
            acc.setdefault(cell_of[pf], {}).setdefault(rz, {}).setdefault(seed, []).append(d)
    return {c: {rz: {s: float(np.mean(v)) for s, v in bs.items()} for rz, bs in rzs.items()}
            for c, rzs in acc.items()}


def cell_means(table):
    cells = sorted(table)
    return cells, np.array([np.mean([np.mean(list(bs.values())) for bs in table[c].values()])
                            for c in cells])


def bootstrap(table, replicates, rng):
    cells = sorted(table)
    seeds = sorted({s for c in cells for bs in table[c].values() for s in bs})
    draws = np.empty(replicates)
    for i in range(replicates):
        ss = rng.choice(seeds, size=len(seeds), replace=True)
        vals = []
        for ci in rng.integers(0, len(cells), size=len(cells)):
            rzs = list(table[cells[ci]].values())
            inner = []
            for ri in rng.integers(0, len(rzs), size=len(rzs)):
                sv = [rzs[ri][s] for s in ss if s in rzs[ri]]
                if sv:
                    inner.append(np.mean(sv))
            if inner:
                vals.append(np.mean(inner))
        draws[i] = np.mean(vals)
    return draws, len(seeds)


def signflip(cm):
    obs = float(np.mean(cm))
    null = np.array([np.mean(cm * np.array(s)) for s in itertools.product([-1, 1], repeat=len(cm))])
    return float(np.mean(np.abs(null) >= abs(obs) - 1e-15))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("registry")
    ap.add_argument("--replicates", type=int, default=10000)
    ap.add_argument("--rng-seed", type=int, default=20260921)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    cell_of = load_cells(Path(a.registry))
    rng = np.random.default_rng(a.rng_seed)

    conds = {d.name: read_condition(d) for d in sorted(root.iterdir()) if d.is_dir()}
    summary = {}
    for name, data in conds.items():
        entry = {}
        for metric in METRICS:
            for screen in (False, True):
                per_seed = {}
                for (seed, pf, rz), vals in data.items():
                    if screen and (pf, rz) in CORRUPTED:
                        continue
                    per_seed.setdefault(seed, []).append(vals[metric])
                means = [np.mean(v) for v in per_seed.values()]
                tag = metric + ("_screened" if screen else "_all")
                entry[tag] = {"mean": float(np.mean(means)), "sd": float(np.std(means, ddof=1)),
                              "n_seeds": len(means)}
        summary[name] = entry

    contrasts = []
    for left, right in CONTRASTS:
        if left not in conds or right not in conds:
            continue
        row = {"left": left, "right": right}
        for metric, direction in METRICS.items():
            for screen in ((True,) if metric in PHYSICAL else (False, True)):
                table = unit_table(conds[left], conds[right], cell_of, metric, direction, screen)
                cells, cm = cell_means(table)
                draws, n_seeds = bootstrap(table, a.replicates, rng)
                lo, hi = np.quantile(draws, [0.025, 0.975])
                t_p = float(stats.ttest_1samp(cm, 0.0).pvalue)
                tag = metric + ("_screened" if screen else "_all")
                row[tag] = {
                    "direction_adjusted_mean": float(np.mean(cm)),
                    "ci95": [float(lo), float(hi)],
                    "excludes_zero": bool(lo > 0 or hi < 0),
                    "p_signflip_exact": signflip(cm),
                    "p_cell_ttest": t_p,
                    "n_cells": len(cells),
                    "n_cells_favouring_left": int((cm > 0).sum()),
                    "n_seeds": n_seeds,
                }
        contrasts.append(row)

    payload = {
        "analysis": "whole-test-set confirmatory contrasts, cell-clustered (v6)",
        "ci_method": "three-level cluster bootstrap: seed x geological cell x realization-within-cell",
        "p_methods": ["exact cell-level sign-flip (floor 2/2^6 = 0.03125)", "cell-level paired t-test"],
        "screened_realizations": sorted("/".join(k) for k in CORRUPTED),
        "direction": "positive favours the left condition for every metric",
        "condition_summary": summary,
        "contrasts": contrasts,
    }
    text = json.dumps(payload, indent=2)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
