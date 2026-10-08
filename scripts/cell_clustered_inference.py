#!/usr/bin/env python3
"""Cell-clustered inference for the held-out-combination (LOCO) contrasts.

Addresses two defects identified in the v5 pre-submission review.

Finding 3.3 -- The earlier hierarchical bootstrap resampled 29 (geological-cell x
realization) blocks as if they were independent. The split is constructed at geological-cell
level and the test partition contains only SIX cells; the five realizations inside a cell
share the geological factor tuple and the split assignment, so they are not independent
clusters. Intervals were therefore too narrow for a claim about unseen geology.

Finding 3.4 -- The earlier p-values were two-sided percentile tail probabilities of an
ordinary bootstrap distribution centred on the observed effect. That is a heuristic, not a
null-calibrated test, yet the values were carried into a Holm correction and reported as
p_Holm.

What this script does instead:

  CI    Three-level cluster bootstrap: resample training seeds, then resample the six
        geological cells with replacement, then resample realizations within each drawn
        cell. Cell-level resampling is what makes the interval honest about generalizing to
        unseen geology.

  p     Exact cell-level sign-flip permutation test. Under the null of no paired effect the
        sign of a cell's mean paired difference is exchangeable. With six cells all
        2^6 = 64 sign assignments are enumerated, so the two-sided p-value has a hard floor
        of 2/64 = 0.03125. That floor is a property of the design, not of the code, and is
        reported explicitly.

Holm-Bonferroni is then applied across the six withheld-pair tests.

Usage:
    python cell_clustered_inference.py <results_root> <registry_csv> [--out FILE]
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

METRICS = {"plume_ssim": 1.0, "global_ssim": 1.0}

HELDOUT = {
    "compositional": ("6.2", "1.0"),
    "loco_0p62_0p1": ("0.62", "0.1"),
    "loco_0p62_1p0": ("0.62", "1.0"),
    "loco_6p2_0p1": ("6.2", "0.1"),
    "loco_62p0_0p1": ("62.0", "0.1"),
    "loco_62p0_1p0": ("62.0", "1.0"),
}


def load_registry(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    combo = {r["param_folder"]: (r["long_dispersivity"], r["trans_ratio"]) for r in rows}
    cell = {r["param_folder"]: (r["sigma2Y"], r["correlation_length"], r["anisotropy"]) for r in rows}
    return combo, cell


def read_variant(root: Path, combo_of, target):
    """{(seed, param_folder, realization): {metric: value}} restricted to the withheld pair."""
    out = {}
    for seed_dir in sorted(root.glob("seed*")):
        seed = int(seed_dir.name.replace("seed", ""))
        f = seed_dir / "per_case_metrics.csv"
        if not f.is_file():
            continue
        with f.open(encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                if combo_of.get(r["param_folder"]) != target:
                    continue
                out[(seed, r["param_folder"], r["realization"])] = {
                    m: float(r[m]) for m in METRICS if m in r
                }
    return out


def cell_table(left, right, cell_of, metric, direction):
    """Paired differences as {cell: {realization: {seed: mean difference}}}."""
    keys = sorted(set(left) & set(right))
    seeds = sorted({k[0] for k in keys})
    acc: dict = {}
    for seed, pf, rz in keys:
        c = cell_of[pf]
        d = direction * (left[(seed, pf, rz)][metric] - right[(seed, pf, rz)][metric])
        if not np.isfinite(d):
            continue
        acc.setdefault(c, {}).setdefault(rz, {}).setdefault(seed, []).append(float(d))
    cells = sorted(acc)
    table = {}
    for c in cells:
        table[c] = {rz: {s: float(np.mean(v)) for s, v in byseed.items()}
                    for rz, byseed in acc[c].items()}
    return cells, table, seeds


def three_level_bootstrap(cells, table, seeds, replicates, rng):
    """Resample seeds, then cells with replacement, then realizations within each cell."""
    draws = np.empty(replicates)
    cell_list = list(cells)
    for i in range(replicates):
        ss = rng.choice(seeds, size=len(seeds), replace=True)
        cc = rng.integers(0, len(cell_list), size=len(cell_list))
        vals = []
        for ci in cc:
            c = cell_list[ci]
            rzs = list(table[c].keys())
            rr = rng.integers(0, len(rzs), size=len(rzs))
            inner = []
            for ri in rr:
                byseed = table[c][rzs[ri]]
                sv = [byseed[s] for s in ss if s in byseed]
                if sv:
                    inner.append(float(np.mean(sv)))
            if inner:
                vals.append(float(np.mean(inner)))
        draws[i] = float(np.mean(vals)) if vals else np.nan
    return draws


def cell_means(cells, table):
    """Mean paired difference per geological cell, averaged over realizations and seeds."""
    out = []
    for c in cells:
        rz_means = [float(np.mean(list(byseed.values()))) for byseed in table[c].values()]
        out.append(float(np.mean(rz_means)))
    return np.array(out)


def signflip_exact(cm):
    """Exact two-sided sign-flip permutation p-value over cell-level means."""
    n = len(cm)
    obs = float(np.mean(cm))
    stats = np.array([float(np.mean(cm * np.array(s)))
                      for s in itertools.product([-1, 1], repeat=n)])
    p = float(np.mean(np.abs(stats) >= abs(obs) - 1e-15))
    return p, 2.0 / (2 ** n), obs


def holm(pvals):
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("registry")
    ap.add_argument("--left", default="v1_factorized_levels")
    ap.add_argument("--right", default="v0_raw_channels")
    ap.add_argument("--replicates", type=int, default=10000)
    ap.add_argument("--rng-seed", type=int, default=20260920)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    combo_of, cell_of = load_registry(Path(a.registry))
    root = Path(a.root)
    rng = np.random.default_rng(a.rng_seed)

    results = {}
    for split, target in HELDOUT.items():
        ldir, rdir = root / split / a.left, root / split / a.right
        if not ldir.is_dir() or not rdir.is_dir():
            continue
        L = read_variant(ldir, combo_of, target)
        R = read_variant(rdir, combo_of, target)
        if not L or not R:
            continue
        entry = {"withheld_pair": {"alpha_L": target[0], "alpha_T_ratio": target[1]}}
        for metric, direction in METRICS.items():
            cells, table, seeds = cell_table(L, R, cell_of, metric, direction)
            if len(cells) < 2:
                continue
            draws = three_level_bootstrap(cells, table, seeds, a.replicates, rng)
            lo, hi = np.nanquantile(draws, [0.025, 0.975])
            cm = cell_means(cells, table)
            p, floor, obs = signflip_exact(cm)
            entry[metric] = {
                "mean_difference": obs,
                "ci95_lower": float(lo),
                "ci95_upper": float(hi),
                "excludes_zero": bool(lo > 0 or hi < 0),
                "p_signflip_exact": p,
                "p_floor_by_design": floor,
                "n_cells": len(cells),
                "n_seeds": len(seeds),
                "cell_level_means": [float(x) for x in cm],
                "n_cells_positive": int((cm > 0).sum()),
            }
        results[split] = entry

    for metric in METRICS:
        splits = [s for s in results if metric in results[s]]
        if not splits:
            continue
        ps = np.array([results[s][metric]["p_signflip_exact"] for s in splits])
        adj = holm(ps)
        for s, v in zip(splits, adj):
            results[s][metric]["p_holm"] = float(v)
            results[s][metric]["significant_holm_005"] = bool(v < 0.05)

    payload = {
        "analysis": "cell-clustered inference for held-out-combination contrasts",
        "contrast": f"{a.left} minus {a.right}",
        "ci_method": "three-level cluster bootstrap (seed x geological cell x realization-within-cell)",
        "p_method": "exact cell-level sign-flip permutation, Holm-corrected across withheld pairs",
        "replicates": a.replicates,
        "rng_seed": a.rng_seed,
        "note": ("With six test geological cells the two-sided sign-flip p-value cannot fall "
                 "below 2/2^6 = 0.03125; this is a property of the design."),
        "results": results,
    }
    text = json.dumps(payload, indent=2)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
