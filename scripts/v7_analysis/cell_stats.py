"""Cell-clustered inference used by every v7 contrast (pre-registration addendum, section 0).

Input: a long table of paired differences with columns seed, cell, unit (realization id within the cell) and d
(the per-case difference already averaged over the transport settings that the analysis names).
Three summaries are returned:
  (i)   three-level cluster bootstrap: seeds, then cells, then units within each drawn cell (percentile 95%);
  (ii)  cell-level sign-flip test on cell means, exact when 2^G <= 100,000, otherwise 100,000 random patterns;
  (iii) t interval with G - 1 degrees of freedom on cell means (small-cluster check).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

N_BOOT = 10_000
N_SIGNFLIP_MC = 100_000
RNG_SEED = 20261003


def _cube(df: pd.DataFrame) -> tuple[np.ndarray, list, list, list[list]]:
    """seed x cell x unit array of mean differences (NaN where a cell has fewer units)."""
    seeds = sorted(df["seed"].unique())
    cells = sorted(df["cell"].unique())
    units = [sorted(df.loc[df["cell"] == c, "unit"].unique()) for c in cells]
    nu = max(len(u) for u in units)
    arr = np.full((len(seeds), len(cells), nu), np.nan)
    g = df.groupby(["seed", "cell", "unit"])["d"].mean()
    s_ix = {s: i for i, s in enumerate(seeds)}
    for ci, (c, us) in enumerate(zip(cells, units)):
        for ui, u in enumerate(us):
            for s in seeds:
                if (s, c, u) in g.index:
                    arr[s_ix[s], ci, ui] = g[(s, c, u)]
    return arr, seeds, cells, units


def summarize(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = RNG_SEED) -> dict:
    arr, seeds, cells, units = _cube(df)
    S, G, U = arr.shape
    nunits = np.array([len(u) for u in units])
    cell_means = np.nanmean(np.nanmean(arr, axis=2), axis=0)          # per cell: mean over units, then seeds
    estimate = float(np.mean(cell_means))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        s_draw = rng.integers(0, S, S)
        c_draw = rng.integers(0, G, G)
        vals = []
        for c in c_draw:
            u_draw = rng.integers(0, nunits[c], nunits[c])
            vals.append(np.nanmean(arr[np.ix_(s_draw, [c], u_draw)]))
        boots[b] = np.mean(vals)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    obs = abs(cell_means.mean())
    if 2 ** G <= N_SIGNFLIP_MC:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=G)))
        exact = True
    else:
        signs = rng.choice([-1.0, 1.0], size=(N_SIGNFLIP_MC, G))
        exact = False
    null = np.abs((signs * cell_means).mean(axis=1))
    p_signflip = float((np.sum(null >= obs - 1e-15) + (0 if exact else 1)) / (len(null) + (0 if exact else 1)))

    se = cell_means.std(ddof=1) / np.sqrt(G) if G > 1 else np.nan
    tq = stats.t.ppf(0.975, G - 1) if G > 1 else np.nan
    t_stat = cell_means.mean() / se if se and se > 0 else np.nan
    p_t = float(2 * stats.t.sf(abs(t_stat), G - 1)) if np.isfinite(t_stat) else np.nan
    return {
        "estimate": estimate, "ci_low": float(lo), "ci_high": float(hi),
        "p_signflip": p_signflip, "signflip_exact": exact, "signflip_floor": float(2 / 2 ** G) if exact else None,
        "t_ci_low": float(estimate - tq * se), "t_ci_high": float(estimate + tq * se), "p_t": p_t,
        "n_seeds": S, "n_cells": G, "n_units_per_cell": nunits.tolist(),
        "cells_positive": int(np.sum(cell_means > 0)), "cells_negative": int(np.sum(cell_means < 0)),
        "cell_means": {str(c): float(v) for c, v in zip(cells, cell_means)},
        "n_boot": n_boot,
    }


def decide(res: dict, rule: str = "superiority", margin: float = 0.0) -> str:
    """Pre-registered decision labels."""
    if rule == "superiority":
        if res["ci_low"] > 0 and res["p_signflip"] < 0.05:
            return "SUPPORTED"
        if res["ci_high"] <= 0:
            return "NOT SUPPORTED"
        return "INCONCLUSIVE"
    if rule == "superiority_strict_negative":   # O3/O4b: NOT SUPPORTED only if interval < 0
        if res["ci_low"] > 0 and res["p_signflip"] < 0.05:
            return "SUPPORTED"
        if res["ci_high"] < 0:
            return "NOT SUPPORTED"
        return "INCONCLUSIVE"
    if rule == "equivalence":
        return "EQUIVALENT" if (res["ci_low"] > -margin and res["ci_high"] < margin) else "NOT EQUIVALENT"
    if rule == "noninferiority":
        return "NON-INFERIOR" if res["ci_low"] > -margin else "NOT SHOWN NON-INFERIOR"
    raise ValueError(rule)
