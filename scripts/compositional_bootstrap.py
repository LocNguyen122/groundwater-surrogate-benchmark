#!/usr/bin/env python3
"""Hierarchical seed x K-block bootstrap for the compositional (held-out pair) comparison.

Matches the inference standard of repo_v5/scripts/hierarchical_bootstrap_confirmatory.py so the
method-upgrade results use the same block-aware machinery the v4 review required: resample training
seeds and geological-factor-by-realization conductivity blocks, then report percentile 95% CIs on
direction-adjusted paired differences.

Difference from the v5 script: the comparison is restricted to evaluation cases carrying ONE
transport pair (the pair withheld from training), so there is no within-block averaging over
transport settings -- each (geology, realization) block contributes exactly that pair's case.

Usage:
    python compositional_bootstrap.py <results_root> <registry_csv> [--replicates N] [--out FILE]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

METRICS = {"plume_ssim": 1.0, "global_ssim": 1.0, "ice_per_timestep": -1.0, "concentration_mae": -1.0}

# split label -> transport pair withheld from training (long_dispersivity, trans_ratio)
HELDOUT = {
    "compositional": ("6.2", "1.0"),
    "loco_0p62_0p1": ("0.62", "0.1"),
    "loco_0p62_1p0": ("0.62", "1.0"),
    "loco_6p2_0p1": ("6.2", "0.1"),
    "loco_62p0_0p1": ("62.0", "0.1"),
    "loco_62p0_1p0": ("62.0", "1.0"),
}
INTERIOR_ALPHA_L = "6.2"  # the only interior level of {0.62, 6.2, 62.0}


def load_registry(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    combo = {r["param_folder"]: (r["long_dispersivity"], r["trans_ratio"]) for r in rows}
    block = {r["param_folder"]: (r["sigma2Y"], r["correlation_length"], r["anisotropy"]) for r in rows}
    return combo, block


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


def bootstrap(left, right, block_of, replicates, rng_seed):
    keys = sorted(set(left) & set(right))
    if not keys:
        return None
    seeds = sorted({k[0] for k in keys})
    blocks = sorted({(*block_of[k[1]], k[2]) for k in keys})
    si = {v: i for i, v in enumerate(seeds)}
    bi = {v: i for i, v in enumerate(blocks)}
    rng = np.random.default_rng(rng_seed)
    res = {}
    for metric, direction in METRICS.items():
        if metric not in left[keys[0]]:
            continue
        cells = [[[] for _ in blocks] for _ in seeds]
        for k in keys:
            d = direction * (left[k][metric] - right[k][metric])
            if np.isfinite(d):
                cells[si[k[0]]][bi[(*block_of[k[1]], k[2])]].append(float(d))
        mat = np.full((len(seeds), len(blocks)), np.nan)
        for i in range(len(seeds)):
            for j in range(len(blocks)):
                if cells[i][j]:
                    mat[i, j] = float(np.mean(cells[i][j]))
        if np.isnan(mat).any():
            # incomplete pairing (campaign still running) -- report and skip rather than guess
            res[metric] = {"status": "incomplete_pairing"}
            continue
        obs = float(mat.mean())
        draws = np.empty(replicates)
        for r in range(replicates):
            ss = rng.integers(0, len(seeds), size=len(seeds))
            bb = rng.integers(0, len(blocks), size=len(blocks))
            draws[r] = mat[np.ix_(ss, bb)].mean()
        lo, hi = np.quantile(draws, [0.025, 0.975])
        # two-sided percentile bootstrap p-value; floored at 1/replicates (cannot resolve below it)
        frac_le = float(np.mean(draws <= 0.0))
        frac_ge = float(np.mean(draws >= 0.0))
        pval = max(min(1.0, 2.0 * min(frac_le, frac_ge)), 1.0 / replicates)
        res[metric] = {
            "mean_difference": obs,
            "ci95_lower": float(lo),
            "ci95_upper": float(hi),
            "p_value": pval,
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_seeds": len(seeds),
            "n_blocks": len(blocks),
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("registry")
    ap.add_argument("--replicates", type=int, default=10000)
    ap.add_argument("--rng-seed", type=int, default=20260723)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(a.root)
    combo_of, block_of = load_registry(Path(a.registry))
    payload = {
        "comparison": "held-out transport pair, zero-shot",
        "resampling_unit": "training seed and geological-factor-by-realization K block",
        "replicates": a.replicates,
        "rng_seed": a.rng_seed,
        "note": "Positive mean_difference favours the first-named variant.",
        "pairs": {},
    }
    for split, target in HELDOUT.items():
        sp = root / split
        if not sp.is_dir():
            continue
        variants = {d.name: read_variant(d, combo_of, target) for d in sorted(sp.iterdir()) if d.is_dir()}
        entry = {
            "held_out_pair": {"alpha_L": target[0], "alpha_T_over_alpha_L": target[1]},
            "level_type": "interior" if target[0] == INTERIOR_ALPHA_L else "boundary",
            "contrasts": {},
        }
        base = "v1_factorized_levels"
        if base in variants:
            for opp in ("v0_raw_channels", "v2_joint_embedding", "v3_factorized_continuous"):
                if opp in variants and variants[opp]:
                    r = bootstrap(variants[base], variants[opp], block_of, a.replicates, a.rng_seed)
                    if r:
                        entry["contrasts"][f"{base}_minus_{opp}"] = r
        payload["pairs"][split] = entry

    # ---- Holm-Bonferroni across the family of withheld pairs (V1 vs V0, plume SSIM) ----
    fam = []
    for split, e in payload["pairs"].items():
        c = e.get("contrasts", {}).get("v1_factorized_levels_minus_v0_raw_channels", {}).get("plume_ssim")
        if isinstance(c, dict) and "p_value" in c:
            fam.append([split, float(c["p_value"])])
    fam.sort(key=lambda r: r[1])
    m = len(fam)
    running = 0.0
    holm = {}
    for i, (split, p_raw) in enumerate(fam):
        adj = min(1.0, (m - i) * p_raw)
        running = max(running, adj)  # enforce monotonicity
        holm[split] = {"p_raw": p_raw, "p_holm": running, "significant_at_0.05": running < 0.05}
    payload["holm_family"] = {
        "family": "V1 minus V0, plume SSIM, one test per withheld transport pair",
        "n_tests": m,
        "method": "Holm-Bonferroni, step-down, monotonicity enforced",
        "results": holm,
    }

    text = json.dumps(payload, indent=2)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"wrote {a.out}")
    # concise console summary
    for split, e in payload["pairs"].items():
        c = e["contrasts"].get("v1_factorized_levels_minus_v0_raw_channels", {}).get("plume_ssim")
        if isinstance(c, dict) and "mean_difference" in c:
            flag = "CI excludes 0" if c["excludes_zero"] else "CI includes 0"
            print(f"  {split:16s} [{e['level_type']:8s}] V1-V0 plume SSIM "
                  f"{c['mean_difference']:+.4f}  95% CI [{c['ci95_lower']:+.4f}, {c['ci95_upper']:+.4f}]  {flag}"
                  f"  (seeds={c['n_seeds']}, blocks={c['n_blocks']})")
    h = payload.get("holm_family", {})
    if h.get("results"):
        print("")
        print(f"  --- Holm-Bonferroni across {h['n_tests']} withheld pairs (V1 vs V0, plume SSIM) ---")
        for split, r in sorted(h["results"].items(), key=lambda kv: kv[1]["p_raw"]):
            mark = "SIG" if r["significant_at_0.05"] else "ns "
            print(f"  {split:16s} p_raw={r['p_raw']:.4f}  p_holm={r['p_holm']:.4f}  {mark}")
        nsig = sum(r["significant_at_0.05"] for r in h["results"].values())
        print(f"  => {nsig}/{h['n_tests']} pairs survive Holm correction at alpha=0.05")


if __name__ == "__main__":
    main()
