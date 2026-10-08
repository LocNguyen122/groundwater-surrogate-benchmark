#!/usr/bin/env python3
"""Aggregate the Obj1 method-upgrade campaign into paper-ready tables.

CPU-only. Reads ONLY on-disk per_case_metrics.csv from the fetched run tree; never fabricates.
Handles: standard split, primary compositional split, 5 leave-one-combo-out (LOCO) splits, V0-V3.

Usage:
    python scripts/aggregate_method_upgrade_campaign.py <runs_root> [--registry CSV]
"""
from __future__ import annotations

import csv
import os
import statistics as st
import sys
from collections import defaultdict

ROOT = ""
REG = ""

VARIANTS = ["v0_raw_channels", "v1_factorized_levels", "v2_joint_embedding", "v3_factorized_continuous"]
SHORT = {"v0_raw_channels": "V0 raw-channels", "v1_factorized_levels": "V1 FiLM-factorized",
         "v2_joint_embedding": "V2 joint-embed", "v3_factorized_continuous": "V3 FiLM-continuous"}
# split label -> held-out transport pair (None = no hold-out)
HELDOUT = {
    "grouped_standard": None,
    "compositional": ("6.2", "1.0"),
    "loco_0p62_0p1": ("0.62", "0.1"),
    "loco_0p62_1p0": ("0.62", "1.0"),
    "loco_6p2_0p1": ("6.2", "0.1"),
    "loco_62p0_0p1": ("62.0", "0.1"),
    "loco_62p0_1p0": ("62.0", "1.0"),
}


def load_registry(path: str) -> dict[str, tuple[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return {r["param_folder"]: (r["long_dispersivity"], r["trans_ratio"]) for r in csv.DictReader(fh)}


def run_mean(path: str, reg: dict, combo=None, exclude=None) -> tuple[float, int]:
    """Mean plume SSIM over cases, optionally restricted to / excluding a transport combo."""
    vals = []
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            c = reg.get(r["param_folder"])
            if combo is not None and c != combo:
                continue
            if exclude is not None and c == exclude:
                continue
            v = float(r["plume_ssim"])
            if v == v:  # NaN guard
                vals.append(v)
    return (sum(vals) / len(vals) if vals else float("nan")), len(vals)


def collect(root: str, reg: dict) -> dict:
    """{(split, variant): {seed: {'all':x, 'held':y, 'seen':z}}}"""
    out: dict = defaultdict(dict)
    for split in sorted(os.listdir(root)):
        sp = os.path.join(root, split)
        if not os.path.isdir(sp):
            continue
        for variant in sorted(os.listdir(sp)):
            vp = os.path.join(sp, variant)
            if not os.path.isdir(vp):
                continue
            for seed_dir in sorted(os.listdir(vp)):
                f = os.path.join(vp, seed_dir, "per_case_metrics.csv")
                if not os.path.isfile(f):
                    continue
                seed = int(seed_dir.replace("seed", ""))
                Z = HELDOUT.get(split)
                rec = {"all": run_mean(f, reg)[0]}
                if Z is not None:
                    rec["held"] = run_mean(f, reg, combo=Z)[0]
                    rec["seen"] = run_mean(f, reg, exclude=Z)[0]
                out[(split, variant)][seed] = rec
    return out


def fmt(vals: list[float]) -> str:
    vals = [v for v in vals if v == v]
    if not vals:
        return "     n/a    "
    m = sum(vals) / len(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.4f}±{s:.4f}(n={len(vals)})"


def main() -> None:
    global ROOT, REG
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs_root", help="fetched run tree holding <split>/<variant>/seed*/per_case_metrics.csv")
    ap.add_argument("--registry", default="metadata/parameter_registry_162_v5.csv")
    args = ap.parse_args()
    ROOT, REG = args.runs_root, args.registry
    reg = load_registry(REG)
    data = collect(ROOT, reg)
    if not data:
        print(f"No runs found under {ROOT}")
        return

    print("=" * 96)
    print("TABLE 1 — Standard grouped split (seen-combo test): overall plume SSIM")
    print("=" * 96)
    for v in VARIANTS:
        d = data.get(("grouped_standard", v))
        if d:
            print(f"  {SHORT[v]:22s} {fmt([r['all'] for r in d.values()])}   seeds={sorted(d)}")

    print("\n" + "=" * 96)
    print("TABLE 2 — Compositional generalization: plume SSIM ON THE HELD-OUT COMBO (zero-shot)")
    print("=" * 96)
    print(f"  {'held-out combo':>16} | " + " | ".join(f"{SHORT[v]:>22}" for v in VARIANTS))
    wins = defaultdict(int)
    totals = 0
    for split, Z in HELDOUT.items():
        if Z is None:
            continue
        cells, means = [], {}
        for v in VARIANTS:
            d = data.get((split, v))
            vals = [r.get("held", float("nan")) for r in d.values()] if d else []
            vals = [x for x in vals if x == x]
            cells.append(fmt(vals) if vals else "        --        ")
            if vals:
                means[v] = sum(vals) / len(vals)
        if means:
            totals += 1
            best = max(means, key=means.get)
            wins[best] += 1
        print(f"  {str(Z):>16} | " + " | ".join(f"{c:>22}" for c in cells))
    if totals:
        print(f"\n  Best-variant tally across {totals} held-out combos: " +
              ", ".join(f"{SHORT[k]}={n}" for k, n in sorted(wins.items(), key=lambda kv: -kv[1])))

    print("\n" + "=" * 96)
    print("TABLE 3 — Compositional gap (seen combos minus held-out combo), same trained model")
    print("=" * 96)
    for split, Z in HELDOUT.items():
        if Z is None:
            continue
        row = []
        for v in VARIANTS:
            d = data.get((split, v))
            if not d:
                row.append("   --   ")
                continue
            gaps = [r["seen"] - r["held"] for r in d.values()
                    if r.get("seen") == r.get("seen") and r.get("held") == r.get("held")]
            row.append(f"{sum(gaps)/len(gaps):+.4f}" if gaps else "   --   ")
        print(f"  {str(Z):>16} | " + " | ".join(f"{c:>22}" for c in row))

    print("\n" + "=" * 96)
    print("DECISION SUPPORT (per P4 rule)")
    print("=" * 96)
    s0 = data.get(("grouped_standard", "v0_raw_channels"), {})
    s1 = data.get(("grouped_standard", "v1_factorized_levels"), {})
    if s0 and s1:
        m0 = st.mean([r["all"] for r in s0.values()])
        m1 = st.mean([r["all"] for r in s1.values()])
        print(f"  (i)  standard test: V1={m1:.4f} V0={m0:.4f} |diff|={abs(m1-m0):.4f} "
              f"-> {'MATCH' if abs(m1-m0) < 0.01 else 'DIFFER'}")
    # paired per-seed win rate of V1 over V0 and V2 on held-out combos
    for opp in ("v0_raw_channels", "v2_joint_embedding"):
        w = n = 0
        for split, Z in HELDOUT.items():
            if Z is None:
                continue
            a, b = data.get((split, "v1_factorized_levels")), data.get((split, opp))
            if not a or not b:
                continue
            for seed in sorted(set(a) & set(b)):
                x, y = a[seed].get("held"), b[seed].get("held")
                if x == x and y == y:
                    n += 1
                    w += int(x > y)
        if n:
            print(f"  (ii) V1 beats {SHORT[opp]:22s} on held-out combo in {w}/{n} paired runs "
                  f"({100*w/n:.0f}%)")


if __name__ == "__main__":
    main()
