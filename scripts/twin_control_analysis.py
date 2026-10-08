#!/usr/bin/env python3
"""Twin control: does the accuracy collapse come from the twins or from less training data?

The realization-disjoint models train on realizations 1-3 and are evaluated on realizations 4-5,
where plume SSIM is far below the grouped-split value. Two explanations compete:

  (a) they saw only 60% of the training files, or
  (b) realizations 4-5 have no amplitude-rescaled training twin, whereas every grouped-split test
      field does.

This compares the SAME checkpoints on the SAME test geological cells under both conditions:
realizations 1-3 (twin present in training) against realizations 4-5 (no twin). Training data is
identical in both columns, so any gap is attributable to twin presence alone.

Usage:
    python twin_control_analysis.py <rd_root> <registry_csv> [--out FILE]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

import numpy as np
from scipy import stats

from cell_clustered_inference import load_registry

VARIANTS = ("k_only", "v0_raw_channels", "v1_factorized_levels")
WITHHELD = ("6.2", "1.0")


def read(path: Path, combo):
    out = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            out[(r["param_folder"], r["realization"])] = {
                "plume_ssim": float(r["plume_ssim"]),
                "global_ssim": float(r["global_ssim"]),
                "withheld": combo[r["param_folder"]] == WITHHELD,
            }
    return out


def summarise(root: Path, variant, combo, withheld):
    twin, notwin = [], []
    for seed_dir in sorted((root / variant).glob("seed*")):
        t = read(seed_dir / "per_case_metrics_twins.csv", combo)
        n = read(seed_dir / "per_case_metrics.csv", combo)
        tv = [v["plume_ssim"] for v in t.values() if v["withheld"] == withheld]
        nv = [v["plume_ssim"] for v in n.values() if v["withheld"] == withheld]
        if tv and nv:
            twin.append(st.mean(tv))
            notwin.append(st.mean(nv))
    return twin, notwin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rd_root")
    ap.add_argument("registry")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    combo, _ = load_registry(Path(a.registry))
    root = Path(a.rd_root)

    payload = {
        "analysis": "twin control: identical checkpoints, test fields with vs without a training twin",
        "note": ("Both columns use models trained on realizations 1-3, so training data is held "
                 "fixed; only the presence of an amplitude-rescaled twin differs."),
        "results": {},
    }
    print(f"{'variant':24s}{'subset':16s}{'twin present':>16}{'no twin':>12}{'gap':>10}{'paired p':>11}")
    for variant in VARIANTS:
        payload["results"][variant] = {}
        for withheld, label in ((False, "trained pairs"), (True, "withheld pair")):
            twin, notwin = summarise(root, variant, combo, withheld)
            if not twin:
                continue
            gap = st.mean(twin) - st.mean(notwin)
            p = float(stats.ttest_rel(twin, notwin).pvalue)
            payload["results"][variant][label] = {
                "twin_present": {"mean": st.mean(twin), "sd": st.stdev(twin)},
                "no_twin": {"mean": st.mean(notwin), "sd": st.stdev(notwin)},
                "gap": gap, "p_paired_ttest": p, "n_seeds": len(twin),
            }
            print(f"{variant:24s}{label:16s}{st.mean(twin):16.4f}{st.mean(notwin):12.4f}"
                  f"{gap:+10.4f}{p:11.5f}")
    if a.out:
        Path(a.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
