"""Target-only solver QA: list realizations whose simulated concentration exceeds the physical bound.

Rule (fixed before any model output is inspected): a stored realization is flagged when any of its 25
concentration snapshots has max C > THRESHOLD * C0, with C0 = 1 the injected source concentration and
THRESHOLD = 10. Advective-dispersive transport of a unit-concentration source cannot exceed C0, so a value an
order of magnitude above it can only come from a non-converged solver step. The rule reads simulator targets
only (never predictions) and is applied to every partition. Flagged realizations are excluded from physical
metrics (ICE/t, concentration MAE) in every analysis; SSIM is reported both with and without them.

Usage:
  python scripts/build_solver_qa_exclusions.py --data-root <FLIPPED archive> --split splits/param_split_grouped_k_v5.json \
      --out metadata/solver_qa_exclusions_v7.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

THRESHOLD = 10.0
DAYS = [1, 2, 3, 4, 5, 7, 10, 14, 15, 20, 30, 40, 50, 60, 75, 90, 100, 120, 150, 180, 210, 240, 270, 300, 365]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="", help="optional: label each flagged realization with its partition")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    part = {}
    if args.split:
        sp = json.loads(Path(args.split).read_text())["splits"]
        part = {f: p for p in ("train", "val", "test") for f in sp[p]["param_folders"]}
    rows, n = [], 0
    for f in sorted(glob.glob(os.path.join(args.data_root, "param_*", "real_*.npz"))):
        n += 1
        with np.load(f) as d:
            c = np.asarray(d["C"], dtype=np.float64)
        cmax = c.reshape(c.shape[0], -1).max(axis=1)
        bad = np.where(cmax > args.threshold)[0]
        if bad.size:
            pf, rz = Path(f).parent.name, Path(f).name
            rows.append({"sample_key": f"{pf}/{rz}", "param_folder": pf, "realization": rz[:-4],
                         "partition": part.get(pf, ""), "flagged_days": " ".join(str(DAYS[i]) for i in bad),
                         "max_concentration": f"{cmax.max():.6g}", "reason": f"max_C_gt_{args.threshold:g}_x_C0"})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_key", "param_folder", "realization", "partition", "flagged_days",
                                           "max_concentration", "reason"])
        w.writeheader()
        w.writerows(rows)
    counts = {}
    for r in rows:
        counts[r["partition"] or "unassigned"] = counts.get(r["partition"] or "unassigned", 0) + 1
    print(json.dumps({"files_scanned": n, "flagged": len(rows), "by_partition": counts, "threshold": args.threshold}))


if __name__ == "__main__":
    main()
