"""Conductivity-twin audit: standardized log-K correlations between evaluation and training fields.

Computes, from the stored conductivity fields, the two numbers the manuscript quotes:
  grouped split          : for every distinct test field, its maximum |r| against any training field
                           (the paper reports the minimum of these maxima: every test field has a twin);
  realization-disjoint   : the maximum |r| between any evaluation field (test cells, realizations 4-5)
                           and any training field (training cells, realizations 1-3).
r is the Pearson correlation of the centred natural-log conductivity over the full 600 x 400 grid. Distinct
fields are identified by SHA-256 of the stored array, so the six transport settings of a realization count once.

Usage:
  python scripts/conductivity_twin_audit.py --data-root <FLIPPED data root> \
      --split splits/param_split_grouped_k_v5.json --exclusions metadata/k_integrity_exclusions_v5.csv \
      --out results/audit_v7/conductivity_twin_audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.shared.data.io import filter_excluded_files, flatten_param_folders, load_param_split


def unit_logk(k: np.ndarray) -> np.ndarray:
    z = np.log(np.maximum(k.astype(np.float64), 1e-6)).ravel()
    z -= z.mean()
    return z / np.linalg.norm(z)


def distinct_fields(files: list[str]) -> dict[str, tuple[np.ndarray, list[str]]]:
    out: dict[str, tuple[np.ndarray, list[str]]] = {}
    for f in files:
        with np.load(f) as d:
            k = np.asarray(d["K"])
        h = hashlib.sha256(np.ascontiguousarray(k).tobytes()).hexdigest()
        key = f"{Path(f).parent.name}/{Path(f).name}"
        if h in out:
            out[h][1].append(key)
        else:
            out[h] = (unit_logk(k), [key])
    return out


def realization(key: str) -> int:
    return int(key.split("real_")[1].split(".")[0])


def max_abs_r(eval_fields: dict, train_fields: dict) -> np.ndarray:
    E = np.stack([v[0] for v in eval_fields.values()])
    T = np.stack([v[0] for v in train_fields.values()])
    return np.abs(E @ T.T).max(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--exclusions", required=True)
    ap.add_argument("--rd-train-realizations", default="1,2,3")
    ap.add_argument("--rd-eval-realizations", default="4,5")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train_p, _, test_p = load_param_split(args.split)
    train = distinct_fields(filter_excluded_files(flatten_param_folders(args.data_root, train_p), args.exclusions))
    test = distinct_fields(filter_excluded_files(flatten_param_folders(args.data_root, test_p), args.exclusions))
    grouped = max_abs_r(test, train)

    rd_train_q = {int(x) for x in args.rd_train_realizations.split(",")}
    rd_eval_q = {int(x) for x in args.rd_eval_realizations.split(",")}
    rd_train = {h: v for h, v in train.items() if all(realization(k) in rd_train_q for k in v[1])}
    rd_eval = {h: v for h, v in test.items() if all(realization(k) in rd_eval_q for k in v[1])}
    rd = max_abs_r(rd_eval, rd_train)

    payload = {
        "statistic": "Pearson r of centred natural-log K over the full grid; distinct arrays by SHA-256",
        "split": args.split, "exclusions": args.exclusions,
        "n_distinct_train_fields": len(train), "n_distinct_test_fields": len(test),
        "grouped": {"min_over_test_of_max_abs_r": float(grouped.min()),
                    "median_over_test_of_max_abs_r": float(np.median(grouped)),
                    "n_test_fields_with_twin_r_ge_0p999": int((grouped >= 0.999).sum())},
        "realization_disjoint": {"train_realizations": sorted(rd_train_q), "eval_realizations": sorted(rd_eval_q),
                                 "n_train_fields": len(rd_train), "n_eval_fields": len(rd_eval),
                                 "max_abs_r": float(rd.max()), "median_of_per_field_max": float(np.median(rd))},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
