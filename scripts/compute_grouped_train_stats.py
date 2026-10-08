"""Compute log-conductivity normalization statistics for the grouped v5 split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shared.data.io import (
    filter_excluded_files,
    flatten_param_folders,
    load_param_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="splits/param_split_grouped_k_v5.json")
    parser.add_argument("--exclusions", default="metadata/k_integrity_exclusions_v5.csv")
    parser.add_argument("--out", default="configs/train_stats_grouped_k_v5.json")
    parser.add_argument("--eps-k", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_params, _, _ = load_param_split(args.split)
    files = filter_excluded_files(
        flatten_param_folders(args.data_root, train_params), args.exclusions
    )
    total = 0.0
    total_squared = 0.0
    count = 0
    for index, path in enumerate(files, start=1):
        with np.load(path) as data:
            conductivity = data["K"].astype(np.float64)
        values = np.log(np.clip(conductivity, args.eps_k, None))
        total += float(values.sum(dtype=np.float64))
        total_squared += float(np.square(values).sum(dtype=np.float64))
        count += int(values.size)
        if index % 100 == 0:
            print(f"processed={index}/{len(files)}", flush=True)
    mean = total / count
    variance = max(total_squared / count - mean * mean, 0.0)
    payload = {
        "schema_version": 1,
        "split": Path(args.split).as_posix(),
        "exclusion_manifest": Path(args.exclusions).as_posix(),
        "n_train_params": len(train_params),
        "n_train_files": len(files),
        "n_values": count,
        "file_glob": "*.npz",
        "use_logK": True,
        "eps_k": args.eps_k,
        "k_mean": mean,
        "k_std": float(np.sqrt(variance)),
        "concentration_statistics": "not used; concentration is transformed analytically with eps_c=1e-12",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
