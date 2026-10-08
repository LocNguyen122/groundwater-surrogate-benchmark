"""Hierarchical seed and K-realization bootstrap for v5 focal comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = {
    "global_ssim": 1.0,
    "plume_ssim": 1.0,
    "ice_per_timestep": -1.0,
    "concentration_mae": -1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-root", required=True, help="Directory containing seed0..seed9/per_case_metrics.csv")
    parser.add_argument("--right-root", required=True)
    parser.add_argument("--registry", default="metadata/parameter_registry_162_v5.csv")
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def read_seed_rows(root: Path) -> dict[tuple[int, str, str], dict[str, float]]:
    rows = {}
    for seed_dir in sorted(root.glob("seed*")):
        seed = int(seed_dir.name.replace("seed", ""))
        path = seed_dir / "per_case_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows[(seed, row["param_folder"], row["realization"])] = {
                    metric: float(row[metric]) for metric in METRICS
                }
    return rows


def main() -> None:
    args = parse_args()
    with Path(args.registry).open("r", encoding="utf-8-sig", newline="") as handle:
        registry_rows = list(csv.DictReader(handle))
    factors = {
        f"param_{int(row['param_id']):03d}": (
            row["sigma2Y"], row["correlation_length"], row["anisotropy"]
        )
        for row in registry_rows
    }
    left = read_seed_rows(Path(args.left_root))
    right = read_seed_rows(Path(args.right_root))
    expected_seeds = list(range(10))
    left_seeds = sorted({key[0] for key in left})
    right_seeds = sorted({key[0] for key in right})
    if left_seeds != expected_seeds or right_seeds != expected_seeds:
        raise ValueError(
            "Confirmatory bootstrap requires exactly seeds 0..9 in both roots; "
            f"left={left_seeds}, right={right_seeds}."
        )
    keys = sorted(set(left) & set(right))
    if not keys:
        raise ValueError("No paired per-case rows were found.")
    if set(left) != set(right):
        raise ValueError("The paired conditions do not contain identical seed and case keys.")
    seeds = sorted({key[0] for key in keys})
    blocks = sorted({(*factors[key[1]], key[2]) for key in keys})
    if not blocks:
        raise ValueError("No geological-factor-by-realization blocks were found.")
    seed_index = {value: index for index, value in enumerate(seeds)}
    block_index = {value: index for index, value in enumerate(blocks)}

    results = []
    rng = np.random.default_rng(args.seed)
    for metric, direction in METRICS.items():
        cells: list[list[list[float]]] = [
            [[] for _ in blocks] for _ in seeds
        ]
        for key in keys:
            seed, param_folder, realization = key
            block = (*factors[param_folder], realization)
            difference = direction * (left[key][metric] - right[key][metric])
            if np.isfinite(difference):
                cells[seed_index[seed]][block_index[block]].append(float(difference))
        matrix = np.full((len(seeds), len(blocks)), np.nan, dtype=np.float64)
        for i in range(len(seeds)):
            for j in range(len(blocks)):
                if cells[i][j]:
                    matrix[i, j] = float(np.mean(cells[i][j]))
        if np.isnan(matrix).any():
            raise ValueError(f"Missing paired data for metric={metric}")
        observed = float(matrix.mean())
        draws = np.empty(args.replicates, dtype=np.float64)
        for replicate in range(args.replicates):
            sampled_seeds = rng.integers(0, len(seeds), size=len(seeds))
            sampled_blocks = rng.integers(0, len(blocks), size=len(blocks))
            draws[replicate] = matrix[np.ix_(sampled_seeds, sampled_blocks)].mean()
        lower, upper = np.quantile(draws, [0.025, 0.975])
        results.append(
            {
                "metric": metric,
                "direction_adjusted_difference": f"{args.left_label} better when positive",
                "mean_difference": observed,
                "bootstrap_ci95_lower": float(lower),
                "bootstrap_ci95_upper": float(upper),
                "n_seeds": len(seeds),
                "n_k_realization_blocks": len(blocks),
                "replicates": args.replicates,
            }
        )

    payload = {
        "left": args.left_label,
        "right": args.right_label,
        "resampling_unit": "training seed and geological-factor-by-realization K block",
        "transport_settings_within_block": "averaged before bootstrap",
        "rng_seed": args.seed,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
