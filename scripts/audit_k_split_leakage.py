"""Audit exact and near-duplicate hydraulic-conductivity fields across splits.

This script reads only the ``K`` array from each simulation NPZ. It records a
SHA-256 digest of the canonical array bytes, joins the full five-factor
parameter registry, and reports whether any conductivity realization occurs in
more than one partition. It does not modify the dataset or a split file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def load_split(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, str] = {}
    for split_name in ("train", "val", "test"):
        for param_id in payload["splits"][split_name]["param_ids"]:
            result[int(param_id)] = split_name
    return result


def load_registry(path: Path) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("dataset") == "training"]
    registry = {int(row["param_id"]): row for row in rows}
    if set(registry) != set(range(162)):
        raise ValueError("The training registry must contain param_id 0 through 161 exactly once.")
    return registry


def audit_one(task: tuple[str, int, str, dict[str, str]]) -> tuple[dict[str, object], np.ndarray]:
    path_text, param_id, split_name, metadata = task
    path = Path(path_text)
    with np.load(path) as archive:
        conductivity = np.ascontiguousarray(archive["K"].astype(np.float32, copy=False))
    digest = hashlib.sha256(conductivity.view(np.uint8)).hexdigest()
    row: dict[str, object] = {
        "param_id": param_id,
        "param_folder": f"param_{param_id:03d}",
        "realization": path.stem,
        "historical_split": split_name,
        "sigma2Y": metadata["sigma2Y"],
        "correlation_length": metadata["correlation_length"],
        "anisotropy": metadata["anisotropy"],
        "long_dispersivity": metadata["long_dispersivity"],
        "trans_ratio": metadata["trans_ratio"],
        "k_shape": "x".join(str(value) for value in conductivity.shape),
        "k_dtype": str(conductivity.dtype),
        "k_min": float(conductivity.min()),
        "k_max": float(conductivity.max()),
        "k_mean": float(conductivity.mean(dtype=np.float64)),
        "k_std": float(conductivity.std(dtype=np.float64)),
        "k_sha256": digest,
    }
    return row, conductivity


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def near_duplicate_rows(representatives: dict[str, tuple[dict[str, object], np.ndarray]]) -> list[dict[str, object]]:
    items = list(representatives.items())
    rows: list[dict[str, object]] = []
    for left_index, (left_hash, (left_meta, left)) in enumerate(items):
        left64 = left.astype(np.float64, copy=False)
        left_centered = left64 - left64.mean()
        left_norm = float(np.linalg.norm(left_centered))
        for right_hash, (right_meta, right) in items[left_index + 1 :]:
            right64 = right.astype(np.float64, copy=False)
            delta = left64 - right64
            scale = max(float(left64.std()), float(right64.std()), 1e-12)
            nrmse = float(np.sqrt(np.mean(delta * delta)) / scale)
            right_centered = right64 - right64.mean()
            denominator = left_norm * float(np.linalg.norm(right_centered))
            correlation = float(np.sum(left_centered * right_centered) / denominator) if denominator else 1.0
            rows.append(
                {
                    "left_hash": left_hash,
                    "right_hash": right_hash,
                    "left_param_id": left_meta["param_id"],
                    "right_param_id": right_meta["param_id"],
                    "left_realization": left_meta["realization"],
                    "right_realization": right_meta["realization"],
                    "normalized_rmse": nrmse,
                    "pearson_correlation": correlation,
                }
            )
    rows.sort(key=lambda row: (float(row["normalized_rmse"]), -float(row["pearson_correlation"])))
    return rows[:100]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    registry = load_registry(args.registry)
    split_by_id = load_split(args.split)
    tasks: list[tuple[str, int, str, dict[str, str]]] = []
    for param_id in range(162):
        folder = args.data_root / f"param_{param_id:03d}"
        files = sorted(folder.glob("real_*.npz"))
        if len(files) != 5:
            raise FileNotFoundError(f"Expected five realizations under {folder}, found {len(files)}")
        tasks.extend((str(path), param_id, split_by_id[param_id], registry[param_id]) for path in files)

    rows: list[dict[str, object]] = []
    representatives: dict[str, tuple[dict[str, object], np.ndarray]] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for row, conductivity in pool.map(audit_one, tasks, chunksize=2):
            rows.append(row)
            representatives.setdefault(str(row["k_sha256"]), (row, conductivity))
    rows.sort(key=lambda row: (int(row["param_id"]), str(row["realization"])))

    by_hash: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_hash.setdefault(str(row["k_sha256"]), []).append(row)
    groups = []
    for digest, members in by_hash.items():
        splits = sorted({str(member["historical_split"]) for member in members})
        groups.append(
            {
                "k_sha256": digest,
                "n_files": len(members),
                "n_param_ids": len({int(member["param_id"]) for member in members}),
                "param_ids": ";".join(str(member["param_id"]) for member in members),
                "realizations": ";".join(sorted({str(member["realization"]) for member in members})),
                "historical_splits": ";".join(splits),
                "cross_split": len(splits) > 1,
                "sigma2Y": members[0]["sigma2Y"],
                "correlation_length": members[0]["correlation_length"],
                "anisotropy": members[0]["anisotropy"],
            }
        )
    groups.sort(key=lambda row: (float(row["sigma2Y"]), float(row["correlation_length"]), float(row["anisotropy"]), str(row["realizations"])))
    near = near_duplicate_rows(representatives)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "k_hash_audit_per_file.csv", rows)
    write_csv(args.out_dir / "k_hash_groups.csv", groups)
    write_csv(args.out_dir / "k_nearest_distinct_pairs.csv", near)
    summary = {
        "n_files": len(rows),
        "n_unique_k_hashes": len(by_hash),
        "expected_files_per_exact_k": sorted({len(members) for members in by_hash.values()}),
        "n_exact_k_groups_crossing_historical_splits": sum(bool(row["cross_split"]) for row in groups),
        "n_exact_k_groups_spanning_all_three_historical_splits": sum(row["historical_splits"] == "test;train;val" for row in groups),
        "historical_split_has_exact_k_leakage": any(bool(row["cross_split"]) for row in groups),
        "closest_distinct_pair_normalized_rmse": near[0]["normalized_rmse"],
        "closest_distinct_pair_pearson_correlation": near[0]["pearson_correlation"],
        "hash_definition": "SHA-256 of contiguous float32 K-array bytes in stored orientation",
        "near_duplicate_definition": "normalized RMSE divided by the larger field SD; Pearson correlation is also reported",
    }
    (args.out_dir / "k_split_leakage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
