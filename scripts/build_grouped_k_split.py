"""Build a deterministic K-grouped, factor-stratified confirmatory split.

The 162 configurations form 27 base conductivity groups defined by
``(sigma2Y, correlation_length, anisotropy)``. Each group contains all six
``(long_dispersivity, trans_ratio)`` combinations. This builder assigns whole
base groups to one partition, preventing identical K realizations from crossing
partitions while balancing every factor level in validation and test.

The historical split is not modified.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


BASE_COLUMNS = ("sigma2Y", "correlation_length", "anisotropy")
TRANSPORT_COLUMNS = ("long_dispersivity", "trans_ratio")


def numeric_tuple(row: dict[str, str], columns: tuple[str, ...]) -> tuple[float, ...]:
    return tuple(float(row[column]) for column in columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--split-out", type=Path, required=True)
    parser.add_argument("--registry-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    with args.registry.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("dataset") == "training"]
    if len(rows) != 162 or {int(row["param_id"]) for row in rows} != set(range(162)):
        raise ValueError("Expected the 162 training configurations with param_id 0 through 161.")

    groups: dict[tuple[float, float, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[numeric_tuple(row, BASE_COLUMNS)].append(row)
    if len(groups) != 27 or {len(members) for members in groups.values()} != {6}:
        raise ValueError("Expected 27 conductivity groups with six transport configurations each.")
    expected_transport = {
        (0.62, 0.1), (0.62, 1.0), (6.2, 0.1),
        (6.2, 1.0), (62.0, 0.1), (62.0, 1.0),
    }
    for key, members in groups.items():
        observed = {numeric_tuple(row, TRANSPORT_COLUMNS) for row in members}
        if observed != expected_transport:
            raise ValueError(f"Incomplete transport factorial for base group {key}: {observed}")

    levels = {column: sorted({float(row[column]) for row in rows}) for column in BASE_COLUMNS}
    if any(len(values) != 3 for values in levels.values()):
        raise ValueError(f"Expected three levels per conductivity factor: {levels}")
    indexed = {
        (
            levels["sigma2Y"].index(key[0]),
            levels["correlation_length"].index(key[1]),
            levels["anisotropy"].index(key[2]),
        ): key
        for key in groups
    }
    val_indices = {(0, 0, 0), (1, 1, 1), (2, 2, 2)}
    test_indices = {(0, 0, 1), (1, 1, 2), (2, 2, 0), (0, 1, 2), (1, 2, 0), (2, 0, 1)}
    assignment: dict[tuple[float, float, float], str] = {}
    for index, key in indexed.items():
        assignment[key] = "val" if index in val_indices else "test" if index in test_indices else "train"

    split_payload: dict[str, object] = {
        "schema_version": 1,
        "name": "param_split_grouped_k_v5",
        "purpose": "Confirmatory split that prevents exact K-field reuse across partitions.",
        "grouping_key": list(BASE_COLUMNS),
        "selection_rule": {
            "description": "Whole 3x3x3 conductivity-factor cells are assigned without using model outputs. Validation uses the main diagonal; test uses two balanced off-diagonal Latin patterns; remaining cells are training.",
            "validation_factor_indices": sorted([list(item) for item in val_indices]),
            "test_factor_indices": sorted([list(item) for item in test_indices]),
        },
        "splits": {},
    }
    export_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {"grouping_key": list(BASE_COLUMNS), "splits": {}}
    for split_name in ("train", "val", "test"):
        selected_keys = sorted(key for key, value in assignment.items() if value == split_name)
        selected_rows = sorted(
            (row for key in selected_keys for row in groups[key]),
            key=lambda row: int(row["param_id"]),
        )
        param_ids = [int(row["param_id"]) for row in selected_rows]
        split_payload["splits"][split_name] = {
            "param_ids": param_ids,
            "param_folders": [f"param_{param_id:03d}" for param_id in param_ids],
            "base_k_groups": [
                {"sigma2Y": key[0], "correlation_length": key[1], "anisotropy": key[2]}
                for key in selected_keys
            ],
        }
        factor_counts = {
            column: dict(sorted(Counter(str(key[index]) for key in selected_keys).items()))
            for index, column in enumerate(BASE_COLUMNS)
        }
        transport_counts = Counter(
            f"{row['long_dispersivity']}|{row['trans_ratio']}" for row in selected_rows
        )
        summary["splits"][split_name] = {
            "n_base_k_groups": len(selected_keys),
            "n_param_configurations": len(selected_rows),
            "n_field_files_expected": len(selected_rows) * 5,
            "base_factor_counts": factor_counts,
            "transport_combination_counts": dict(sorted(transport_counts.items())),
        }
        for row in selected_rows:
            export_rows.append(
                {
                    "param_folder": f"param_{int(row['param_id']):03d}",
                    "param_id": int(row["param_id"]),
                    "split_v5": split_name,
                    "k_group_id": f"s{row['sigma2Y']}_l{row['correlation_length']}_a{row['anisotropy']}",
                    "sigma2Y": row["sigma2Y"],
                    "correlation_length": row["correlation_length"],
                    "anisotropy": row["anisotropy"],
                    "long_dispersivity": row["long_dispersivity"],
                    "trans_ratio": row["trans_ratio"],
                }
            )

    args.split_out.parent.mkdir(parents=True, exist_ok=True)
    args.registry_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.split_out.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
    with args.registry_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(export_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(export_rows, key=lambda row: int(row["param_id"])))
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
