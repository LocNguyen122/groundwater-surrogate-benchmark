"""Build the confirmatory exclusion manifest for incomplete six-way K groups."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-csv",
        default="results/audit_v5/k_hash_audit_per_file.csv",
    )
    parser.add_argument(
        "--out-csv",
        default="metadata/k_integrity_exclusions_v5.csv",
    )
    parser.add_argument(
        "--out-json",
        default="results/audit_v5/k_integrity_exclusions_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_path = Path(args.audit_csv)
    with audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    expected_transport_pairs = {
        ("0.62", "0.1"),
        ("0.62", "1.0"),
        ("6.2", "0.1"),
        ("6.2", "1.0"),
        ("62.0", "0.1"),
        ("62.0", "1.0"),
    }
    by_factor_realization: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["sigma2Y"],
            row["correlation_length"],
            row["anisotropy"],
            row["realization"],
        )
        by_factor_realization[key].append(row)

    exclusions: list[dict[str, str]] = []
    affected_groups: list[dict[str, object]] = []
    for key, group in sorted(by_factor_realization.items()):
        transport_pairs = {(row["long_dispersivity"], row["trans_ratio"]) for row in group}
        hashes = {row["k_sha256"] for row in group}
        complete = len(group) == 6 and transport_pairs == expected_transport_pairs and len(hashes) == 1
        if complete:
            continue
        reason = "incomplete_or_nonidentical_six_way_K_group"
        affected_groups.append(
            {
                "sigma2Y": key[0],
                "correlation_length": key[1],
                "anisotropy": key[2],
                "realization": key[3],
                "n_files": len(group),
                "n_unique_k_hashes": len(hashes),
                "param_ids": sorted(int(row["param_id"]) for row in group),
            }
        )
        for row in group:
            exclusions.append(
                {
                    "sample_key": f"{row['param_folder']}/{row['realization']}.npz",
                    "param_id": row["param_id"],
                    "param_folder": row["param_folder"],
                    "realization": row["realization"],
                    "k_sha256": row["k_sha256"],
                    "reason": reason,
                }
            )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_key",
        "param_id",
        "param_folder",
        "realization",
        "k_sha256",
        "reason",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(exclusions, key=lambda row: row["sample_key"]))

    payload = {
        "rule": "Exclude every file in a geological-factor/realization group unless all six transport configurations share one exact K hash.",
        "n_affected_groups": len(affected_groups),
        "n_excluded_files": len(exclusions),
        "n_retained_files": len(rows) - len(exclusions),
        "affected_groups": affected_groups,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
