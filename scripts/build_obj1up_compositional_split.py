"""Build an isolated held-out transport-combination split for the Obj1 pilot.

The geological group assignment from the certified grouped-K split is retained.
One transport pair is omitted from optimization and validation, while the
existing held-out geological test groups retain all six transport pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path


PAIR_COLUMNS = ("long_dispersivity", "trans_ratio")
GROUP_COLUMNS = ("sigma2Y", "correlation_length", "anisotropy")
DEFAULT_HELDOUT = (6.2, 1.0)


def _pair(row: dict[str, str]) -> tuple[float, float]:
    return float(row[PAIR_COLUMNS[0]]), float(row[PAIR_COLUMNS[1]])


def build_compositional_split(
    base_split_path: Path,
    registry_path: Path,
    output_path: Path,
    heldout_pair: tuple[float, float] = DEFAULT_HELDOUT,
) -> dict[str, object]:
    base = json.loads(base_split_path.read_text(encoding="utf-8"))
    with registry_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {int(row["param_id"]): row for row in rows}
    if set(by_id) != set(range(162)):
        raise ValueError("Registry must contain exactly param_id 0 through 161.")

    payload = deepcopy(base)
    payload["schema_version"] = 2
    payload["name"] = "param_split_grouped_k_compositional_v1"
    payload["purpose"] = (
        "Analysis split for transport-combination generalization while preserving "
        "the certified grouped-K geological partitions."
    )
    payload["parent_split"] = str(base_split_path.as_posix())
    payload["compositional_holdout"] = {
        "long_dispersivity": heldout_pair[0],
        "trans_ratio": heldout_pair[1],
        "selection_basis": (
            "Prospective central alpha_L level and full-ratio level; selected without model outputs."
        ),
        "train_val_policy": "omit held-out transport pair without reassigning geological groups",
        "test_policy": "retain all six pairs in the existing held-out geological groups",
    }

    for split_name in ("train", "val"):
        source_ids = [int(value) for value in base["splits"][split_name]["param_ids"]]
        kept = [param_id for param_id in source_ids if _pair(by_id[param_id]) != heldout_pair]
        payload["splits"][split_name]["param_ids"] = kept
        payload["splits"][split_name]["param_folders"] = [
            f"param_{param_id:03d}" for param_id in kept
        ]
        payload["splits"][split_name]["omitted_heldout_pair_param_ids"] = sorted(
            set(source_ids) - set(kept)
        )

    test_ids = [int(value) for value in payload["splits"]["test"]["param_ids"]]
    payload["compositional_evaluation"] = {
        "heldout_pair_param_ids": [
            param_id for param_id in test_ids if _pair(by_id[param_id]) == heldout_pair
        ],
        "seen_pair_param_ids": [
            param_id for param_id in test_ids if _pair(by_id[param_id]) != heldout_pair
        ],
    }

    expected_counts = {"train": 90, "val": 15, "test": 36}
    observed_counts = {
        name: len(payload["splits"][name]["param_ids"])
        for name in ("train", "val", "test")
    }
    if observed_counts != expected_counts:
        raise AssertionError(f"Unexpected split counts: {observed_counts}")

    partition_ids = {
        name: set(map(int, payload["splits"][name]["param_ids"]))
        for name in ("train", "val", "test")
    }
    if any(
        partition_ids[left] & partition_ids[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise AssertionError("Parameter IDs overlap across partitions.")
    for split_name in ("train", "val"):
        if any(_pair(by_id[param_id]) == heldout_pair for param_id in partition_ids[split_name]):
            raise AssertionError(f"Held-out pair remains in {split_name}.")
    if not any(_pair(by_id[param_id]) == heldout_pair for param_id in partition_ids["test"]):
        raise AssertionError("Held-out pair is missing from test.")

    train_pairs = {_pair(by_id[param_id]) for param_id in partition_ids["train"]}
    if {pair[0] for pair in train_pairs} != {0.62, 6.2, 62.0}:
        raise AssertionError("Training split lacks alpha_L marginal coverage.")
    if {pair[1] for pair in train_pairs} != {0.1, 1.0}:
        raise AssertionError("Training split lacks ratio marginal coverage.")

    groups = {
        name: {
            tuple(float(row[column]) for column in GROUP_COLUMNS)
            for row in payload["splits"][name]["base_k_groups"]
        }
        for name in ("train", "val", "test")
    }
    if groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]:
        raise AssertionError("Geological group leakage detected.")
    if sum(map(len, groups.values())) != 27:
        raise AssertionError("Expected the original 27 geological groups.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-split", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout-alpha-l", type=float, default=DEFAULT_HELDOUT[0])
    parser.add_argument("--heldout-ratio", type=float, default=DEFAULT_HELDOUT[1])
    args = parser.parse_args()
    payload = build_compositional_split(
        args.base_split,
        args.registry,
        args.output,
        (args.heldout_alpha_l, args.heldout_ratio),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "counts": {
                    name: len(payload["splits"][name]["param_ids"])
                    for name in ("train", "val", "test")
                },
                "heldout_pair": payload["compositional_holdout"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

