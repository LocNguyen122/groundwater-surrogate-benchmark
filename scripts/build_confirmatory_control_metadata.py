"""Generate matched and negative-control conditioning registries for v5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


FACTOR_COLUMNS = ("sigma2Y", "correlation_length", "anisotropy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="metadata/parameter_registry_162_v5.csv")
    parser.add_argument("--out-dir", default="metadata/confirmatory_controls_v5")
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["param_folder", "alpha_L", "alpha_T_ratio"]
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    registry_path = Path(args.registry)
    with registry_path.open("r", encoding="utf-8-sig", newline="") as handle:
        registry = list(csv.DictReader(handle))
    if len(registry) != 162:
        raise ValueError(f"Expected 162 registry rows, found {len(registry)}")

    by_factor: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in registry:
        by_factor[tuple(row[column] for column in FACTOR_COLUMNS)].append(row)
    if len(by_factor) != 27 or any(len(rows) != 6 for rows in by_factor.values()):
        raise ValueError("Registry must contain 27 geological cells with six transport settings each.")

    matched: list[dict[str, object]] = []
    permuted: list[dict[str, object]] = []
    constant: list[dict[str, object]] = []
    nuisance: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed)

    # A cyclic shift is a complete derangement and preserves both transport marginals.
    for factor in sorted(by_factor):
        rows = sorted(by_factor[factor], key=lambda row: int(row["param_id"]))
        values = [(float(row["long_dispersivity"]), float(row["trans_ratio"])) for row in rows]
        shifted = values[1:] + values[:1]
        for row, true_pair, shuffled_pair in zip(rows, values, shifted):
            folder = f"param_{int(row['param_id']):03d}"
            matched.append(
                {"param_folder": folder, "alpha_L": true_pair[0], "alpha_T_ratio": true_pair[1]}
            )
            permuted.append(
                {"param_folder": folder, "alpha_L": shuffled_pair[0], "alpha_T_ratio": shuffled_pair[1]}
            )
            constant.append(
                {
                    "param_folder": folder,
                    "alpha_L": 6.2,
                    "alpha_T_ratio": float(np.sqrt(0.1)),
                }
            )
            nuisance.append(
                {
                    "param_folder": folder,
                    "alpha_L": float(10.0 ** rng.normal(0.0, 1.0)),
                    "alpha_T_ratio": float(10.0 ** rng.normal(-0.5, 0.5)),
                }
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "matched": out_dir / "matched_transport_metadata.csv",
        "permuted": out_dir / "permuted_transport_metadata.csv",
        "constant_placebo": out_dir / "constant_placebo_metadata.csv",
        "independent_nuisance": out_dir / "independent_nuisance_metadata.csv",
    }
    write_csv(paths["matched"], sorted(matched, key=lambda row: row["param_folder"]))
    write_csv(paths["permuted"], sorted(permuted, key=lambda row: row["param_folder"]))
    write_csv(paths["constant_placebo"], sorted(constant, key=lambda row: row["param_folder"]))
    write_csv(paths["independent_nuisance"], sorted(nuisance, key=lambda row: row["param_folder"]))

    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "matched": "Authoritative transport coefficients.",
        "permuted": "Fixed cyclic derangement within each geological factor cell; transport marginals are preserved and no row keeps its original coefficient pair.",
        "constant_placebo": "Two constant channels. After training-split normalization both channels are identically zero.",
        "independent_nuisance": "Fixed continuous nuisance values sampled independently of all simulation factors in transformed space.",
        "files": {
            name: {"path": path.as_posix(), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }
    (out_dir / "CONTROL_METADATA_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
