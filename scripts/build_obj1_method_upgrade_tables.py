"""Build release-safe conditioning-study tables from retrieved run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


HELDOUT = {
    "compositional": (6.2, 1.0),
    "loco_0p62_0p1": (0.62, 0.1),
    "loco_0p62_1p0": (0.62, 1.0),
    "loco_6p2_0p1": (6.2, 0.1),
    "loco_62p0_0p1": (62.0, 0.1),
    "loco_62p0_1p0": (62.0, 1.0),
}
METRICS = ("global_ssim", "plume_ssim", "ice_per_timestep", "concentration_mae")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(rows: list[dict[str, str]], metric: str) -> float:
    values = [float(row[metric]) for row in rows]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Expected finite {metric} values")
    return statistics.mean(values)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Expected finite values, got {values}")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    registry = {
        row["param_folder"]: (
            float(row["long_dispersivity"]),
            float(row["trans_ratio"]),
        )
        for row in read_csv(args.registry)
    }

    per_seed: list[dict[str, object]] = []
    for split_dir in sorted(path for path in args.runs_root.iterdir() if path.is_dir()):
        split = split_dir.name
        target = HELDOUT.get(split)
        for variant_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for seed_dir in sorted(variant_dir.glob("seed*"), key=lambda path: int(path.name[4:])):
                cases = read_csv(seed_dir / "per_case_metrics.csv")
                heldout_cases = (
                    [row for row in cases if registry.get(row["param_folder"]) == target]
                    if target is not None
                    else []
                )
                row: dict[str, object] = {
                    "split": split,
                    "heldout_alpha_L": "" if target is None else target[0],
                    "heldout_ratio": "" if target is None else target[1],
                    "level_type": (
                        "standard"
                        if target is None
                        else "interior" if target[0] == 6.2 else "boundary"
                    ),
                    "variant": variant_dir.name,
                    "seed": int(seed_dir.name[4:]),
                    "n_all_cases": len(cases),
                    "n_heldout_cases": len(heldout_cases),
                }
                for metric in METRICS:
                    row[f"all_{metric}"] = finite_mean(cases, metric)
                    row[f"heldout_{metric}"] = (
                        finite_mean(heldout_cases, metric) if heldout_cases else ""
                    )
                per_seed.append(row)

    per_seed_fields = [
        "split",
        "heldout_alpha_L",
        "heldout_ratio",
        "level_type",
        "variant",
        "seed",
        "n_all_cases",
        "n_heldout_cases",
        *[
            f"{scope}_{metric}"
            for scope in ("all", "heldout")
            for metric in METRICS
        ],
    ]
    write_csv(
        args.output_dir / "method_upgrade_per_seed.csv",
        per_seed,
        per_seed_fields,
    )

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in per_seed:
        grouped[(str(row["split"]), str(row["variant"]))].append(row)
    summary: list[dict[str, object]] = []
    for (split, variant), rows in sorted(grouped.items()):
        first = rows[0]
        output: dict[str, object] = {
            "split": split,
            "heldout_alpha_L": first["heldout_alpha_L"],
            "heldout_ratio": first["heldout_ratio"],
            "level_type": first["level_type"],
            "variant": variant,
            "n_seeds": len(rows),
        }
        for scope in ("all", "heldout"):
            for metric in METRICS:
                values = [
                    float(row[f"{scope}_{metric}"])
                    for row in rows
                    if row[f"{scope}_{metric}"] != ""
                ]
                if values:
                    mean, std = mean_std(values)
                    output[f"{scope}_{metric}_mean"] = mean
                    output[f"{scope}_{metric}_std"] = std
                else:
                    output[f"{scope}_{metric}_mean"] = ""
                    output[f"{scope}_{metric}_std"] = ""
        summary.append(output)
    summary_fields = [
        "split",
        "heldout_alpha_L",
        "heldout_ratio",
        "level_type",
        "variant",
        "n_seeds",
        *[
            f"{scope}_{metric}_{stat}"
            for scope in ("all", "heldout")
            for metric in METRICS
            for stat in ("mean", "std")
        ],
    ]
    write_csv(
        args.output_dir / "method_upgrade_summary.csv",
        summary,
        summary_fields,
    )

    bootstrap = json.loads(args.bootstrap.read_text(encoding="utf-8"))
    holm = bootstrap["holm_family"]["results"]
    contrasts: list[dict[str, object]] = []
    for split, entry in bootstrap["pairs"].items():
        pair = entry["held_out_pair"]
        for contrast_name, metric_entries in entry["contrasts"].items():
            for metric, result in metric_entries.items():
                if "mean_difference" not in result:
                    continue
                row = {
                    "split": split,
                    "heldout_alpha_L": pair["alpha_L"],
                    "heldout_ratio": pair["alpha_T_over_alpha_L"],
                    "level_type": entry["level_type"],
                    "contrast": contrast_name,
                    "metric": metric,
                    **result,
                    "p_holm_v1_vs_v0_plume_family": "",
                    "holm_significant_at_0.05": "",
                }
                if (
                    contrast_name == "v1_factorized_levels_minus_v0_raw_channels"
                    and metric == "plume_ssim"
                ):
                    row["p_holm_v1_vs_v0_plume_family"] = holm[split]["p_holm"]
                    row["holm_significant_at_0.05"] = holm[split][
                        "significant_at_0.05"
                    ]
                contrasts.append(row)
    contrast_fields = [
        "split",
        "heldout_alpha_L",
        "heldout_ratio",
        "level_type",
        "contrast",
        "metric",
        "mean_difference",
        "ci95_lower",
        "ci95_upper",
        "p_value",
        "excludes_zero",
        "n_seeds",
        "n_blocks",
        "p_holm_v1_vs_v0_plume_family",
        "holm_significant_at_0.05",
    ]
    write_csv(
        args.output_dir / "method_upgrade_bootstrap_contrasts.csv",
        contrasts,
        contrast_fields,
    )
    print(
        json.dumps(
            {
                "per_seed_rows": len(per_seed),
                "summary_rows": len(summary),
                "contrast_rows": len(contrasts),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
