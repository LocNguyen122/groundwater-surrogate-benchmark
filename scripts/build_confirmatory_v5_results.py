"""Build lightweight v5 confirmatory tables from certified per-case artifacts."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


FAMILIES = {
    "cta_ffl_k_only": "CTA-UNet+FFL K-only",
    "cta_ffl_matched": "CTA-UNet+FFL matched codes",
    "cta_ffl_permuted": "CTA-UNet+FFL cyclic relabeling",
    "cta_ffl_constant_placebo": "CTA-UNet+FFL constant placebo",
    "cta_ffl_independent_nuisance": "CTA-UNet+FFL nuisance placebo",
    "ms_tmo_matched": "MS-TMO-UNet matched codes",
}

METRICS = (
    "global_ssim",
    "plume_ssim",
    "ice_per_timestep",
    "concentration_mae",
)

COUNTERFACTUAL_TAGS = (
    "fixed_alphaL_0p62_ratio_0p1",
    "fixed_alphaL_0p62_ratio_1p0",
    "fixed_alphaL_6p2_ratio_0p1",
    "fixed_alphaL_6p2_ratio_1p0",
    "fixed_alphaL_62p0_ratio_0p1",
    "fixed_alphaL_62p0_ratio_1p0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--matched-metadata", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty list.")
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("Sample SD requires at least two values.")
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def case_key(row: dict[str, str]) -> tuple[str, str]:
    return row["param_folder"], row["realization"]


def summarize_base(run_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_seed: list[dict[str, object]] = []
    for family, display_name in FAMILIES.items():
        family_root = run_root / family
        seed_dirs = sorted(family_root.glob("seed*"), key=lambda path: int(path.name[4:]))
        seeds = [int(path.name[4:]) for path in seed_dirs]
        if seeds != list(range(10)):
            raise ValueError(f"{family} requires exactly seeds 0..9; found {seeds}")
        for seed, seed_dir in zip(seeds, seed_dirs):
            rows = read_csv(seed_dir / "per_case_metrics.csv")
            if len(rows) != 174:
                raise ValueError(f"Expected 174 cases in {seed_dir}; found {len(rows)}")
            record: dict[str, object] = {
                "family": family,
                "display_name": display_name,
                "seed": seed,
                "n_cases": len(rows),
            }
            for metric in METRICS:
                values = [float(row[metric]) for row in rows]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Non-finite {metric} in {seed_dir}")
                record[metric] = mean(values)
            per_seed.append(record)

    summary: list[dict[str, object]] = []
    for family, display_name in FAMILIES.items():
        family_rows = [row for row in per_seed if row["family"] == family]
        record = {"family": family, "display_name": display_name, "n_seeds": 10}
        for metric in METRICS:
            values = [float(row[metric]) for row in family_rows]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_sd"] = sample_sd(values)
        summary.append(record)
    return per_seed, summary


def summarize_optimization(run_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family, display_name in FAMILIES.items():
        skipped_by_seed: list[int] = []
        epochs_with_skips = 0
        last_skip_epoch = 0
        for seed in range(10):
            log_rows = read_csv(run_root / family / f"seed{seed}" / "train_log.csv")
            if len(log_rows) != 200:
                raise ValueError(f"Expected 200 epochs for {family} seed {seed}")
            skipped = [int(row["skipped"]) for row in log_rows]
            skipped_by_seed.append(sum(skipped))
            epochs_with_skips += sum(value > 0 for value in skipped)
            last_skip_epoch = max(
                last_skip_epoch,
                max((int(row["epoch"]) for row in log_rows if int(row["skipped"]) > 0), default=0),
            )
        scheduled = 10 * 200 * 33
        total_skipped = sum(skipped_by_seed)
        rows.append(
            {
                "family": family,
                "display_name": display_name,
                "n_seeds": 10,
                "scheduled_batches": scheduled,
                "skipped_batches": total_skipped,
                "skip_rate_percent": 100.0 * total_skipped / scheduled,
                "affected_seeds": sum(value > 0 for value in skipped_by_seed),
                "epochs_with_skips": epochs_with_skips,
                "last_skip_epoch": last_skip_epoch,
            }
        )
    return rows


def metadata_tags(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in read_csv(path):
        alpha_l = str(float(row["alpha_L"])).replace(".", "p")
        ratio = str(float(row["alpha_T_ratio"])).replace(".", "p")
        mapping[row["param_folder"]] = f"fixed_alphaL_{alpha_l}_ratio_{ratio}"
    return mapping


def summarize_counterfactual(
    run_root: Path, matched_metadata: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    true_tag = metadata_tags(matched_metadata)
    per_seed: list[dict[str, object]] = []
    matrix_cells: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for seed in range(10):
        seed_root = run_root / "cta_ffl_matched" / f"seed{seed}"
        base_rows = {case_key(row): row for row in read_csv(seed_root / "per_case_metrics.csv")}
        supplied: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
        for tag in COUNTERFACTUAL_TAGS:
            rows = read_csv(seed_root / "counterfactual_sweep" / f"{tag}_per_case.csv")
            supplied[tag] = {case_key(row): row for row in rows}
            if set(supplied[tag]) != set(base_rows):
                raise ValueError(f"Counterfactual case mismatch for seed {seed}, tag {tag}")

        correct_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
        wrong_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
        exact_match_max_abs = 0.0
        for key, base in base_rows.items():
            tag_true = true_tag[key[0]]
            if tag_true not in supplied:
                raise ValueError(f"Unknown true transport tag for {key[0]}: {tag_true}")
            for metric in METRICS:
                base_value = float(base[metric])
                replay_value = float(supplied[tag_true][key][metric])
                exact_match_max_abs = max(exact_match_max_abs, abs(base_value - replay_value))
                correct_values[metric].append(base_value)
                for tag in COUNTERFACTUAL_TAGS:
                    value = float(supplied[tag][key][metric])
                    matrix_cells[(tag_true, tag, metric)].append(value)
                    if tag != tag_true:
                        wrong_values[metric].append(value)
        if exact_match_max_abs > 1e-12:
            raise ValueError(
                f"Matched-code replay differs from base evaluation for seed {seed}: "
                f"max_abs={exact_match_max_abs}"
            )
        record: dict[str, object] = {
            "seed": seed,
            "n_cases": len(base_rows),
            "n_wrong_assignments_per_case": 5,
            "matching_replay_max_abs_difference": exact_match_max_abs,
        }
        for metric in METRICS:
            correct = mean(correct_values[metric])
            wrong = mean(wrong_values[metric])
            direction = 1.0 if metric.endswith("ssim") else -1.0
            record[f"{metric}_correct"] = correct
            record[f"{metric}_wrong_mean"] = wrong
            record[f"{metric}_correct_advantage"] = direction * (correct - wrong)
        per_seed.append(record)

    summary: list[dict[str, object]] = []
    for metric in METRICS:
        values = [float(row[f"{metric}_correct_advantage"]) for row in per_seed]
        summary.append(
            {
                "metric": metric,
                "direction": "positive favors correct code",
                "mean_correct_advantage": mean(values),
                "sd_correct_advantage": sample_sd(values),
                "n_seeds": 10,
                "n_cases_per_seed": 174,
                "wrong_codes_per_case": 5,
            }
        )

    matrix: list[dict[str, object]] = []
    for tag_true in COUNTERFACTUAL_TAGS:
        for tag_supplied in COUNTERFACTUAL_TAGS:
            record: dict[str, object] = {
                "true_code": tag_true,
                "supplied_code": tag_supplied,
                "is_correct": tag_true == tag_supplied,
            }
            for metric in METRICS:
                record[metric] = mean(matrix_cells[(tag_true, tag_supplied, metric)])
            matrix.append(record)
    return per_seed, summary, matrix


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    base_per_seed, base_summary = summarize_base(run_root)
    optimization = summarize_optimization(run_root)
    cf_per_seed, cf_summary, cf_matrix = summarize_counterfactual(
        run_root, Path(args.matched_metadata)
    )

    write_csv(out_dir / "confirmatory_v5_per_seed.csv", list(base_per_seed[0]), base_per_seed)
    write_csv(out_dir / "confirmatory_v5_summary.csv", list(base_summary[0]), base_summary)
    write_csv(out_dir / "confirmatory_v5_optimization_stability.csv", list(optimization[0]), optimization)
    write_csv(out_dir / "confirmatory_v5_counterfactual_per_seed.csv", list(cf_per_seed[0]), cf_per_seed)
    write_csv(out_dir / "confirmatory_v5_counterfactual_summary.csv", list(cf_summary[0]), cf_summary)
    write_csv(out_dir / "confirmatory_v5_counterfactual_matrix.csv", list(cf_matrix[0]), cf_matrix)
    print(f"Wrote confirmatory v5 tables to {out_dir}")


if __name__ == "__main__":
    main()
