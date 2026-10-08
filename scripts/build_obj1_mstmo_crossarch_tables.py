"""Build release tables from a passing MS-TMO cross-architecture certification."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certification", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    cert = json.loads(args.certification.read_text(encoding="utf-8"))
    if cert.get("status") != "PASS" or cert.get("certified_runs") != 10:
        raise ValueError("A passing 10-run certification is required")

    per_seed: list[dict[str, object]] = []
    for split, seed_rows in cert["per_seed"].items():
        for seed, metrics in sorted(seed_rows.items(), key=lambda item: int(item[0])):
            per_seed.append({
                "model": "MS-TMO-UNet",
                "conditioning": "factorized_levels",
                "split": split,
                "seed": int(seed),
                "global_ssim": metrics["global_ssim"],
                "plume_ssim": metrics["plume_ssim"],
                "n_cases": int(metrics["n_cases"]),
            })

    summary: list[dict[str, object]] = []
    for split, metrics in cert["summaries"].items():
        for metric, values in metrics.items():
            summary.append({
                "comparison": "factorized_absolute",
                "split": split,
                "metric": metric,
                "n_seeds": values["n"],
                "mean": values["mean"],
                "std": values["std"],
            })
    for metric, values in cert["paired_standard_v1_minus_raw_tc_baseline"].items():
        summary.append({
            "comparison": "factorized_minus_raw_tc",
            "split": "grouped_standard",
            "metric": metric,
            "n_seeds": values["n"],
            "mean": values["mean"],
            "std": values["std"],
        })

    write_csv(
        args.out_dir / "method_upgrade_mstmo_crossarch_per_seed.csv",
        ["model", "conditioning", "split", "seed", "global_ssim", "plume_ssim", "n_cases"],
        per_seed,
    )
    write_csv(
        args.out_dir / "method_upgrade_mstmo_crossarch_summary.csv",
        ["comparison", "split", "metric", "n_seeds", "mean", "std"],
        summary,
    )
    print(json.dumps({"status": "ok", "per_seed_rows": len(per_seed), "summary_rows": len(summary)}))


if __name__ == "__main__":
    main()
