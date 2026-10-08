"""Validate current summaries, portable inputs and both terminal checkpoint sets."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate(root: Path = ROOT) -> dict:
    config_path = root / "scripts/v7_analysis/inputs_v7.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inputs = dict(config["json"])
    inputs.update({k: v for k, v in config.items() if k != "json"})
    for key, value in inputs.items():
        if Path(value).is_absolute() or ":" in value or "\\" in value:
            raise ValueError(f"Nonportable input path: {key}")
        path = (config_path.parent / value).resolve()
        if root.resolve() not in path.parents or not path.is_file():
            raise ValueError(f"Missing or out-of-package input: {key}")
    analysis = json.loads((config_path.parent / config["json"]["poolT"]).read_text())
    secondary = analysis["O3_compositional"]["loco_secondary"]
    if len(secondary) != 5 or any(r["n_seeds"] != 5 or r["n_cells"] != 16 for r in secondary.values()):
        raise ValueError("Incomplete five-pair independent-geology screen")
    for row in secondary.values():
        for key in ("estimate", "ci_low", "ci_high", "p_holm"):
            if not math.isfinite(row[key]):
                raise ValueError(f"Nonfinite secondary contrast: {key}")
        if row["ci_low"] > row["ci_high"] or not 0 <= row["p_holm"] <= 1:
            raise ValueError("Invalid interval or adjusted probability")
    files = sorted((root / "results/v7/per_case/poolT_per_case").rglob("per_case_metrics.csv"))
    best = [p for p in files if p.parent.name == "best" and p.parent.parent.name == "e1_fresh_seed_pool_T"]
    last = [p for p in files if p.parent.name == "last" and p.parent.parent.name == "e1_fresh_seed_pool_T"]
    if len(best) != 250 or len(last) != 250:
        raise ValueError(f"Pool T coverage: best {len(best)}/250, last {len(last)}/250")
    certificate = json.loads((root / "results/v7/E1_TERMINAL_CERTIFICATION_2026-10-08.json").read_text())
    from certify_e1_terminal import certify
    record = {"status": "FETCHED_AND_VERIFIED", "failures": certificate["failures"],
              "counts": certificate["arrays"], "task_manifest_sha256": certificate["task_manifest_sha256"]}
    if certificate != certify(root, record):
        raise ValueError("Terminal certificate differs from current artifacts")
    for checkpoint in ("poolT", "last"):
        current = json.loads((config_path.parent / config["json"][checkpoint]).read_text())
        if current["O3_compositional"]["primary_v1_minus_v0"]["n_boot"] != 10000:
            raise ValueError("10,000 bootstrap replicates required")
    for path in best + last:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 794:
            raise ValueError(f"Incomplete per-case CSV: {path.relative_to(root)} ({len(rows)})")
        keys = {(r["param_folder"], r["realization"]) for r in rows}
        if len(keys) != 794:
            raise ValueError(f"Duplicate case keys: {path.relative_to(root)}")
        for row in rows:
            for metric in ("global_ssim", "plume_ssim"):
                if not math.isfinite(float(row[metric])):
                    raise ValueError(f"Nonfinite {metric}: {path.relative_to(root)}")
    return {"status": "ok", "portable_inputs": len(inputs), "best_poolT_csvs": len(best),
            "last_poolT_csvs": len(last), "secondary_pairs": len(secondary), "final_epoch_certified": True,
            "note": "Terminal scoring validated; author data/license/venue decisions are separate."}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
