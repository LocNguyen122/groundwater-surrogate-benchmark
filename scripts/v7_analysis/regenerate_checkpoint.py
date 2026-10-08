"""Recompute one checkpoint's Pool T analysis using only released per-case summaries."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from analyze_v7 import Data, pool_section

ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", choices=("best", "last"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Use a new output; historical results are preserved")
    args = SimpleNamespace(
        master_root=ROOT / "results/v7/per_case/master_test_per_case",
        poolT_root=ROOT / "results/v7/per_case/poolT_per_case",
        registry_v7=ROOT / "metadata/parameter_registry_162_v7.csv",
        fresh_registry=ROOT / "metadata/fresh_seed/param_registry_fresh_seed.csv",
        seed_classes=ROOT / "metadata/fresh_seed/randf_seed_classes.json",
        split=ROOT / "splits/param_split_grouped_k_v5.json",
        solver_qa=ROOT / "metadata/solver_qa_exclusions_v7.csv",
        setting_mean_csv=ROOT / "results/v7/per_case/setting_mean/setting_mean_A_poolT_per_case.csv")
    result = pool_section(Data(args), a.checkpoint)
    a.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print("Regenerated", a.checkpoint, "using released per-case summaries")
