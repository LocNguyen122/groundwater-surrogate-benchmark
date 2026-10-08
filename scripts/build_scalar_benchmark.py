from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))


def main() -> None:
    path = RELEASE_ROOT / "results" / "tables" / "scalar_benchmark_per_seed.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    group_cols = ["family", "regime"]
    metrics = ["global_ssim", "plume_ssim", "mass_err", "physics_l1"]
    summary = df.groupby(group_cols)[metrics].agg(["count", "mean", "std"])
    rows = []
    for (family, regime), values in summary.iterrows():
        row = {"family": family, "regime": regime, "n": int(values[("global_ssim", "count")])}
        for metric in metrics:
            row[f"{metric}_mean"] = values[(metric, "mean")]
            row[f"{metric}_std"] = values[(metric, "std")]
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["family", "regime"])
    out = out.rename(columns={"mass_err_mean": "ICE_t_mean", "mass_err_std": "ICE_t_std"})
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
