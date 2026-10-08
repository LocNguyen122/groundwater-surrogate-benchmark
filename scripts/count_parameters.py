from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))


def main() -> None:
    path = RELEASE_ROOT / "results" / "tables" / "parameter_counts.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    display_cols = [c for c in ["family", "params", "params_millions"] if c in df.columns]
    if not display_cols:
        raise ValueError(f"No parameter-count columns found in {path}")
    print(df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
