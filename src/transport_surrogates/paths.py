from __future__ import annotations

from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = RELEASE_ROOT / "configs"
SPLIT_ROOT = RELEASE_ROOT / "splits"
RESULT_TABLE_ROOT = RELEASE_ROOT / "results" / "tables"
RUNS_ROOT = RELEASE_ROOT / "runs"


def resolve_release_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return RELEASE_ROOT / p
