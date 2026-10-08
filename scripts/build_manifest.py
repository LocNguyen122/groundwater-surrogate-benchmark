"""Generate the deterministic SHA-256 release manifest."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST_CODE_RELEASE.csv"
IGNORED = {".git", ".pytest_cache", "__pycache__"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


files = [
    path for path in ROOT.rglob("*")
    if path.is_file() and path != OUTPUT
    and not (set(path.relative_to(ROOT).parts) & IGNORED)
    and not path.relative_to(ROOT).parts[0].startswith("reproduced")
    and path.suffix.lower() not in {".pyc", ".pyo"}
]
with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(("relative_path", "size_bytes", "sha256"))
    for path in sorted(files, key=lambda item: item.relative_to(ROOT).as_posix().lower()):
        writer.writerow((path.relative_to(ROOT).as_posix(), path.stat().st_size, sha256(path)))
print(f"Wrote {len(files)} entries to {OUTPUT}")
