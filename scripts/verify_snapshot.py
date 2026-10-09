"""Verify the exact allowlisted snapshot, source lineage, syntax and numerical coverage."""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", "__pycache__", ".venv", ".pytest_cache"}
ALLOWED_SUFFIXES = {".py", ".csv", ".json", ".yaml", ".yml", ".md", ".txt", ".cff"}
SPECIAL_NAMES = {".gitignore", ".gitattributes", "LICENSE"}
FORBIDDEN_PARTS = {".claude", ".codex", "simulation", "slurm", "raapoi", "checkpoints"}
RESTRICTED_TEXT = re.compile(
    r"/nfs/scratch|ecs-gateway|nguyenk1@|[A-Za-z]:[\\/]Users[\\/]|V:[\\/]|PhD_Project|"
    r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lf_text_sha256(data: bytes) -> str:
    """Normalize line endings only; numerical values and all other bytes stay significant."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def check_path(relative: Path) -> None:
    if relative.is_absolute() or ".." in relative.parts or set(relative.parts) & FORBIDDEN_PARTS or relative.parts[0] == "data":
        raise ValueError("Restricted snapshot path: " + relative.as_posix())
    if relative.suffix not in ALLOWED_SUFFIXES and relative.name not in SPECIAL_NAMES:
        raise ValueError("Disallowed file extension: " + relative.as_posix())


def validate_license(root: Path, status: dict) -> str | None:
    """Do not confuse a selected license with a verified, effective grant."""
    active = status.get("open_source_license")
    path = root / "LICENSE"
    if active is None:
        if path.exists():
            raise ValueError("LICENSE exists but no effective license is declared")
        return None
    if active != "MIT":
        raise ValueError("This release guard currently supports confirmed MIT grants only")
    if not status.get("original_code_licensing_authority_confirmed", False):
        raise ValueError("MIT selected, but original-code licensing authority is unconfirmed")
    if not path.is_file():
        raise ValueError("MIT declared without its LICENSE text")
    text = path.read_text(encoding="utf-8-sig")
    required = ("MIT License", "Copyright (c)", "Permission is hereby granted, free of charge",
                "The above copyright notice and this permission notice",
                'THE SOFTWARE IS PROVIDED "AS IS"')
    if not all(token in text for token in required):
        raise ValueError("Incomplete MIT text")
    if re.search(r"\[(?:year|fullname|copyright holder|confirmed[^\]]*)\]|TBD|PENDING", text, re.I):
        raise ValueError("Unresolved MIT copyright-holder placeholder")
    return active


def verify() -> dict:
    manifest = ROOT / "MANIFEST_CODE_RELEASE.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig", newline="")))
    expected = {r["relative_path"] for r in rows}
    if len(expected) != len(rows):
        raise ValueError("Duplicate manifest entries")
    actual = set()
    parsed = 0
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if not path.is_file() or set(rel.parts) & IGNORED or rel.parts[0].startswith("reproduced"):
            continue
        if path.suffix in {".pyc", ".pyo"} or path == manifest:
            continue
        check_path(rel)
        text = path.read_text(encoding="utf-8-sig")
        # The scanner's own pattern definitions describe restricted strings, not operational values.
        if rel.as_posix() != "scripts/verify_snapshot.py" and RESTRICTED_TEXT.search(text):
            raise ValueError("Restricted text detected in " + rel.as_posix())
        if path.suffix == ".py":
            ast.parse(text, filename=rel.as_posix())
            parsed += 1
        actual.add(rel.as_posix())
    if actual != expected:
        raise ValueError(f"Unmanifested or missing files: {sorted(actual ^ expected)}")
    for row in rows:
        path = ROOT / row["relative_path"]
        if path.stat().st_size != int(row["size_bytes"]) or digest(path) != row["sha256"]:
            raise ValueError("Manifest mismatch: " + row["relative_path"])
    changes = json.loads((ROOT / "provenance/PACKAGING_RECORD.json").read_text())
    changed = set(changes["snapshot_only_edits"])
    sources = list(csv.DictReader((ROOT / "provenance/SOURCE_SNAPSHOT.csv").open(encoding="utf-8-sig")))
    for row in sources:
        path = ROOT / row["snapshot_relative_path"]
        if row["snapshot_relative_path"] not in changed and digest(path) != row["source_sha256"]:
            raise ValueError("Unrecorded source modification: " + row["snapshot_relative_path"])
    status = json.loads((ROOT / "RELEASE_STATUS.json").read_text())
    active_license = validate_license(ROOT, status)
    return {"status": "PASS", "manifest_entries": len(rows), "python_asts": parsed,
            "source_files": len(sources), "license": active_license or "unassigned",
            "selected_license": status.get("selected_original_code_license"),
            "final_paper_release": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
