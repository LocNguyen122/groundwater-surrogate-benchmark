"""Certify retrieved E1 CSVs against the terminal retrieval record, without inference."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def certify(root, status):
    if status.get("status") != "FETCHED_AND_VERIFIED" or status.get("failures"):
        raise ValueError("Successful terminal retrieval required")
    for job in ("3481420", "3485378"):
        if status["counts"][job] != {"COMPLETED": 250, "RUNNING": 0, "PENDING": 0}:
            raise ValueError("Incomplete array: " + job)
    manifest_path = root / "results/v7/E1_TERMINAL_ARTIFACTS_2026-10-08.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["task_sha256"] != status["task_manifest_sha256"]:
        raise ValueError("Task manifest mismatch")
    artifacts = manifest["artifacts"]
    if len(artifacts) != 1000 or len({r["path"] for r in artifacts}) != 1000:
        raise ValueError("Expected 1,000 unique remote CSV/JSON artifacts")
    counts = {"best": 0, "last": 0}
    for row in artifacts:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe artifact path")
        if relative.name != "per_case_metrics.csv":
            continue  # Original JSONs remain private because their protocol contains infrastructure paths.
        path = root / "results/v7/per_case/poolT_per_case" / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("CSV checksum mismatch: " + row["path"])
        counts[relative.parent.name] += 1
    if counts != {"best": 250, "last": 250}:
        raise ValueError("Incomplete checkpoint coverage")
    return {"status": "PASS", "arrays": status["counts"], "failures": [],
            "task_manifest_sha256": manifest["task_sha256"],
            "remote_artifacts_verified": 1000, "public_csvs_verified": counts,
            "cases_per_csv": 794, "checkpoint_existence_verified_at_retrieval": True,
            "scope": "Terminal scoring and retrieval, not a training or submission certificate",
            "original_jsons": "Validated and checksum-verified in the private lightweight mirror; not redistributed",
            "artifact_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new certification output")
    result = certify(ROOT, json.loads(args.status.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
