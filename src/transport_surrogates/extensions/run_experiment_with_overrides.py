from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from src.transport_surrogates.extensions.registry import (
    DATA_ROOT,
    EXPERIMENT_RUNS_ROOT,
    RELEASE_ROOT,
    SPLIT_JSON,
    STATS_JSON,
    get_experiment,
)

BASELINE_UPGRADE_KEYS = {"deeponet_canonical_baseline", "fno_replicate_baseline"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch transport-conditioning benchmark experiments with external-output output overrides.")
    parser.add_argument("--experiment-key", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=320)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--split-json", type=str, default="")
    parser.add_argument("--stats-json", type=str, default="")
    parser.add_argument("--out-root", type=str, default="")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_command(command: List[str], cwd: Path, dry_run: bool) -> None:
    print(json.dumps({"cwd": str(cwd), "command": command}, indent=2))
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd), check=True)


def resolve_path(value: str, default: Path) -> Path:
    path = Path(value) if value else default
    if not path.is_absolute():
        path = RELEASE_ROOT / path
    return path


def build_train_command(args: argparse.Namespace, run_family_dir: Path, run_name: str) -> Dict[str, object]:
    spec = get_experiment(args.experiment_key)
    launch = spec.launch
    data_root = resolve_path(args.data_root, DATA_ROOT)
    split_json = resolve_path(args.split_json, SPLIT_JSON)
    stats_json = resolve_path(args.stats_json, STATS_JSON)
    common_args = [
        "--seed", str(args.seed),
        "--data_root", str(data_root),
        "--split_json", str(split_json),
        "--stats_json", str(stats_json),
        "--out_root", str(run_family_dir),
        "--run_name", run_name,
        "--patch_size", str(args.patch_size),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.learning_rate),
    ]
    if launch.launch_type == "module":
        command = [sys.executable, "-m", launch.target, *common_args, *launch.args]
    else:
        command = [sys.executable, launch.target, *common_args, *launch.args]
    return {"cwd": launch.cwd, "command": command}


def build_eval_command(args: argparse.Namespace, run_seed_dir: Path) -> Dict[str, object]:
    checkpoint_path = run_seed_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for evaluation: {checkpoint_path}")
    data_root = resolve_path(args.data_root, DATA_ROOT)
    split_json = resolve_path(args.split_json, SPLIT_JSON)
    stats_json = resolve_path(args.stats_json, STATS_JSON)

    eval_module = (
        "src.transport_surrogates.extensions.eval.eval_transport_baseline_upgrades"
        if args.experiment_key in BASELINE_UPGRADE_KEYS
        else "src.shared.eval.eval_transport_unified_logc_global_plumemask"
    )

    command = [
        sys.executable,
        "-m",
        eval_module,
        "--data_root",
        str(data_root),
        "--split_json",
        str(split_json),
        "--stats_json",
        str(stats_json),
        "--ckpt",
        str(checkpoint_path),
        "--patch",
        str(args.patch_size),
        "--stride",
        str(args.patch_size // 2),
        "--use_logK",
        "--out_json",
        str(run_seed_dir / "test_report_combined.json"),
        "--out_global_json",
        str(run_seed_dir / "test_report_global.json"),
        "--out_plume_json",
        str(run_seed_dir / "test_report_plumemask.json"),
    ]
    return {"cwd": RELEASE_ROOT, "command": command}


def write_manifest(args: argparse.Namespace, run_seed_dir: Path, run_family_dir: Path, run_name: str) -> None:
    spec = get_experiment(args.experiment_key)
    ensure_dir(run_seed_dir)
    data_root = resolve_path(args.data_root, DATA_ROOT)
    split_json = resolve_path(args.split_json, SPLIT_JSON)
    stats_json = resolve_path(args.stats_json, STATS_JSON)
    manifest = {
        "experiment_key": spec.key,
        "label": spec.label,
        "claim_scope": spec.claim_scope,
        "priority": spec.priority,
        "status": spec.status,
        "why": spec.why,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "learning_rate": args.learning_rate,
        "data_root": str(data_root),
        "split_json": str(split_json),
        "stats_json": str(stats_json),
        "run_family_dir": str(run_family_dir),
        "run_name": run_name,
    }
    with (run_seed_dir / "manuscript_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> None:
    args = parse_args()
    if not args.train and not args.eval:
        raise SystemExit("Specify at least one of --train or --eval.")

    spec = get_experiment(args.experiment_key)
    run_family_dir = Path(args.out_root) if args.out_root else (EXPERIMENT_RUNS_ROOT / spec.key)
    if not run_family_dir.is_absolute():
        run_family_dir = RELEASE_ROOT / run_family_dir
    run_name = args.run_name or f"seed{args.seed}"
    run_seed_dir = run_family_dir / run_name

    write_manifest(args, run_seed_dir, run_family_dir, run_name)

    if args.train:
        train_job = build_train_command(args, run_family_dir, run_name)
        run_command(train_job["command"], train_job["cwd"], args.dry_run)

    if args.eval:
        eval_job = build_eval_command(args, run_seed_dir)
        run_command(eval_job["command"], eval_job["cwd"], args.dry_run)


if __name__ == "__main__":
    main()
