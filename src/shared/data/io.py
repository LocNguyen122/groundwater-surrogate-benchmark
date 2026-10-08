from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


def load_param_split(split_json_path: str) -> Tuple[List[str], List[str], List[str]]:
    with open(split_json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    splits = payload["splits"]
    return (
        splits["train"]["param_folders"],
        splits["val"]["param_folders"],
        splits["test"]["param_folders"],
    )


def flatten_param_folders(
    data_root: str,
    param_folders: List[str],
    file_glob: str = "*.npz",
) -> List[str]:
    files: List[str] = []
    for folder in param_folders:
        files.extend(sorted(glob.glob(os.path.join(data_root, folder, file_glob))))
    if not files:
        raise FileNotFoundError(
            f"No files found under data_root='{data_root}' for glob='{file_glob}'."
        )
    return files


def load_exclusion_keys(exclusion_csv: str = "") -> set[str]:
    """Load portable ``param_folder/real_XXX.npz`` keys from a CSV manifest."""
    if not exclusion_csv:
        return set()
    path = Path(exclusion_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Missing exclusion manifest: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Exclusion manifest has no header: {path}")
        if "sample_key" in reader.fieldnames:
            keys = {str(row["sample_key"]).replace("\\", "/") for row in reader}
        elif {"param_folder", "realization"}.issubset(reader.fieldnames):
            keys = {
                f"{row['param_folder']}/{row['realization']}.npz".replace(".npz.npz", ".npz")
                for row in reader
            }
        else:
            raise ValueError(
                "Exclusion manifest needs sample_key or param_folder and realization columns."
            )
    return {key for key in keys if key}


def sample_key(path: str | Path) -> str:
    value = Path(path)
    parent = next((part for part in reversed(value.parts[:-1]) if part.startswith("param_")), None)
    if parent is None:
        raise ValueError(f"No param_XXX folder in path: {path}")
    return f"{parent}/{value.name}"


def filter_excluded_files(files: List[str], exclusion_csv: str = "") -> List[str]:
    excluded = load_exclusion_keys(exclusion_csv)
    if not excluded:
        return list(files)
    kept = [path for path in files if sample_key(path) not in excluded]
    if not kept:
        raise ValueError("Exclusion manifest removed every file.")
    return kept


def load_train_stats(stats_json_path: str) -> Dict[str, float]:
    with open(stats_json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)
