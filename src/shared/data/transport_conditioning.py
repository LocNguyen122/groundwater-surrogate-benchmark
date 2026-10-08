from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

from src.shared.data.dataset_patch_multiout import GroundwaterPatchDatasetMultiOut


RELEASE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METADATA_CSV = RELEASE_ROOT / "metadata" / "transport_parameters_162.csv"
SUPPORTED_CONDITION_PARAMS = ("alpha_L", "alpha_T_ratio")


@dataclass(frozen=True)
class ScalarNorm:
    transform: str
    mean: float
    std: float


def param_folder_from_path(path: str | Path) -> str:
    for part in Path(path).parts:
        if part.startswith("param_"):
            return part
    raise ValueError(f"Could not find a param_XXX folder in path: {path}")


def param_id_from_folder(param_folder: str) -> int:
    if not param_folder.startswith("param_"):
        raise ValueError(f"Expected param_XXX folder, got {param_folder}")
    return int(param_folder.split("_", 1)[1])


def load_transport_metadata(metadata_csv: str = "") -> Dict[str, Dict[str, float]]:
    """Load the authoritative per-parameter transport metadata table.

    The bundled 162-row CSV is the default. It was exported from the simulation
    parameter registry so parameter IDs are never reconstructed from an assumed
    Cartesian-loop order. An alternate CSV may be supplied explicitly and must
    contain ``param_folder`` or ``param_id``, ``alpha_L``, and
    ``alpha_T_ratio``.
    """
    path = Path(metadata_csv) if metadata_csv else DEFAULT_METADATA_CSV
    if not path.exists():
        raise FileNotFoundError(f"Missing transport metadata CSV: {path}")

    rows: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Transport metadata CSV has no header: {path}")
        fields = set(reader.fieldnames)
        if "param_folder" not in fields and "param_id" not in fields:
            raise ValueError("Transport metadata CSV needs param_folder or param_id.")
        missing = [name for name in SUPPORTED_CONDITION_PARAMS if name not in fields]
        if missing:
            raise ValueError(f"Transport metadata CSV missing columns: {missing}")

        for row in reader:
            if row.get("param_folder"):
                folder = row["param_folder"]
            else:
                folder = f"param_{int(row['param_id']):03d}"
            parsed: Dict[str, float] = {}
            for key, value in row.items():
                if key in (None, "param_folder") or value in ("", None):
                    continue
                parsed[key] = float(value)
            rows[folder] = parsed
    return rows


def parse_condition_params(value: str | Sequence[str]) -> List[str]:
    if isinstance(value, str):
        params = [item.strip() for item in value.split(",") if item.strip()]
    else:
        params = [str(item).strip() for item in value if str(item).strip()]
    if not params:
        return []
    unsupported = [param for param in params if param not in SUPPORTED_CONDITION_PARAMS]
    if unsupported:
        raise ValueError(f"Unsupported condition params {unsupported}; supported={SUPPORTED_CONDITION_PARAMS}")
    return params


def transform_scalar(value: float, transform: str) -> float:
    if transform == "identity":
        return float(value)
    if transform == "log10":
        if value <= 0.0:
            raise ValueError(f"log10 transform needs positive value, got {value}")
        return float(math.log10(value))
    raise ValueError(f"Unknown scalar transform: {transform}")


def compute_conditioning_stats(
    train_files: Iterable[str | Path],
    metadata: Mapping[str, Mapping[str, float]],
    condition_params: Sequence[str],
    transforms: Mapping[str, str],
) -> Dict[str, Dict[str, float | str]]:
    train_folders = sorted({param_folder_from_path(path) for path in train_files})
    stats: Dict[str, Dict[str, float | str]] = {}
    for param in condition_params:
        values = []
        transform = transforms[param]
        for folder in train_folders:
            if folder not in metadata:
                raise KeyError(f"Missing transport metadata for {folder}")
            values.append(transform_scalar(float(metadata[folder][param]), transform))
        arr = np.asarray(values, dtype=np.float64)
        std = float(arr.std())
        stats[param] = {
            "transform": transform,
            "mean": float(arr.mean()),
            "std": std if std > 0.0 else 1.0,
        }
    return stats


def condition_values_for_folder(
    param_folder: str,
    metadata: Mapping[str, Mapping[str, float]],
    condition_params: Sequence[str],
    norm_stats: Mapping[str, Mapping[str, float | str]],
) -> np.ndarray:
    if param_folder not in metadata:
        raise KeyError(f"Missing transport metadata for {param_folder}")
    values = []
    for param in condition_params:
        stat = norm_stats[param]
        raw = float(metadata[param_folder][param])
        transformed = transform_scalar(raw, str(stat["transform"]))
        normalized = (transformed - float(stat["mean"])) / (float(stat["std"]) + 1e-8)
        values.append(normalized)
    return np.asarray(values, dtype=np.float32)


def make_condition_maps(
    param_folder: str,
    height: int,
    width: int,
    metadata: Mapping[str, Mapping[str, float]],
    condition_params: Sequence[str],
    norm_stats: Mapping[str, Mapping[str, float | str]],
) -> np.ndarray:
    values = condition_values_for_folder(param_folder, metadata, condition_params, norm_stats)
    return np.broadcast_to(values[:, None, None], (len(values), height, width)).astype(np.float32).copy()


class GroundwaterPatchDatasetMultiOutTransportConditioned(GroundwaterPatchDatasetMultiOut):
    """transport surrogate benchmark patch dataset with optional constant transport-parameter channels."""

    def __init__(
        self,
        file_list,
        stats,
        patch_size=320,
        use_logK=True,
        eps_k=1e-6,
        eps_c=1e-12,
        timesteps=None,
        crop_mode="random",
        fixed_crop_seed=0,
        require_source_in_crop=False,
        source_row=399,
        source_col=199,
        crop_key="absolute_legacy",
        *,
        metadata: Mapping[str, Mapping[str, float]],
        condition_params: Sequence[str],
        conditioning_stats: Mapping[str, Mapping[str, float | str]],
    ):
        super().__init__(
            file_list=file_list,
            stats=stats,
            patch_size=patch_size,
            use_logK=use_logK,
            eps_k=eps_k,
            eps_c=eps_c,
            timesteps=timesteps,
            crop_mode=crop_mode,
            fixed_crop_seed=fixed_crop_seed,
            require_source_in_crop=require_source_in_crop,
            source_row=source_row,
            source_col=source_col,
            crop_key=crop_key,
        )
        self.metadata = metadata
        self.condition_params = list(condition_params)
        self.conditioning_stats = conditioning_stats

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        k_tensor = item["K"]
        _, height, width = k_tensor.shape
        folder = param_folder_from_path(self.files[idx])
        cond = make_condition_maps(
            folder,
            height,
            width,
            self.metadata,
            self.condition_params,
            self.conditioning_stats,
        )
        item["K"] = torch.cat([k_tensor, torch.from_numpy(cond)], dim=0)
        item["transport_conditioning"] = torch.from_numpy(cond)
        return item
