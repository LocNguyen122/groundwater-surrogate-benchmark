"""Train the focal v5 models under the grouped-K confirmatory protocol."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.shared.data.concentration import LOG10_MAX, LOG10_MIN, bounded_log10_tensor
from src.shared.data.dataset_patch_multiout import GroundwaterPatchDatasetMultiOut
from src.shared.data.io import (
    filter_excluded_files,
    flatten_param_folders,
    load_param_split,
    load_train_stats,
)
from src.shared.data.transport_conditioning import (
    GroundwaterPatchDatasetMultiOutTransportConditioned,
    compute_conditioning_stats,
    load_transport_metadata,
    param_folder_from_path,
    parse_condition_params,
    transform_scalar,
)
from src.shared.utils.seed import make_generator, seed_worker, set_seed
from src.transport_surrogates.baselines.fno2d import FNO2D
from src.transport_surrogates.baselines.focal_frequency_loss import FocalFrequencyLoss
from src.transport_surrogates.baselines.temporal_attn_unet import CTAUNet
from src.transport_surrogates.baselines.train_unet_multiout_logc_B_aspp_attn import (
    UNet_ASPP_Attn,
)
from src.transport_surrogates.confirmatory_v5.conditioning import (
    CONDITIONING_VARIANTS,
    RegimeConditionedCTAUNet,
)
from src.transport_surrogates.confirmatory_v5.conditioning_mstmo import (
    RegimeConditionedMSTMO,
)


@dataclass
class Config:
    data_root: str
    split_json: str = "splits/param_split_grouped_k_v5.json"
    stats_json: str = "configs/train_stats_grouped_k_v5.json"
    exclusion_csv: str = "metadata/k_integrity_exclusions_v5.csv"
    out_root: str = "runs_v5_confirmatory"
    run_name: str = ""
    seed: int = 0
    model: str = "cta_ffl"
    transport_metadata_csv: str = ""
    condition_params: str = ""
    conditioning_variant: str = "v0_raw_channels"
    film_embed_dim: int = 8
    film_hidden_dim: int = 32
    patch_size: int = 320
    batch_size: int = 16
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    amp: bool = True
    base_channels: int = 64
    t_embed_dim: int = 25
    t_heads: int = 5
    pool_stride: int = 4
    attn_heads: int = 4
    alpha: float = 3.0
    y_bg: float = -12.0
    y_cap: float = -2.0
    huber_delta: float = 1.0
    t_w_min: float = 0.5
    t_w_max: float = 1.5
    lambda_temp: float = 0.05
    lambda_ffl: float = 0.01
    ffl_alpha: float = 1.0
    fixed_val_crop_seed: int = 20260715
    source_row: int = 399
    source_col: int = 199
    limit_train_files: int = 0
    limit_val_files: int = 0
    resume: bool = False
    val_crop_key: str = "relative"
    fno_width: int = 64
    fno_modes1: int = 20
    fno_modes2: int = 20
    fno_depth: int = 4
    fno_pad_ratio: float = 0.125
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def validation_anchor_row(self) -> int:
        """Legacy source_row is a crop anchor, not the injection-well coordinate."""
        return self.source_row

    @property
    def validation_anchor_col(self) -> int:
        return self.source_col


def warn_legacy_anchor_flags(argv: list[str]) -> None:
    """Preserve old CLI behavior while making its anchor semantics explicit."""
    if any(a.split("=", 1)[0] in {"--source_row", "--source_col"} for a in argv):
        warnings.warn("--source_row/--source_col are deprecated aliases for the validation crop anchor, "
                      "not the injection well. Use --validation_anchor_row/--validation_anchor_col.",
                      FutureWarning, stacklevel=2)


def parse_args() -> Config:
    warn_legacy_anchor_flags(sys.argv[1:])
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_json", default=Config.split_json)
    parser.add_argument("--stats_json", default=Config.stats_json)
    parser.add_argument("--exclusion_csv", default=Config.exclusion_csv)
    parser.add_argument("--out_root", default=Config.out_root)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", choices=["cta_ffl", "ms_tmo", "fno"], default="cta_ffl")
    parser.add_argument("--transport_metadata_csv", default="")
    parser.add_argument("--condition_params", default="")
    parser.add_argument(
        "--conditioning_variant",
        choices=CONDITIONING_VARIANTS,
        default="v0_raw_channels",
    )
    parser.add_argument("--film_embed_dim", type=int, default=8)
    parser.add_argument("--film_hidden_dim", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--t_embed_dim", type=int, default=25)
    parser.add_argument("--t_heads", type=int, default=5)
    parser.add_argument("--pool_stride", type=int, default=4)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--y_bg", type=float, default=-12.0)
    parser.add_argument("--y_cap", type=float, default=-2.0)
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--t_w_min", type=float, default=0.5)
    parser.add_argument("--t_w_max", type=float, default=1.5)
    parser.add_argument("--lambda_temp", type=float, default=0.05)
    parser.add_argument("--lambda_ffl", type=float, default=0.01)
    parser.add_argument("--ffl_alpha", type=float, default=1.0)
    parser.add_argument("--fixed_val_crop_seed", type=int, default=20260715)
    parser.add_argument("--validation_anchor_row", "--source_row", dest="source_row", type=int, default=399,
                        help="Validation crop anchor row; legacy --source_row is retained. Not the injection well.")
    parser.add_argument("--validation_anchor_col", "--source_col", dest="source_col", type=int, default=199,
                        help="Validation crop anchor column; legacy --source_col is retained.")
    parser.add_argument("--limit_train_files", type=int, default=0)
    parser.add_argument("--limit_val_files", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--val_crop_key", choices=["relative", "absolute_legacy"], default="relative",
                        help="Fixed validation-crop hash key. absolute_legacy reproduces the v5/v6 runs.")
    parser.add_argument("--fno_width", type=int, default=64)
    parser.add_argument("--fno_modes1", type=int, default=20)
    parser.add_argument("--fno_modes2", type=int, default=20)
    parser.add_argument("--fno_depth", type=int, default=4)
    parser.add_argument("--fno_pad_ratio", type=float, default=0.125)
    parser.add_argument("--device", default=Config.device)
    args = parser.parse_args()
    values = vars(args)
    values["amp"] = not values.pop("no_amp")
    if values["model"] == "fno":
        values["amp"] = False  # fairness rule: FNO trains in fp32 (complex weights vs GradScaler)
    return Config(**values)


def timestep_weights(count: int, cfg: Config, device: torch.device) -> torch.Tensor:
    positions = torch.linspace(0.0, 1.0, steps=count, device=device)
    return (cfg.t_w_min + (cfg.t_w_max - cfg.t_w_min) * positions.square()).view(
        1, count, 1, 1
    )


def normalized_level_centers(
    train_files,
    metadata,
    conditioning_stats,
) -> dict[str, list[float]]:
    """Return sorted normalized marginal levels using training-support metadata only."""
    train_folders = sorted({param_folder_from_path(path) for path in train_files})
    centers: dict[str, list[float]] = {}
    for name in ("alpha_L", "alpha_T_ratio"):
        stat = conditioning_stats[name]
        transformed = {
            transform_scalar(float(metadata[folder][name]), str(stat["transform"]))
            for folder in train_folders
        }
        centers[name] = sorted(
            (value - float(stat["mean"])) / (float(stat["std"]) + 1e-8)
            for value in transformed
        )
    if len(centers["alpha_L"]) != 3 or len(centers["alpha_T_ratio"]) != 2:
        raise ValueError(f"Training split lacks required marginal coverage: {centers}")
    return centers


def compute_loss(raw: torch.Tensor, target: torch.Tensor, cfg: Config, ffl) -> dict[str, torch.Tensor]:
    raw_nonfinite = (~torch.isfinite(raw)).float().mean().detach()
    raw = torch.nan_to_num(raw.float(), nan=0.0, posinf=20.0, neginf=-20.0)
    prediction = bounded_log10_tensor(raw, (LOG10_MIN, LOG10_MAX))
    target = target.float().clamp(LOG10_MIN, LOG10_MAX)
    concentration_weight = 1.0 + cfg.alpha * torch.clamp(
        (target - cfg.y_bg) / max(cfg.y_cap - cfg.y_bg, 1e-12), 0.0, 1.0
    )
    time_weight = timestep_weights(target.shape[1], cfg, target.device)
    huber = F.smooth_l1_loss(
        prediction, target, beta=cfg.huber_delta, reduction="none"
    )
    main = (concentration_weight * time_weight * huber).mean()
    first_difference = prediction[:, 1:] - prediction[:, :-1]
    temporal = (first_difference[:, 1:] - first_difference[:, :-1]).abs().mean()
    frequency = ffl(prediction, target) if ffl is not None else prediction.new_zeros(())
    total = main + cfg.lambda_temp * temporal
    if ffl is not None:
        total = total + cfg.lambda_ffl * frequency
    near_lower = (prediction <= LOG10_MIN + 1e-2).float().mean()
    near_upper = (prediction >= LOG10_MAX - 1e-2).float().mean()
    return {
        "total": total,
        "main": main,
        "temporal": temporal,
        "frequency": frequency,
        "near_lower": near_lower,
        "near_upper": near_upper,
        "raw_nonfinite": raw_nonfinite,
    }


def run_epoch(model, loader, cfg: Config, ffl, optimizer=None, scaler=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ["total", "main", "temporal", "frequency", "near_lower", "near_upper",
                                   "raw_nonfinite"]}
    used = skipped = 0
    skipped_nonfinite_loss = skipped_nonfinite_grad = 0
    amp_scale_start = (
        float(scaler.get_scale())
        if training and scaler is not None and scaler.is_enabled()
        else 0.0
    )
    for batch in loader:
        inputs = batch["K"].to(cfg.device, non_blocking=True)
        targets = batch["C_log"].to(cfg.device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=cfg.amp and cfg.device.startswith("cuda")):
                raw = model(inputs)
            losses = compute_loss(raw, targets, cfg, ffl)
            if not torch.isfinite(losses["total"]):
                skipped += 1
                skipped_nonfinite_loss += 1
                continue
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(losses["total"]).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False
                    )
                    if not torch.isfinite(grad_norm):
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        skipped += 1
                        skipped_nonfinite_grad += 1
                        continue
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["total"].backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False
                    )
                    if not torch.isfinite(grad_norm):
                        optimizer.zero_grad(set_to_none=True)
                        skipped += 1
                        skipped_nonfinite_grad += 1
                        continue
                    optimizer.step()
        for key in totals:
            totals[key] += float(losses[key].detach().item())
        used += 1
    if used == 0:
        raise RuntimeError("No finite batches were processed.")
    result = {key: value / used for key, value in totals.items()}
    amp_scale_end = (
        float(scaler.get_scale())
        if training and scaler is not None and scaler.is_enabled()
        else 0.0
    )
    result.update(
        {
            "batches": used,
            "skipped": skipped,
            "skipped_nonfinite_loss": skipped_nonfinite_loss,
            "skipped_nonfinite_grad": skipped_nonfinite_grad,
            "amp_scale_start": amp_scale_start,
            "amp_scale_end": amp_scale_end,
        }
    )
    return result


def write_validation_manifest(dataset, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_key", "crop_y0", "crop_x0", "validation_anchor_row", "validation_anchor_col"])
        for path in dataset.files:
            y0, x0 = dataset._fixed_crop_origin(path, 600, 400)
            writer.writerow([f"{Path(path).parent.name}/{Path(path).name}", y0, x0, dataset.source_row, dataset.source_col])


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    condition_params = parse_condition_params(cfg.condition_params)
    if bool(condition_params) != bool(cfg.transport_metadata_csv):
        raise ValueError("Provide both condition_params and transport_metadata_csv, or neither.")
    if cfg.conditioning_variant != "v0_raw_channels":
        if cfg.model not in {"cta_ffl", "ms_tmo"}:  # fno is rejected above
            raise ValueError(
                "Learned conditioning variants require a supported confirmatory backbone."
            )
        if condition_params != ["alpha_L", "alpha_T_ratio"]:
            raise ValueError(
                "Learned conditioning variants require condition_params=alpha_L,alpha_T_ratio "
                "in that order."
            )
    if cfg.model == "ms_tmo" and cfg.conditioning_variant == "v4_joint_continuous":
        raise ValueError("v4_joint_continuous is implemented for cta_ffl only.")
    if cfg.model == "fno" and cfg.conditioning_variant != "v0_raw_channels":
        raise ValueError("The FNO baseline uses raw constant input channels (v0_raw_channels) only.")
    if cfg.model == "ms_tmo" and not condition_params:
        raise ValueError("The confirmatory MS-TMO focal run is transport-conditioned.")
    if cfg.epochs != 200 and cfg.limit_train_files == 0:
        raise ValueError("Full confirmatory training must use 200 epochs.")
    if cfg.batch_size != 16 and cfg.limit_train_files == 0:
        raise ValueError("Full confirmatory training must use batch size 16.")

    if not cfg.run_name:
        regime = "tc" if condition_params else "k_only"
        cfg.run_name = f"{cfg.model}_{regime}_grouped_k_seed{cfg.seed}"
    out_dir = Path(cfg.out_root) / cfg.run_name
    if out_dir.exists() and any(out_dir.iterdir()) and not cfg.resume:
        raise FileExistsError(f"Refusing to overwrite nonempty run directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_params, val_params, _ = load_param_split(cfg.split_json)
    train_files_all = filter_excluded_files(
        flatten_param_folders(cfg.data_root, train_params), cfg.exclusion_csv
    )
    val_files_all = filter_excluded_files(
        flatten_param_folders(cfg.data_root, val_params), cfg.exclusion_csv
    )
    train_files = list(train_files_all)
    val_files = list(val_files_all)
    if cfg.limit_train_files > 0:
        train_files = train_files[: cfg.limit_train_files]
    if cfg.limit_val_files > 0:
        val_files = val_files[: cfg.limit_val_files]
    stats = load_train_stats(cfg.stats_json)

    dataset_class = GroundwaterPatchDatasetMultiOut
    dataset_kwargs: dict[str, object] = {}
    conditioning_stats = {}
    level_centers: dict[str, list[float]] = {}
    if condition_params:
        metadata = load_transport_metadata(cfg.transport_metadata_csv)
        transforms = {name: "log10" for name in condition_params}
        conditioning_stats = compute_conditioning_stats(
            train_files_all, metadata, condition_params, transforms
        )
        if cfg.conditioning_variant != "v0_raw_channels":
            level_centers = normalized_level_centers(
                train_files_all, metadata, conditioning_stats
            )
        dataset_class = GroundwaterPatchDatasetMultiOutTransportConditioned
        dataset_kwargs = {
            "metadata": metadata,
            "condition_params": condition_params,
            "conditioning_stats": conditioning_stats,
        }

    common = {
        "stats": stats,
        "patch_size": cfg.patch_size,
        "use_logK": True,
        "eps_k": 1e-6,
        "eps_c": 1e-12,
        **dataset_kwargs,
    }
    train_dataset = dataset_class(train_files, crop_mode="random", **common)
    val_dataset = dataset_class(
        val_files,
        crop_mode="fixed",
        fixed_crop_seed=cfg.fixed_val_crop_seed,
        require_source_in_crop=True,
        crop_key=cfg.val_crop_key,
        source_row=cfg.validation_anchor_row,
        source_col=cfg.validation_anchor_col,
        **common,
    )
    write_validation_manifest(val_dataset, out_dir / "validation_crop_manifest.csv")

    generator = make_generator(cfg.seed)
    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.device.startswith("cuda"),
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    input_channels = 1 + len(condition_params)
    if cfg.model == "cta_ffl":
        if cfg.conditioning_variant == "v0_raw_channels":
            model = CTAUNet(
                in_ch=input_channels,
                out_ch=25,
                base=cfg.base_channels,
                t_embed_dim=cfg.t_embed_dim,
                t_heads=cfg.t_heads,
                pool_stride=cfg.pool_stride,
            ).to(cfg.device)
            model_family = "cta_unet_ffl_transport_conditioned" if condition_params else "cta_unet"
        else:
            model = RegimeConditionedCTAUNet(
                conditioning_variant=cfg.conditioning_variant,
                normalized_level_centers=level_centers,
                out_ch=25,
                base=cfg.base_channels,
                t_embed_dim=cfg.t_embed_dim,
                t_heads=cfg.t_heads,
                pool_stride=cfg.pool_stride,
                film_embed_dim=cfg.film_embed_dim,
                film_hidden_dim=cfg.film_hidden_dim,
            ).to(cfg.device)
            model_family = "cta_unet_ffl_regime_conditioned"
        ffl = FocalFrequencyLoss(alpha=cfg.ffl_alpha).to(cfg.device)
    elif cfg.model == "fno":
        model = FNO2D(
            in_ch=input_channels, out_ch=25, width=cfg.fno_width, modes1=cfg.fno_modes1,
            modes2=cfg.fno_modes2, depth=cfg.fno_depth, use_coords=True, pad_ratio=cfg.fno_pad_ratio,
        ).to(cfg.device)
        model_family = "fno_transport_conditioned" if condition_params else "fno2d"
        ffl = None
    else:
        if cfg.conditioning_variant == "v0_raw_channels":
            model = UNet_ASPP_Attn(
                in_ch=input_channels,
                out_ch=25,
                base=cfg.base_channels,
                attn_heads=cfg.attn_heads,
            ).to(cfg.device)
            model_family = "ms_tmo_transport_conditioned"
        else:
            model = RegimeConditionedMSTMO(
                conditioning_variant=cfg.conditioning_variant,
                normalized_level_centers=level_centers,
                out_ch=25,
                base=cfg.base_channels,
                attn_heads=cfg.attn_heads,
                film_embed_dim=cfg.film_embed_dim,
                film_hidden_dim=cfg.film_hidden_dim,
            ).to(cfg.device)
            model_family = "ms_tmo_regime_conditioned"
        ffl = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and cfg.device.startswith("cuda"))

    config_payload = asdict(cfg)
    config_payload.update(
        {
            "model_family": model_family,
            "input_channels": input_channels,
            "use_logK": True,
            "eps_k": 1e-6,
            "eps_c": 1e-12,
            "use_transport_conditioning": bool(condition_params),
            "condition_params": condition_params,
            "transport_conditioning_stats": conditioning_stats,
            "conditioning_level_centers": level_centers,
            "transport_conditioning_transforms": {name: "log10" for name in condition_params},
            "backbone_input_channels": 1 if cfg.conditioning_variant != "v0_raw_channels" else input_channels,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "output_parameterization": "bounded_sigmoid_log10",
            "output_bounds": [LOG10_MIN, LOG10_MAX],
            "validation_crop_policy": "fixed_anchor_containing",
            "validation_anchor_row": cfg.validation_anchor_row,
            "validation_anchor_col": cfg.validation_anchor_col,
            "legacy_coordinate_names": "source_row/source_col refer to the validation anchor, not the injection well",
            "validation_crop_key": cfg.val_crop_key,
            "width": cfg.fno_width, "modes1": cfg.fno_modes1, "modes2": cfg.fno_modes2,
            "depth": cfg.fno_depth, "use_coords": True, "pad_ratio": cfg.fno_pad_ratio,
            "n_train_files": len(train_files),
            "n_val_files": len(val_files),
            "n_train_files_for_conditioning_stats": len(train_files_all),
            "n_val_files_before_smoke_limit": len(val_files_all),
        }
    )
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    best_val = float("inf")
    start_epoch = 1
    last_path = out_dir / "last.pt"
    best_path = out_dir / "best.pt"
    if cfg.resume:
        checkpoint = torch.load(last_path, map_location=cfg.device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler.is_enabled() and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint["best_val"])
        rng = checkpoint.get("rng")
        if rng is not None:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            generator.set_state(rng["loader"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        else:
            print("WARNING: checkpoint has no RNG state (pre-v7); resumed run is not bit-identical")

    log_path = out_dir / "train_log.csv"
    if not log_path.exists():
        with log_path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    "epoch", "train_loss", "val_loss", "train_main", "val_main",
                    "train_temporal", "val_temporal", "train_frequency", "val_frequency",
                    "val_near_lower", "val_near_upper", "lr", "skipped",
                    "skipped_nonfinite_loss", "skipped_nonfinite_grad",
                    "amp_scale_start", "amp_scale_end", "train_raw_nonfinite", "val_raw_nonfinite",
                ]
            )

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_metrics = run_epoch(model, train_loader, cfg, ffl, optimizer, scaler)
        val_metrics = run_epoch(model, val_loader, cfg, ffl)
        scheduler.step()
        improved = val_metrics["total"] < best_val
        if improved:
            best_val = val_metrics["total"]
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "cfg": config_payload,
            "stats": stats,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "best_val": best_val,
            "rng": {
                "python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "loader": generator.get_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        torch.save(checkpoint, last_path)
        if improved:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "cfg": config_payload, "stats": stats, "best_val": best_val},
                best_path,
            )
        with log_path.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    epoch, train_metrics["total"], val_metrics["total"],
                    train_metrics["main"], val_metrics["main"],
                    train_metrics["temporal"], val_metrics["temporal"],
                    train_metrics["frequency"], val_metrics["frequency"],
                    val_metrics["near_lower"], val_metrics["near_upper"],
                    scheduler.get_last_lr()[0], train_metrics["skipped"],
                    train_metrics["skipped_nonfinite_loss"],
                    train_metrics["skipped_nonfinite_grad"],
                    train_metrics["amp_scale_start"], train_metrics["amp_scale_end"],
                    train_metrics["raw_nonfinite"], val_metrics["raw_nonfinite"],
                ]
            )
        print(
            f"epoch={epoch:03d} train={train_metrics['total']:.6f} "
            f"val={val_metrics['total']:.6f} best={best_val:.6f} "
            f"lower={val_metrics['near_lower']:.4f} upper={val_metrics['near_upper']:.4f} "
            f"skipped={train_metrics['skipped']} "
            f"loss_nonfinite={train_metrics['skipped_nonfinite_loss']} "
            f"grad_nonfinite={train_metrics['skipped_nonfinite_grad']} "
            f"amp_scale={train_metrics['amp_scale_start']:.0f}->"
            f"{train_metrics['amp_scale_end']:.0f}"
        )


if __name__ == "__main__":
    main()
