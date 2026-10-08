"""Registry of the reported paper training entry points.

Only runnable modules included in this code release are registered here.
Experimental families not reported in the manuscript are intentionally omitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal

from src.transport_surrogates.paths import RELEASE_ROOT, RUNS_ROOT


DATA_ROOT = RELEASE_ROOT / "data"
SPLIT_JSON = RELEASE_ROOT / "splits" / "param_split_fixed.json"
STATS_JSON = RELEASE_ROOT / "configs" / "train_stats.json"
EXPERIMENT_RUNS_ROOT = RUNS_ROOT

LaunchType = Literal["module", "script"]
ClaimScope = Literal["core", "contextual"]


@dataclass(frozen=True)
class LaunchSpec:
    launch_type: LaunchType
    target: str
    cwd: Path
    args: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    label: str
    claim_scope: ClaimScope
    priority: str
    status: str
    why: str
    launch: LaunchSpec


def _spec(
    key: str,
    label: str,
    target: str,
    args: List[str],
    *,
    contextual: bool = False,
    why: str = "Reported deterministic surrogate training entry point.",
) -> ExperimentSpec:
    return ExperimentSpec(
        key=key,
        label=label,
        claim_scope="contextual" if contextual else "core",
        priority="paper",
        status="contextual_protocol" if contextual else "ready",
        why=why,
        launch=LaunchSpec("module", target, RELEASE_ROOT, args),
    )


UNET_ARGS = [
    "--use_logK", "--amp", "--alpha", "3.0",
    "--t_weight_mode", "quad", "--t_w_min", "0.5", "--t_w_max", "1.5",
    "--lambda_temp", "0.05", "--lambda_mass", "0.0",
]
FNO_ARGS = [
    "--use_logK", "--width", "64", "--modes1", "20", "--modes2", "20",
    "--depth", "4", "--alpha", "3.0", "--t_weight_mode", "quad",
    "--t_w_min", "0.5", "--t_w_max", "1.5", "--lambda_temp", "0.05",
]
DEEPONET_ARGS = [
    "--use_logK", "--branch_width", "128", "--branch_latent", "384",
    "--trunk_width", "384", "--basis_rank", "96", "--alpha", "3.0",
    "--t_weight_mode", "quad", "--t_w_min", "0.5", "--t_w_max", "1.5",
    "--lambda_temp", "0.05",
]
TC_ARGS = [
    "--condition_params", "alpha_L,alpha_T_ratio",
    "--alpha_L_transform", "log10",
    "--alpha_T_ratio_transform", "log10",
]
CTA_FFL_ARGS = [
    "--use_logK", "--amp", "--t_embed_dim", "25", "--t_heads", "5",
    "--pool_stride", "4", "--alpha", "3.0", "--t_weight_mode", "quad",
    "--t_w_min", "0.5", "--t_w_max", "1.5", "--lambda_temp", "0.05",
    "--lambda_ffl", "0.01", "--ffl_alpha", "1.0", "--lambda_mass", "0.0",
]


EXPERIMENTS: Dict[str, ExperimentSpec] = {
    "cnn_baseline": _spec(
        "cnn_baseline", "CNN/U-Net K-only",
        "src.transport_surrogates.baselines.train.train_unet_multioutput_patch_logc",
        ["--use_logK", "--amp"],
    ),
    "tmo_unet_tl_baseline": _spec(
        "tmo_unet_tl_baseline", "TMO-UNet+TL K-only",
        "src.transport_surrogates.baselines.train_unet_multiout_logc_A_tweight_tsmooth",
        UNET_ARGS,
    ),
    "ms_tmo_baseline": _spec(
        "ms_tmo_baseline", "MS-TMO-UNet K-only",
        "src.transport_surrogates.baselines.train.train_unet_multiout_logc_B_aspp_attn",
        UNET_ARGS,
    ),
    "cta_unet_ffl_baseline": _spec(
        "cta_unet_ffl_baseline", "CTA-UNet+FFL K-only",
        "src.transport_surrogates.baselines.train.train_cta_unet_ffl_multiout_logc",
        CTA_FFL_ARGS,
    ),
    "fno_baseline": _spec(
        "fno_baseline", "FNO K-only",
        "src.transport_surrogates.baselines.train.train_fno_multiout_logc_baseline",
        FNO_ARGS,
    ),
    "deeponet_baseline": _spec(
        "deeponet_baseline", "DeepONet K-only",
        "src.transport_surrogates.baselines.train.train_deeponet_multiout_logc_baseline",
        DEEPONET_ARGS,
    ),
    "pix2pix_baseline": _spec(
        "pix2pix_baseline", "Pix2Pix K-only",
        "src.transport_surrogates.baselines.train.train_pix2pix_multioutput_patch_logc",
        ["--use_logK", "--amp", "--alpha", "20.0"],
        contextual=True,
        why="Historical comparator with constant learning rate and concentration weight 20.",
    ),
    "timecond_baseline": _spec(
        "timecond_baseline", "Time-conditioned U-Net K-only",
        "src.transport_surrogates.baselines.train_unet_timecond_multihead_logc",
        ["--use_logK", "--amp", "--alpha", "3.0", "--mid_bias_sigma", "0.18"],
        contextual=True,
        why="Historical per-timestep comparator; cross-regime training details are not matched.",
    ),
    "cnn_transport_conditioned": _spec(
        "cnn_transport_conditioned", "CNN/U-Net structured-channel input",
        "src.transport_surrogates.baselines.train.train_cnn_transport_conditioned",
        ["--use_logK", "--amp", *TC_ARGS],
    ),
    "tmo_unet_tl_transport_conditioned": _spec(
        "tmo_unet_tl_transport_conditioned", "TMO-UNet+TL structured-channel input",
        "src.transport_surrogates.baselines.train.train_unet_tl_transport_conditioned",
        [*UNET_ARGS, *TC_ARGS],
    ),
    "ms_tmo_transport_conditioned": _spec(
        "ms_tmo_transport_conditioned", "MS-TMO-UNet structured-channel input",
        "src.transport_surrogates.baselines.train.train_ms_tmo_transport_conditioned",
        [*UNET_ARGS, "--use_transport_conditioning", *TC_ARGS],
    ),
    "cta_unet_ffl_transport_conditioned": _spec(
        "cta_unet_ffl_transport_conditioned", "CTA-UNet+FFL structured-channel input",
        "src.transport_surrogates.baselines.train.train_cta_unet_ffl_transport_conditioned",
        [*CTA_FFL_ARGS, *TC_ARGS],
    ),
    "cta_unet_ffl_alphaL": _spec(
        "cta_unet_ffl_alphaL", "CTA-UNet+FFL alpha_L channel",
        "src.transport_surrogates.baselines.train.train_cta_unet_ffl_transport_conditioned",
        [*CTA_FFL_ARGS, "--condition_params", "alpha_L", "--alpha_L_transform", "log10"],
    ),
    "cta_unet_ffl_alphaT_ratio": _spec(
        "cta_unet_ffl_alphaT_ratio", "CTA-UNet+FFL alpha_T/alpha_L channel",
        "src.transport_surrogates.baselines.train.train_cta_unet_ffl_transport_conditioned",
        [*CTA_FFL_ARGS, "--condition_params", "alpha_T_ratio", "--alpha_T_ratio_transform", "log10"],
    ),
    "fno_transport_conditioned": _spec(
        "fno_transport_conditioned", "FNO structured-channel input",
        "src.transport_surrogates.baselines.train.train_fno_transport_conditioned",
        [*FNO_ARGS, *TC_ARGS],
    ),
    "deeponet_transport_conditioned": _spec(
        "deeponet_transport_conditioned", "DeepONet structured-channel input",
        "src.transport_surrogates.baselines.train.train_deeponet_transport_conditioned",
        ["--amp", *DEEPONET_ARGS, *TC_ARGS],
    ),
    "pix2pix_transport_conditioned": _spec(
        "pix2pix_transport_conditioned", "Pix2Pix structured-channel input",
        "src.transport_surrogates.baselines.train.train_pix2pix_transport_conditioned",
        ["--use_logK", "--amp", "--alpha", "3.0", *TC_ARGS],
        contextual=True,
        why="Cross-regime comparison is contextual because the K-only schedule and weight differ.",
    ),
    "timecond_transport_conditioned": _spec(
        "timecond_transport_conditioned", "Time-conditioned U-Net structured-channel input",
        "src.transport_surrogates.baselines.train.train_timecond_transport_conditioned",
        ["--use_logK", "--alpha", "3.0", "--mid_bias_sigma", "0.18", *TC_ARGS],
        contextual=True,
        why="Cross-regime comparison is contextual because sampling, cropping, validation, and scheduler details differ.",
    ),
}


def get_experiment(key: str) -> ExperimentSpec:
    try:
        return EXPERIMENTS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown experiment key '{key}'. Available: {', '.join(sorted(EXPERIMENTS))}") from exc
