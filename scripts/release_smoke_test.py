"""CPU-only smoke checks for the public transport-conditioning release."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_TABLES = {
    "scalar_benchmark_summary.csv",
    "scalar_benchmark_per_seed.csv",
    "shape_metrics_summary.csv",
    "shape_metrics_per_seed.csv",
    "shape_metrics_six_family_summary.csv",
    "shape_metrics_six_family_per_seed.csv",
    "shape_metrics_by_timestep.csv",
    "roi_ssim_gt_pred_union_summary.csv",
    "roi_ssim_gt_pred_union_per_seed.csv",
    "threshold_sensitivity_summary.csv",
    "threshold_sensitivity_per_seed.csv",
    "inference_latency_summary.csv",
    "parameter_counts.csv",
    "paired_transport_conditioning_statistics.csv",
    "alpha_channel_ablation_summary.csv",
    "alpha_channel_ablation_per_seed.csv",
}

IMPORTS = (
    "src.shared.data.dataset_patch_multiout",
    "src.shared.data.dataset_timecond_patch",
    "src.shared.data.transport_conditioning",
    "src.shared.eval.centroid_error",
    "src.shared.eval.plume_ssim",
    "src.transport_surrogates.baselines.temporal_attn_unet",
    "src.transport_surrogates.baselines.pmt_unet_aspp_attn",
    "src.transport_surrogates.baselines.unet2d",
    "src.transport_surrogates.baselines.fno2d",
    "src.transport_surrogates.baselines.deeponet2d",
    "src.transport_surrogates.baselines.train.train_fno_multiout_logc_baseline",
    "src.transport_surrogates.baselines.train.train_deeponet_multiout_logc_baseline",
    "src.transport_surrogates.baselines.train.train_pix2pix_multioutput_patch_logc",
)


def main() -> None:
    torch.set_num_threads(1)
    for module in IMPORTS:
        importlib.import_module(module)

    from src.transport_surrogates.baselines.temporal_attn_unet import CTAUNet

    model = CTAUNet(in_ch=3, out_ch=25, base=8).eval()
    sample = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        prediction = model(sample)
    if tuple(prediction.shape) != (1, 25, 64, 64):
        raise RuntimeError(f"Unexpected CTA output shape: {tuple(prediction.shape)}")
    if not torch.isfinite(prediction).all():
        raise RuntimeError("CTA forward pass produced non-finite values")

    tables_dir = ROOT / "results" / "tables"
    missing = sorted(name for name in REQUIRED_TABLES if not (tables_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"Missing manuscript table artifacts: {missing}")

    split = json.loads((ROOT / "splits" / "param_split_fixed.json").read_text(encoding="utf-8"))
    counts = {key: len(split["splits"][key]["param_ids"]) for key in ("train", "val", "test")}
    if counts != {"train": 112, "val": 16, "test": 34}:
        raise RuntimeError(f"Unexpected split counts: {counts}")

    from src.shared.data.transport_conditioning import load_transport_metadata

    metadata = load_transport_metadata()
    if len(metadata) != 162:
        raise RuntimeError(f"Unexpected transport metadata row count: {len(metadata)}")

    dummy = np.ones((8, 8), dtype=np.float32)
    if not np.isfinite(dummy).all():
        raise RuntimeError("NumPy sanity check failed")

    print(json.dumps({"status": "ok", "split_counts": counts, "cta_output": list(prediction.shape)}, indent=2))


if __name__ == "__main__":
    main()
