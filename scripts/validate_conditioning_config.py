from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from src.transport_surrogates.baselines.temporal_attn_unet import CTAUNet

torch.set_num_threads(2)

CONFIGS = [
    "cta_unet_ffl_alphaL.yaml",
    "cta_unet_ffl_alphaT_ratio.yaml",
    "cta_unet_ffl_transport_conditioned.yaml",
]


def check_config(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_channels = int(cfg["input_channels"])
    output_channels = int(cfg.get("output_channels", 25))
    condition_params = list(cfg.get("condition_params", []))

    x = torch.randn(1, input_channels, 32, 32)
    y = torch.randn(1, output_channels, 32, 32)
    model = CTAUNet(in_ch=input_channels, out_ch=output_channels, base=8)
    model.eval()
    with torch.no_grad():
        pred = model(x)
        loss = F.mse_loss(pred, y)

    if pred.shape != y.shape:
        raise RuntimeError(f"Unexpected output shape for {config_path.name}: {tuple(pred.shape)}")
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss for {config_path.name}")

    return {
        "config": config_path.name,
        "input_channels": input_channels,
        "condition_params": condition_params,
        "input_shape": list(x.shape),
        "output_shape": list(pred.shape),
        "loss": float(loss.item()),
    }


def main() -> None:
    results = [check_config(RELEASE_ROOT / "configs" / name) for name in CONFIGS]
    print(json.dumps({"status": "ok", "results": results}, indent=2))


if __name__ == "__main__":
    main()
