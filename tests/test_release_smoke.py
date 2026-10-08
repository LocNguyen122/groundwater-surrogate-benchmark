"""CPU-only release smoke tests."""

from __future__ import annotations

import importlib
import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ReleaseSmokeTest(unittest.TestCase):
    def test_core_imports(self) -> None:
        modules = (
            "src.shared.data.dataset_patch_multiout",
            "src.shared.data.dataset_timecond_patch",
            "src.shared.data.transport_conditioning",
            "src.shared.eval.centroid_error",
            "src.shared.eval.plume_ssim",
            "src.transport_surrogates.baselines.temporal_attn_unet",
            "src.transport_surrogates.baselines.fno2d",
            "src.transport_surrogates.baselines.deeponet2d",
        )
        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_cta_forward_shape(self) -> None:
        from src.transport_surrogates.baselines.temporal_attn_unet import CTAUNet

        torch.set_num_threads(1)
        model = CTAUNet(in_ch=3, out_ch=25, base=8).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 25, 64, 64))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_fixed_split_counts(self) -> None:
        split = json.loads((ROOT / "splits" / "param_split_fixed.json").read_text(encoding="utf-8"))
        counts = {key: len(split["splits"][key]["param_ids"]) for key in ("train", "val", "test")}
        self.assertEqual(counts, {"train": 112, "val": 16, "test": 34})


if __name__ == "__main__":
    unittest.main()
