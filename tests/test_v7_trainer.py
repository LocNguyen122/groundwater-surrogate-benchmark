"""Tests for the v7 trainer changes (pre-submission review v6, R1 and R2)."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.shared.data.dataset_patch_multiout import GroundwaterPatchDatasetMultiOut
from src.transport_surrogates.baselines.fno2d import FNO2D
from src.transport_surrogates.confirmatory_v5.conditioning import RegimeConditionedCTAUNet
from src.transport_surrogates.confirmatory_v5.conditioning_mstmo import RegimeConditionedMSTMO
from src.transport_surrogates.confirmatory_v5.train import compute_loss

CENTERS = {"alpha_L": [-1.118, 0.0, 1.118], "alpha_T_ratio": [-1.0, 1.0]}
STATS = {"k_mean": 0.0, "k_std": 1.0}


def loss_cfg():
    return SimpleNamespace(alpha=3.0, y_bg=-12.0, y_cap=-2.0, huber_delta=1.0, t_w_min=0.5, t_w_max=1.5,
                           lambda_temp=0.05, lambda_ffl=0.01)


class RawNonFiniteTests(unittest.TestCase):
    def test_raw_nonfinite_fraction_counted_before_sanitising(self):
        raw = torch.zeros(1, 25, 4, 4)
        raw[0, 0, 0, 0] = float("nan")
        raw[0, 1, 0, 0] = float("inf")
        losses = compute_loss(raw, torch.full_like(raw, -6.0), loss_cfg(), None)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertAlmostEqual(float(losses["raw_nonfinite"]), 2 / raw.numel(), places=7)

    def test_finite_raw_reports_zero(self):
        raw = torch.zeros(1, 25, 4, 4)
        self.assertEqual(float(compute_loss(raw, raw - 6.0, loss_cfg(), None)["raw_nonfinite"]), 0.0)


class CropKeyTests(unittest.TestCase):
    def make(self, root, key):
        path = f"{root}/param_021/real_003.npz"
        return GroundwaterPatchDatasetMultiOut([path], STATS, crop_mode="fixed", fixed_crop_seed=20260715,
                                               require_source_in_crop=True, crop_key=key), path

    def test_relative_key_is_independent_of_data_root(self):
        a, pa = self.make("/cluster/data", "relative")
        b, pb = self.make("D:/local/data", "relative")
        self.assertEqual(a._fixed_crop_origin(pa, 600, 400), b._fixed_crop_origin(pb, 600, 400))

    def test_legacy_key_reproduces_root_dependent_crops(self):
        a, pa = self.make("/cluster/data", "absolute_legacy")
        b, pb = self.make("D:/local/data", "absolute_legacy")
        self.assertNotEqual(a._fixed_crop_origin(pa, 600, 400), b._fixed_crop_origin(pb, 600, 400))

    def test_default_key_is_legacy(self):
        ds = GroundwaterPatchDatasetMultiOut(["x/param_000/real_001.npz"], STATS)
        self.assertEqual(ds.crop_key, "absolute_legacy")

    def test_unknown_key_rejected(self):
        with self.assertRaises(ValueError):
            GroundwaterPatchDatasetMultiOut(["x.npz"], STATS, crop_key="hash")


class AllowedBackboneVariantTests(unittest.TestCase):
    def forward(self, model, channels):
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, channels, 64, 64))
        self.assertEqual(tuple(out.shape), (1, 25, 64, 64))
        self.assertTrue(torch.isfinite(out).all())

    def test_cta_every_learned_variant(self):
        for variant in ("v1_factorized_levels", "v2_joint_embedding", "v3_factorized_continuous",
                        "v4_joint_continuous"):
            with self.subTest(variant=variant):
                self.forward(RegimeConditionedCTAUNet(conditioning_variant=variant, normalized_level_centers=CENTERS,
                                                      out_ch=25, base=16, t_embed_dim=25, t_heads=5, pool_stride=4,
                                                      film_embed_dim=8, film_hidden_dim=32), 3)

    def test_mstmo_supported_variants(self):
        for variant in ("v1_factorized_levels", "v2_joint_embedding", "v3_factorized_continuous"):
            with self.subTest(variant=variant):
                self.forward(RegimeConditionedMSTMO(conditioning_variant=variant, normalized_level_centers=CENTERS,
                                                    out_ch=25, base=16, attn_heads=4, film_embed_dim=8,
                                                    film_hidden_dim=32), 3)

    def test_mstmo_v4_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            RegimeConditionedMSTMO(conditioning_variant="v4_joint_continuous", normalized_level_centers=CENTERS)

    def test_fno_raw_channels(self):
        self.forward(FNO2D(in_ch=3, out_ch=25, width=8, modes1=4, modes2=4, depth=2), 3)


if __name__ == "__main__":
    unittest.main()
