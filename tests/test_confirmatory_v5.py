import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.shared.data.concentration import LOG10_MAX, LOG10_MIN, bounded_log10_tensor
from src.shared.data.dataset_patch_multiout import GroundwaterPatchDatasetMultiOut
from src.shared.data.io import filter_excluded_files
from src.shared.eval.eval_transport_unified_logc_global_plumemask import ssim_fixed
from src.transport_surrogates.baselines.focal_frequency_loss import FocalFrequencyLoss
from src.transport_surrogates.baselines.temporal_attn_unet import CTAUNet
from src.transport_surrogates.baselines.train_unet_multiout_logc_B_aspp_attn import (
    UNet_ASPP_Attn,
)
from src.transport_surrogates.confirmatory_v5.conditioning import RegimeConditionedCTAUNet
from src.transport_surrogates.confirmatory_v5.conditioning_mstmo import (
    RegimeConditionedMSTMO,
)
from src.transport_surrogates.confirmatory_v5.train import Config, compute_loss
from scripts.build_obj1up_compositional_split import build_compositional_split


ROOT = Path(__file__).resolve().parents[1]


class ConfirmatoryV5Tests(unittest.TestCase):
    @staticmethod
    def _conditioning_centers():
        return {"alpha_L": [-1.2247448, 0.0, 1.2247448], "alpha_T_ratio": [-1.0, 1.0]}

    def test_regime_conditioning_variants_preserve_output_shape_and_small_parameter_delta(self):
        baseline = CTAUNet(in_ch=1, out_ch=25, base=8)
        baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
        inputs = torch.randn(2, 3, 16, 16)
        for variant in (
            "v1_factorized_levels",
            "v2_joint_embedding",
            "v3_factorized_continuous",
        ):
            model = RegimeConditionedCTAUNet(
                conditioning_variant=variant,
                normalized_level_centers=self._conditioning_centers(),
                base=8,
            )
            self.assertEqual(model(inputs).shape, (2, 25, 16, 16))
            delta = sum(parameter.numel() for parameter in model.parameters()) - baseline_count
            self.assertGreater(delta, 0)
            self.assertLess(delta, 10_000)

    def test_factorized_conditioning_encoders_receive_gradient(self):
        inputs = torch.randn(2, 3, 16, 16)
        inputs[:, 1, :, :] = torch.tensor([-1.2247448, 1.2247448])[:, None, None]
        inputs[:, 2, :, :] = torch.tensor([-1.0, 1.0])[:, None, None]
        for variant in ("v1_factorized_levels", "v3_factorized_continuous"):
            model = RegimeConditionedCTAUNet(
                conditioning_variant=variant,
                normalized_level_centers=self._conditioning_centers(),
                base=8,
            )
            model(inputs).square().mean().backward()
            encoder_grad = sum(
                float(parameter.grad.abs().sum())
                for name, parameter in model.named_parameters()
                if "_encoder" in name and parameter.grad is not None
            )
            self.assertGreater(encoder_grad, 0.0)

    def test_mstmo_regime_conditioning_preserves_shape_and_small_parameter_delta(self):
        baseline = UNet_ASPP_Attn(in_ch=1, out_ch=25, base=8, attn_heads=4)
        baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
        inputs = torch.randn(2, 3, 16, 16)
        for variant in (
            "v1_factorized_levels",
            "v2_joint_embedding",
            "v3_factorized_continuous",
        ):
            model = RegimeConditionedMSTMO(
                conditioning_variant=variant,
                normalized_level_centers=self._conditioning_centers(),
                base=8,
                attn_heads=4,
            )
            self.assertEqual(model(inputs).shape, (2, 25, 16, 16))
            delta = sum(parameter.numel() for parameter in model.parameters()) - baseline_count
            self.assertGreater(delta, 0)
            self.assertLess(delta, 10_000)

    def test_mstmo_factorized_encoders_receive_gradient(self):
        inputs = torch.randn(2, 3, 16, 16)
        inputs[:, 1, :, :] = torch.tensor([-1.2247448, 1.2247448])[:, None, None]
        inputs[:, 2, :, :] = torch.tensor([-1.0, 1.0])[:, None, None]
        for variant in ("v1_factorized_levels", "v3_factorized_continuous"):
            model = RegimeConditionedMSTMO(
                conditioning_variant=variant,
                normalized_level_centers=self._conditioning_centers(),
                base=8,
                attn_heads=4,
            )
            model(inputs).square().mean().backward()
            encoder_grad = sum(
                float(parameter.grad.abs().sum())
                for name, parameter in model.named_parameters()
                if "_encoder" in name and parameter.grad is not None
            )
            self.assertGreater(encoder_grad, 0.0)

    def test_mstmo_rejects_raw_channel_variant(self):
        with self.assertRaises(ValueError):
            RegimeConditionedMSTMO(
                conditioning_variant="v0_raw_channels",
                normalized_level_centers=self._conditioning_centers(),
                base=8,
                attn_heads=4,
            )

    def test_joint_heldout_embedding_row_has_zero_gradient(self):
        model = RegimeConditionedCTAUNet(
            conditioning_variant="v2_joint_embedding",
            normalized_level_centers=self._conditioning_centers(),
            base=8,
        )
        inputs = torch.randn(2, 3, 16, 16)
        inputs[:, 1, :, :] = torch.tensor([-1.2247448, 1.2247448])[:, None, None]
        inputs[:, 2, :, :] = torch.tensor([-1.0, 1.0])[:, None, None]
        model(inputs).square().mean().backward()
        gradient = model.joint_encoder.weight.grad
        self.assertGreater(float(gradient[0].abs().sum()), 0.0)
        self.assertGreater(float(gradient[5].abs().sum()), 0.0)
        self.assertEqual(float(gradient[3].abs().sum()), 0.0)

    def test_compositional_split_preserves_geology_and_marginal_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "split.json"
            payload = build_compositional_split(
                ROOT / "splits" / "param_split_grouped_k_v5.json",
                ROOT / "metadata" / "parameter_registry_162_v5.csv",
                output,
            )
        self.assertEqual(
            {name: len(payload["splits"][name]["param_ids"]) for name in ("train", "val", "test")},
            {"train": 90, "val": 15, "test": 36},
        )
        self.assertEqual(len(payload["compositional_evaluation"]["heldout_pair_param_ids"]), 6)

    def test_grouped_split_has_disjoint_geological_cells(self):
        payload = json.loads(
            (ROOT / "splits" / "param_split_grouped_k_v5.json").read_text(encoding="utf-8")
        )
        groups = {}
        for split_name in ("train", "val", "test"):
            groups[split_name] = {
                (row["sigma2Y"], row["correlation_length"], row["anisotropy"])
                for row in payload["splits"][split_name]["base_k_groups"]
            }
        self.assertFalse(groups["train"] & groups["val"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["val"] & groups["test"])
        self.assertEqual(sum(map(len, groups.values())), 27)

    def test_integrity_manifest_excludes_two_complete_six_file_groups(self):
        path = ROOT / "metadata" / "k_integrity_exclusions_v5.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 12)
        keys = {row["sample_key"] for row in rows}
        included = str(ROOT / "data" / "param_000" / "real_001.npz")
        files = [str(ROOT / "data" / key) for key in sorted(keys)] + [included]
        self.assertEqual(filter_excluded_files(files, str(path)), [included])

    def test_fixed_validation_crop_is_stable_and_contains_source(self):
        dataset = GroundwaterPatchDatasetMultiOut(
            ["param_000/real_001.npz"],
            {"k_mean": 0.0, "k_std": 1.0},
            patch_size=320,
            crop_mode="fixed",
            fixed_crop_seed=20260715,
            require_source_in_crop=True,
            source_row=399,
            source_col=199,
        )
        first = dataset._fixed_crop_origin(dataset.files[0], 600, 400)
        second = dataset._fixed_crop_origin(dataset.files[0], 600, 400)
        self.assertEqual(first, second)
        y0, x0 = first
        self.assertLessEqual(y0, 399)
        self.assertLess(399, y0 + 320)
        self.assertLessEqual(x0, 199)
        self.assertLess(199, x0 + 320)

    def test_bounded_output_is_finite_and_differentiable(self):
        raw = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0], requires_grad=True)
        bounded = bounded_log10_tensor(raw)
        self.assertTrue(torch.all(bounded >= LOG10_MIN))
        self.assertTrue(torch.all(bounded <= LOG10_MAX))
        bounded[1:4].sum().backward()
        self.assertTrue(torch.all(raw.grad[1:4] > 0.0))

    def test_confirmatory_loss_is_finite_for_nonfinite_raw_outputs(self):
        raw = torch.tensor(
            [[[[float("nan"), float("inf")], [float("-inf"), 0.0]]]],
            dtype=torch.float32,
        ).repeat(1, 3, 1, 1)
        raw.requires_grad_(True)
        target = torch.zeros_like(raw)
        loss = compute_loss(raw, target, Config(data_root="unused"), FocalFrequencyLoss())
        self.assertTrue(all(torch.isfinite(value) for value in loss.values()))
        loss["total"].backward()

    def test_ssim_uses_requested_nondefault_bounds(self):
        ground_truth = np.array([[-5.0, 0.0], [5.0, 15.0]], dtype=np.float32)
        prediction = np.array([[-10.0, 0.0], [5.0, 20.0]], dtype=np.float32)
        clipped_gt = np.clip(ground_truth, 0.0, 10.0)
        clipped_prediction = np.clip(prediction, 0.0, 10.0)
        # The 2x2 arrays are too small for the default SSIM window, so repeat to 8x8.
        expected = ssim_fixed(
            np.tile(clipped_gt, (4, 4)),
            np.tile(clipped_prediction, (4, 4)),
            data_range=10.0,
            bounds=(0.0, 10.0),
        )
        repeated = ssim_fixed(
            np.tile(ground_truth, (4, 4)),
            np.tile(prediction, (4, 4)),
            data_range=10.0,
            bounds=(0.0, 10.0),
        )
        self.assertAlmostEqual(repeated, expected, places=7)

    def test_control_metadata_is_complete_and_cyclic_relabeling_is_deranged(self):
        control_dir = ROOT / "metadata" / "confirmatory_controls_v5"
        with (control_dir / "matched_transport_metadata.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            matched = {row["param_folder"]: (row["alpha_L"], row["alpha_T_ratio"]) for row in csv.DictReader(handle)}
        with (control_dir / "permuted_transport_metadata.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            permuted = {row["param_folder"]: (row["alpha_L"], row["alpha_T_ratio"]) for row in csv.DictReader(handle)}
        self.assertEqual(len(matched), 162)
        self.assertEqual(set(matched), set(permuted))
        self.assertTrue(all(matched[key] != permuted[key] for key in matched))
        expected_cycle = {}
        for index in range(6):
            true_key = f"param_{index:03d}"
            next_key = f"param_{(index + 1) % 6:03d}"
            expected_cycle[matched[true_key]] = matched[next_key]
        for index in range(162):
            key = f"param_{index:03d}"
            self.assertEqual(permuted[key], expected_cycle[matched[key]])

    def test_counterfactual_sweep_has_six_complete_fixed_pair_mappings(self):
        sweep_dir = ROOT / "metadata" / "confirmatory_controls_v5" / "counterfactual_sweep"
        manifest = json.loads(
            (sweep_dir / "COUNTERFACTUAL_SWEEP_MANIFEST.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(manifest["files"]), 6)
        observed_pairs = set()
        for entry in manifest["files"]:
            path = ROOT / entry["path"]
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 162)
            pairs = {(float(row["alpha_L"]), float(row["alpha_T_ratio"])) for row in rows}
            self.assertEqual(len(pairs), 1)
            observed_pairs.update(pairs)
        self.assertEqual(observed_pairs, {(a, r) for a in (0.62, 6.2, 62.0) for r in (0.1, 1.0)})


if __name__ == "__main__":
    unittest.main()
