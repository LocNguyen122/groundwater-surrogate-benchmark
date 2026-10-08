"""Public synthetic regression checks; no simulator fields or model execution."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from src.transport_surrogates.confirmatory_v5.train import Config
from src.shared.data.dataset_patch_multiout import GroundwaterPatchDatasetMultiOut

ROOT = Path(__file__).resolve().parents[1]


class ReviewCorrections(unittest.TestCase):
    def test_anchor_alias_preserves_coordinates(self):
        cfg = Config(data_root="authorized-data", source_row=399, source_col=199)
        self.assertEqual((cfg.validation_anchor_row, cfg.validation_anchor_col), (399, 199))

    def test_downstream_anchor_does_not_guarantee_source_in_crop(self):
        ds = GroundwaterPatchDatasetMultiOut([], {"k_mean": 0., "k_std": 1.}, crop_mode="fixed",
                                            fixed_crop_seed=20260715, require_source_in_crop=True,
                                            crop_key="relative", source_row=399, source_col=199)
        origins = [ds._fixed_crop_origin(f"param_000/real_{i:03d}.npz", 600, 400) for i in range(100)]
        self.assertTrue(all(y <= 399 < y + 320 and x <= 199 < x + 320 for y, x in origins))
        self.assertTrue(any(y > 200 for y, _ in origins))

    def test_conductivity_panels_share_global_range(self):
        path = ROOT / "scripts/build_v7_example_figure.py"
        spec = importlib.util.spec_from_file_location("example_figure", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.shared_conductivity_limits([np.array([1., 2.]), np.array([-3., 8.])]), (-3., 8.))
        with self.assertRaises(ValueError):
            module.shared_conductivity_limits([np.array([np.nan])])

    def test_well_confusion_rates(self):
        spec = importlib.util.spec_from_file_location("review_diagnostics", ROOT / "scripts/review_diagnostics.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        result = module.rates(np.array([8., 3., 7., 2.]))
        self.assertAlmostEqual(result['false_negative_rate'], .2)
        self.assertAlmostEqual(result['false_positive_rate'], .3)
        self.assertAlmostEqual(result['balanced_accuracy'], .75)


if __name__ == "__main__":
    unittest.main()
