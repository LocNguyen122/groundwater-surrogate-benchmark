"""CPU-only guards for new descriptive revision diagnostics."""
import importlib.util
from pathlib import Path
import unittest
import warnings
import numpy as np
import pandas as pd
import json
from src.transport_surrogates.confirmatory_v5.train import warn_legacy_anchor_flags

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("revision_diagnostics", ROOT / "scripts/revision_diagnostics.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RevisionDiagnosticsTests(unittest.TestCase):
    def test_illustrative_cost_has_explicit_denominator(self):
        self.assertAlmostEqual(module.cost_per_decision(np.array([8, 3, 7, 2]), 5), 13 / 20)

    def test_invalid_costs_rejected(self):
        for counts, cost in (([0, 0, 0, 0], 1), ([8, -1, 7, 2], 1), ([8, 3, 7, 2], -1)):
            with self.assertRaises(ValueError):
                module.cost_per_decision(np.array(counts), cost)

    def test_boolean_metadata_is_not_truthy_string(self):
        self.assertEqual(module.flag(pd.Series(["False", "True"])).tolist(), [False, True])
        with self.assertRaises(ValueError):
            module.flag(pd.Series(["unknown"]))

    def test_retained_groups_and_qa_counts(self):
        counts, _, correlations = module.qa_tables()
        self.assertEqual(counts['T']['retained_cases'], 794)
        self.assertEqual(counts['T']['cell_realization_groups_with_retained_cases'], 135)
        self.assertEqual(counts['T']['pure_no_blowup_failures'], 11)
        self.assertEqual(counts['T']['other_run_or_size_failures'], 5)
        self.assertEqual(counts['T2']['retained_cases'], 797)
        self.assertEqual(len(correlations), 270)

    def test_counterfactual_reconstructs_original_consistency(self):
        frame = pd.read_csv(ROOT / 'results/v9/counterfactual_case_scores.csv')
        reference = json.loads((ROOT / 'figures/v7/fig_v7_counterfactual.json').read_text())
        result = module.counterfactual(frame, reference)
        self.assertEqual(result['consistency']['plume_ssim']['seed_rows_correct_highest'], 60)
        with self.assertRaises(ValueError):
            module.counterfactual(pd.concat([frame, frame.iloc[:1]]), reference)

    def test_legacy_anchor_alias_warns_without_changing_coordinates(self):
        with self.assertWarns(FutureWarning):
            warn_legacy_anchor_flags(['--source_row=399'])
        with warnings.catch_warnings(record=True) as caught:
            warn_legacy_anchor_flags(['--validation_anchor_row', '399'])
        self.assertFalse(caught)
