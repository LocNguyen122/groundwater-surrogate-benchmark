import unittest

import numpy as np

from src.shared.eval.centroid_error import compute_concentration_centroid_error


class CentroidMetricTests(unittest.TestCase):
    def test_full_field_concentration_weighted_centroid(self):
        ground_truth = np.zeros((4, 5), dtype=np.float32)
        prediction = np.zeros((4, 5), dtype=np.float32)
        ground_truth[1, 1] = 2.0
        prediction[1, 3] = 2.0
        error, eligible = compute_concentration_centroid_error(ground_truth, prediction)
        self.assertEqual(eligible, 1)
        self.assertAlmostEqual(error, 2.0)

    def test_empty_field_is_ineligible(self):
        empty = np.zeros((3, 3), dtype=np.float32)
        error, eligible = compute_concentration_centroid_error(empty, empty)
        self.assertEqual(eligible, 0)
        self.assertTrue(np.isnan(error))


if __name__ == "__main__":
    unittest.main()
