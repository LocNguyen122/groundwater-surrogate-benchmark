import unittest

import numpy as np

from src.shared.data.concentration import (
    LOG10_MAX,
    LOG10_MIN,
    SSIM_DATA_RANGE,
    clip_log10_for_ssim,
    log10_to_physical,
    physical_to_log10,
)


class ConcentrationTransformTests(unittest.TestCase):
    def test_unclipped_transform_and_inverse(self) -> None:
        values = np.array([-1.0, 0.0, 1e-8, 1.0, 1e3], dtype=np.float32)
        transformed = physical_to_log10(values, clip_for_ssim=False)
        # float32 log10(1e-12) is -12.000000953674316; compare at float32 precision, not 6 decimal places.
        np.testing.assert_allclose(float(transformed[0]), LOG10_MIN, rtol=0, atol=1e-5)
        np.testing.assert_allclose(float(transformed[-1]), 3.0, rtol=0, atol=1e-5)
        restored = log10_to_physical(transformed)
        np.testing.assert_allclose(restored[1:], np.clip(values[1:], 0.0, None), rtol=1e-5, atol=1e-10)

    def test_ssim_view_is_symmetric_and_fixed_range(self) -> None:
        raw = np.array([-20.0, -12.0, 0.0, 2.0, 9.0], dtype=np.float32)
        clipped = clip_log10_for_ssim(raw)
        np.testing.assert_array_equal(clipped, np.array([-12.0, -12.0, 0.0, 2.0, 2.0], dtype=np.float32))
        self.assertEqual(SSIM_DATA_RANGE, LOG10_MAX - LOG10_MIN)


if __name__ == "__main__":
    unittest.main()
