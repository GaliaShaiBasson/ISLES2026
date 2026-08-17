import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from custom_trainers.sampling_utils import curriculum_power, inverse_volume_probabilities, sampling_diagnostics


class SamplingUtilsTests(unittest.TestCase):
    def test_power_zero_is_uniform(self):
        probs = inverse_volume_probabilities([1, 10, 100], power=0)
        np.testing.assert_allclose(probs, [1/3, 1/3, 1/3])

    def test_inverse_volume_favors_small_lesions(self):
        probs = inverse_volume_probabilities([1, 2, 4], power=1)
        self.assertAlmostEqual(float(probs.sum()), 1.0)
        self.assertGreater(probs[0], probs[1])
        self.assertGreater(probs[1], probs[2])

    def test_curriculum_phases(self):
        self.assertEqual(curriculum_power(0, 1000), ("uniform", 0.0))
        self.assertEqual(curriculum_power(400, 1000), ("mild", 0.5))
        self.assertEqual(curriculum_power(800, 1000), ("full", 1.0))

    def test_sampling_diagnostics_report_concentration(self):
        diagnostics = sampling_diagnostics([0.5, 0.25, 0.25])
        self.assertEqual(diagnostics["max_to_min_ratio"], 2.0)
        self.assertAlmostEqual(diagnostics["effective_sample_size"], 8 / 3)


if __name__ == "__main__":
    unittest.main()
