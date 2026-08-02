import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_prep"))
from metadata_utils import assign_size_bin, sampling_weights_from_volume


class MetadataUtilsTests(unittest.TestCase):
    def test_size_bins_handle_repeated_values(self):
        result = assign_size_bin(pd.Series([10, 10, 10, 20, 20, 30]))
        self.assertEqual(set(result.dropna()), {"small", "medium", "large"})
        self.assertEqual(len(result), 6)

    def test_size_bins_handle_single_case(self):
        result = assign_size_bin(pd.Series([42.0]))
        self.assertEqual(result.iloc[0], "small")

    def test_inverse_volume_weights_are_normalized(self):
        weights = sampling_weights_from_volume(pd.Series([1.0, 2.0, 4.0]))
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertTrue(np.all(np.diff(weights.to_numpy()) < 0))


if __name__ == "__main__":
    unittest.main()
