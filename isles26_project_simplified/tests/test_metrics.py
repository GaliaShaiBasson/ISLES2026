import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from compute_metrics import dice_score, hausdorff_distance_95


class MetricTests(unittest.TestCase):
    def test_dice_identity(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[1:3, 1:3, 1:3] = True
        self.assertEqual(dice_score(mask, mask), 1.0)

    def test_hd95_identity(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[1:3, 1:3, 1:3] = True
        self.assertEqual(hausdorff_distance_95(mask, mask, (1, 1, 1)), 0.0)

    def test_hd95_empty_is_nan(self):
        empty = np.zeros((3, 3, 3), dtype=bool)
        filled = empty.copy()
        filled[1, 1, 1] = True
        self.assertTrue(np.isnan(hausdorff_distance_95(empty, filled, (1, 1, 1))))


if __name__ == "__main__":
    unittest.main()
