import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from plot_results import (
    save_by_center,
    save_by_size,
    save_empty_prediction_rate,
    save_overall,
    save_size_center_heatmaps,
    save_volume_scatter,
)


class PlottingTests(unittest.TestCase):
    def test_all_expected_figures_are_created(self):
        rows = []
        for experiment, offset in (("baseline", 0.0), ("sampling", 0.1)):
            for index, size in enumerate(("small", "medium", "large")):
                rows.append(
                    {
                        "experiment": experiment,
                        "case_id": f"{experiment}_{index}",
                        "dice": 0.5 + offset + 0.1 * index,
                        "hd95_mm": 5.0 - index,
                        "hd95_penalized_mm": 5.0 - index,
                        "pred_empty": index == 0 and experiment == "baseline",
                        "size_bin": size,
                        "center": f"R{index % 2:03d}",
                        "lesion_volume_mm3": 10 ** (index + 1),
                    }
                )
        frame = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            save_overall(frame, output)
            save_empty_prediction_rate(frame, output)
            save_by_size(frame, output)
            save_by_center(frame, output)
            save_size_center_heatmaps(frame, output)
            save_volume_scatter(frame, output)
            expected = {
                "overall_dice.png",
                "overall_hd95.png",
                "empty_prediction_rate.png",
                "dice_by_size_bin.png",
                "dice_by_center.png",
                "dice_vs_volume_scatter.png",
                "dice_center_by_size_baseline.png",
                "dice_center_by_size_sampling.png",
            }
            self.assertTrue(expected.issubset({path.name for path in output.glob("*.png")}))


if __name__ == "__main__":
    unittest.main()
