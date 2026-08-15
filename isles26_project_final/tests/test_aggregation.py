import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class AggregationTests(unittest.TestCase):
    def test_center_and_size_summaries_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows = [
                {"case_id": "a", "experiment": "baseline", "dice": 0.5, "hd95_mm": 3.0, "size_bin": "small", "center": "R001"},
                {"case_id": "b", "experiment": "baseline", "dice": 0.8, "hd95_mm": 2.0, "size_bin": "large", "center": "R002"},
            ]
            input_csv = tmp / "results_baseline.csv"
            pd.DataFrame(rows).to_csv(input_csv, index=False)
            outputs = [tmp / name for name in ["results.csv", "overall.csv", "size.csv", "center.csv", "size_center.csv"]]
            command = [
                sys.executable, str(ROOT / "evaluation" / "aggregate_results.py"), str(input_csv),
                "--out-combined", str(outputs[0]),
                "--out-summary", str(outputs[1]),
                "--out-summary-by-size", str(outputs[2]),
                "--out-summary-by-center", str(outputs[3]),
                "--out-summary-by-size-center", str(outputs[4]),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            for path in outputs:
                self.assertTrue(path.is_file(), path)


if __name__ == "__main__":
    unittest.main()
