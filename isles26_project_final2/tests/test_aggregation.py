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
            outputs = [tmp / name for name in ["results.csv", "overall.csv", "size.csv", "center.csv", "size_center.csv", "paired.csv"]]
            command = [
                sys.executable, str(ROOT / "evaluation" / "aggregate_results.py"), str(input_csv),
                "--out-combined", str(outputs[0]),
                "--out-summary", str(outputs[1]),
                "--out-summary-by-size", str(outputs[2]),
                "--out-summary-by-center", str(outputs[3]),
                "--out-summary-by-size-center", str(outputs[4]),
                "--out-paired", str(outputs[5]),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            for path in outputs:
                self.assertTrue(path.is_file(), path)

    def test_unpaired_experiments_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            baseline = tmp / "baseline.csv"
            candidate = tmp / "candidate.csv"
            pd.DataFrame([
                {"case_id": "a", "experiment": "baseline", "dice": 0.5, "hd95_mm": 3.0},
                {"case_id": "b", "experiment": "baseline", "dice": 0.6, "hd95_mm": 2.0},
            ]).to_csv(baseline, index=False)
            pd.DataFrame([
                {"case_id": "a", "experiment": "sampling", "dice": 0.7, "hd95_mm": 1.0},
            ]).to_csv(candidate, index=False)
            result = subprocess.run(
                [sys.executable, str(ROOT / "evaluation" / "aggregate_results.py"), str(baseline), str(candidate)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("same evaluation cohort", result.stderr)

    def test_paired_statistics_use_case_level_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            baseline = tmp / "baseline.csv"
            candidate = tmp / "candidate.csv"
            paired = tmp / "paired.csv"
            pd.DataFrame([
                {"case_id": "a", "experiment": "baseline", "dice": 0.5, "hd95_mm": 3.0},
                {"case_id": "b", "experiment": "baseline", "dice": 0.6, "hd95_mm": 4.0},
            ]).to_csv(baseline, index=False)
            pd.DataFrame([
                {"case_id": "a", "experiment": "sampling", "dice": 0.7, "hd95_mm": 2.0},
                {"case_id": "b", "experiment": "sampling", "dice": 0.8, "hd95_mm": 3.0},
            ]).to_csv(candidate, index=False)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "evaluation" / "aggregate_results.py"),
                    str(baseline),
                    str(candidate),
                    "--out-combined", str(tmp / "combined.csv"),
                    "--out-summary", str(tmp / "summary.csv"),
                    "--out-summary-by-size", str(tmp / "size.csv"),
                    "--out-summary-by-center", str(tmp / "center.csv"),
                    "--out-summary-by-size-center", str(tmp / "size_center.csv"),
                    "--out-paired", str(paired),
                    "--bootstrap-iterations", "200",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = pd.read_csv(paired)
            dice = result[result["metric"] == "dice"].iloc[0]
            hd95 = result[result["metric"] == "hd95_mm"].iloc[0]
            self.assertAlmostEqual(dice["mean_improvement"], 0.2)
            self.assertAlmostEqual(hd95["mean_improvement"], 1.0)


if __name__ == "__main__":
    unittest.main()
