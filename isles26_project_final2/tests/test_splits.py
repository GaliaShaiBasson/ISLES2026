import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_prep"))
from create_stratified_splits import create_splits, subject_group


class SplitTests(unittest.TestCase):
    def test_subject_sessions_stay_together_and_each_case_is_validated_once(self):
        rows = []
        for index in range(12):
            subject = f"ATLAS_sub{index:02d}"
            for session in (["1", "2"] if index == 0 else ["1"]):
                rows.append(
                    {
                        "case_id": f"{subject}_ses{session}",
                        "center": f"R{index % 3:03d}",
                        "size_bin": ["small", "medium", "large"][index % 3],
                    }
                )
        splits = create_splits(pd.DataFrame(rows), n_splits=5, seed=2026)
        validation = [case for split in splits for case in split["val"]]
        self.assertCountEqual(validation, [row["case_id"] for row in rows])
        fold_by_group = {}
        for fold, split in enumerate(splits):
            for case in split["val"]:
                group = subject_group(case)
                self.assertEqual(fold_by_group.setdefault(group, fold), fold)


if __name__ == "__main__":
    unittest.main()
