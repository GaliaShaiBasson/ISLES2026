import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isles26 import _torch_version_supported


class RunnerUtilityTests(unittest.TestCase):
    def test_torch_minimum_patch_version_is_enforced(self):
        self.assertFalse(_torch_version_supported("2.1.1"))
        self.assertTrue(_torch_version_supported("2.1.2"))

    def test_torch_29_is_excluded(self):
        self.assertFalse(_torch_version_supported("2.9.0+cu128"))
        self.assertTrue(_torch_version_supported("2.10.0"))


if __name__ == "__main__":
    unittest.main()
