import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # permits the lightweight test suite before GPU PyTorch setup
    torch = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if torch is not None:
    from custom_trainers.losses import FocalLoss, TverskyLoss


class LossTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_losses_are_finite_and_differentiable_with_ignore_label(self):
        logits = torch.randn(2, 2, 4, 4, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 4, 4))
        target[0, 0, 0, 0] = 255
        total = FocalLoss(ignore_index=255)(logits, target) + TverskyLoss(ignore_index=255)(logits, target)
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
