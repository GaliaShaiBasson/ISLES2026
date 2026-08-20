import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from custom_trainers.losses import DiceOnlyLoss, FocalLoss, FocalTverskyLoss, TverskyLoss


class LossTests(unittest.TestCase):
    def test_losses_are_finite_and_differentiable_with_ignore_label(self):
        logits = torch.randn(2, 2, 4, 4, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 4, 4))
        target[0, 0, 0, 0] = 255
        losses = [
            FocalLoss(ignore_index=255),
            TverskyLoss(ignore_index=255),
            FocalTverskyLoss(ignore_index=255),
            DiceOnlyLoss(ignore_index=255),
        ]
        total = sum(loss(logits, target) for loss in losses)
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
