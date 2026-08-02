"""Loss-function trainer variants pinned to nnU-Net 2.8.1 behavior."""
from __future__ import annotations

import numpy as np
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from .losses import DiceOnlyLoss, FocalLoss, FocalTverskyLoss, TverskyLoss


def _ignore_label(trainer: nnUNetTrainer) -> int:
    value = trainer.label_manager.ignore_label
    return -100 if value is None else int(value)


def _wrap_with_deep_supervision(trainer: nnUNetTrainer, loss):
    if not trainer.enable_deep_supervision:
        return loss
    scales = trainer._get_deep_supervision_scales()
    weights = np.array([1 / (2**index) for index in range(len(scales))], dtype=float)
    # Mirror nnU-Net 2.8.1's DDP workaround exactly.
    weights[-1] = 1e-6 if trainer.is_ddp and not trainer._do_i_compile() else 0
    weights /= weights.sum()
    return DeepSupervisionWrapper(loss, weights)


class nnUNetTrainerDiceOnly(nnUNetTrainer):
    def _build_loss(self):
        return _wrap_with_deep_supervision(self, DiceOnlyLoss(ignore_index=_ignore_label(self)))


class nnUNetTrainerFocal(nnUNetTrainer):
    def _build_loss(self):
        return _wrap_with_deep_supervision(
            self,
            FocalLoss(gamma=2.0, alpha=0.25, ignore_index=_ignore_label(self)),
        )


class nnUNetTrainerTversky(nnUNetTrainer):
    def _build_loss(self):
        return _wrap_with_deep_supervision(
            self,
            TverskyLoss(alpha=0.3, beta=0.7, ignore_index=_ignore_label(self)),
        )


class nnUNetTrainerFocalTversky(nnUNetTrainer):
    def _build_loss(self):
        return _wrap_with_deep_supervision(
            self,
            FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75, ignore_index=_ignore_label(self)),
        )
