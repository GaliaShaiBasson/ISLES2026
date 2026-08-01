"""
Trainer variants for the loss-function study (Phase 2).

Each class only overrides `_build_loss`, following the same pattern nnU-Net
uses internally for `DC_and_CE_loss`. After running `install_custom_trainers.py`,
train each with, e.g.:

    nnUNetv2_train <DATASET_ID> 3d_fullres 0 -tr nnUNetTrainerFocalTversky

nnU-Net auto-discovers any `nnUNetTrainer` subclass placed under
`nnunetv2/training/nnUNetTrainer/variants/`, matched by class name via the
`-tr` flag -- no registration step needed beyond the file being importable.

NOTE: `_build_loss`'s exact internals can shift slightly between nnunetv2
releases. This follows the structure as of nnunetv2 2.4/2.5; if your
installed version errors here, compare against
`nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py::_build_loss` in your
installed package and adjust the deep-supervision-weight computation to
match (the loss classes in `losses.py` themselves don't need to change).
"""
import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

try:
    # Works once installed inside nnunetv2/.../variants/isles26_ext/ as a
    # proper subpackage (see install_custom_trainers.py).
    from .losses import FocalLoss, TverskyLoss, FocalTverskyLoss, DiceOnlyLoss
except ImportError:
    # Fallback for local testing directly from this project folder, where
    # there's no enclosing package.
    from losses import FocalLoss, TverskyLoss, FocalTverskyLoss, DiceOnlyLoss


def _wrap_with_deep_supervision(trainer: nnUNetTrainer, loss):
    if not trainer.enable_deep_supervision:
        return loss
    deep_supervision_scales = trainer._get_deep_supervision_scales()
    weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
    weights[-1] = 0  # nnU-Net convention: ignore the lowest-resolution head
    weights = weights / weights.sum()
    return DeepSupervisionWrapper(loss, weights)


class nnUNetTrainerDiceOnly(nnUNetTrainer):
    """Control condition: plain soft Dice, no CE, no class-imbalance handling
    beyond Dice's own implicit weighting."""

    def _build_loss(self):
        loss = DiceOnlyLoss(smooth=1e-5, ignore_index=self.label_manager.ignore_label or -100)
        return _wrap_with_deep_supervision(self, loss)


class nnUNetTrainerFocal(nnUNetTrainer):
    """Focal loss only (no explicit Dice term) -- isolates the effect of
    hard-example weighting from the region-overlap term."""

    def _build_loss(self):
        loss = FocalLoss(gamma=2.0, alpha=0.25, ignore_index=self.label_manager.ignore_label or -100)
        return _wrap_with_deep_supervision(self, loss)


class nnUNetTrainerTversky(nnUNetTrainer):
    """Tversky loss with beta > alpha (recall-favoring) -- standard choice
    for small structures where missing lesion voxels is worse than a bit of
    over-segmentation."""

    def _build_loss(self):
        loss = TverskyLoss(alpha=0.3, beta=0.7, smooth=1e-5, ignore_index=self.label_manager.ignore_label or -100)
        return _wrap_with_deep_supervision(self, loss)


class nnUNetTrainerFocalTversky(nnUNetTrainer):
    """Focal-Tversky -- combines both mechanisms above. Expected best
    performer on the small-lesion bin if the imbalance hypothesis holds;
    the interesting result either way is *how much* it helps versus plain
    Tversky or Focal alone."""

    def _build_loss(self):
        loss = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-5,
                                 ignore_index=self.label_manager.ignore_label or -100)
        return _wrap_with_deep_supervision(self, loss)
