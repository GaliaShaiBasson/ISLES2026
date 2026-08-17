"""Loss-function trainer variants pinned to nnU-Net 2.8.1 behavior."""
from __future__ import annotations

import numpy as np
import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from .losses import DiceOnlyLoss, DiceTverskyLoss, FocalLoss, FocalTverskyLoss, TverskyLoss


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


class nnUNetTrainerTverskyMild(nnUNetTrainer):
    """Dice + softened-asymmetry Tversky (alpha=0.4/beta=0.6), not Tversky alone.

    Follow-up to nnUNetTrainerTversky (alpha=0.3/beta=0.7, Tversky-only): that variant
    genuinely improved small-lesion recall (best true-positive/false-negative rate of
    every condition tested on the real dataset) but lost enough precision (extra
    false-positive components) that lesion-wise F1 still landed below baseline. Two
    changes here, both aimed at keeping the recall gain while reining in the
    false-positive cost -- see DiceTverskyLoss's docstring and the 2026-08-18 CLAUDE.md
    entry for the evidence this is based on:
      - alpha/beta pulled toward 0.5 (less aggressive false-negative penalty)
      - Dice added back into the loss (nnUNetTrainerTversky drops it entirely)
    """

    def _build_loss(self):
        return _wrap_with_deep_supervision(
            self,
            DiceTverskyLoss(tversky_alpha=0.4, tversky_beta=0.6, ignore_index=_ignore_label(self)),
        )


class _Epochs250Mixin:
    """Rescales the poly-LR schedule to a 250-epoch budget instead of nnU-Net's default 1000.

    Mirrors nnunetv2's own ``nnUNetTrainer_250epochs`` convention: only
    ``num_epochs`` changes, so the schedule shape (SGD + PolyLRScheduler) is
    preserved, just compressed. Applied identically to every loss variant so
    the cross-condition comparison stays controlled.

    Also drops ``save_every`` from nnU-Net's default of 50 epochs to 10: these
    runs are meant to survive an unattended, unsupervised multi-hour stretch, so
    the worst-case amount of training lost to an unnoticed crash matters more
    here than the small extra disk-write overhead of checkpointing more often.

    __init__ must declare nnU-Net's exact named parameters (not *args/**kwargs):
    nnUNetTrainer.__init__ builds self.my_init_kwargs via
    ``inspect.signature(self.__init__).parameters`` -- since `self.__init__`
    resolves through the MRO to *this* __init__, a generic *args/**kwargs
    signature makes that introspection collect the wrong parameter names and
    crash with `KeyError: 'args'` the moment a trainer using this mixin is
    actually instantiated. This bit us for real: caught during pre-launch
    testing, before it could silently break every 250-epoch trainer on the
    unattended overnight run. Mirrors nnUNetTrainerDebugMixin's pattern.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 250
        self.save_every = 10


class nnUNetTrainerBaseline_250epochs(_Epochs250Mixin, nnUNetTrainer):
    """Plain Dice+CE baseline at the 250-epoch/save_every=10 budget.

    Exists instead of using nnunetv2's stock ``nnUNetTrainer_250epochs``
    specifically to get the reduced save_every -- the stock class doesn't
    have it, and we want every condition in the study checkpointing on the
    same safe schedule.
    """

    pass


class nnUNetTrainerDiceOnly_250epochs(_Epochs250Mixin, nnUNetTrainerDiceOnly):
    pass


class nnUNetTrainerFocal_250epochs(_Epochs250Mixin, nnUNetTrainerFocal):
    pass


class nnUNetTrainerTversky_250epochs(_Epochs250Mixin, nnUNetTrainerTversky):
    pass


class nnUNetTrainerFocalTversky_250epochs(_Epochs250Mixin, nnUNetTrainerFocalTversky):
    pass


class nnUNetTrainerTverskyMild_250epochs(_Epochs250Mixin, nnUNetTrainerTverskyMild):
    pass
