"""1000-epoch baseline trainer for dataset002 (corrected full-split).

Mirrors ``nnUNetTrainer500epochs.nnUNetTrainerBaseline_500epochs`` exactly, at
double the epoch budget -- nnU-Net's own stock default epoch count, but still
routed through this project's own trainer (save_every=10, unattended-run
checkpoint-safety convention) rather than nnU-Net's built-in
``nnUNetTrainer`` (which defaults to save_every=50).

Why 1000 now: the 500-epoch/dataset002 baseline result was the deciding data
point the 500-epoch file's docstring said to wait for -- see
nnUNetTrainer500epochs.py's module docstring ("If 500 still shows Dice
climbing, that's the evidence to justify 1000 next time") and the 500-epoch
baseline result in CLAUDE.md.

Baseline only (not every loss/sampling variant re-run at 1000 epochs) --
this is a single additional data point on whether this dataset keeps
improving past 500 epochs, not a full re-run of the study at a new budget.

Targets Dataset002_ATLAS (the corrected 1,453-case split, not the original
1,284-case Dataset001_ATLAS) -- pass --dataset-id 2 when training.
"""
from __future__ import annotations

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class _Epochs1000Mixin:
    """Rescales the poly-LR schedule to a 1000-epoch budget. See module
    docstring for why 1000, and why baseline-only for now.

    save_every=10, same as every other trainer in this project -- unattended-
    run safety margin, unchanged from the 250/500-epoch convention.

    __init__ must declare nnU-Net's exact named parameters, not *args/**kwargs
    -- see the long comment on nnUNetTrainerLossVariants._Epochs250Mixin for
    why (this is the same bug class, avoided the same way).
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
        self.num_epochs = 1000
        self.save_every = 10


class nnUNetTrainerBaseline_1000epochs(_Epochs1000Mixin, nnUNetTrainer):
    """Plain Dice+CE baseline at the 1000-epoch budget, dataset002."""

    pass
