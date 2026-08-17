"""500-epoch trainer variants for the dataset002 (corrected full-split) overnight run.

Mirrors ``nnUNetTrainerLossVariants._Epochs250Mixin`` exactly, just at double the
epoch budget. Kept in its own file rather than editing the existing 250-epoch
files -- those stay untouched (still the record of the first, dataset001 run);
this file only *imports* their base (loss-only, non-epoch-mixed) classes and
combines them with a new mixin, the same pattern the 250-epoch file itself uses
for the sampling trainer.

Why 500 and not the still-open 1000 question: covered in conversation (not yet
in CLAUDE.md as of this file's creation) -- 500 gets a first real data point on
"does more than 250 epochs help on this dataset" at half the unattended-run
exposure of 1000, given how few overnight runs remain for this project. If 500
still shows Dice climbing, that's the evidence to justify 1000 next time.

DA5 augmentation was investigated (real, tested-by-nnU-Net's-own-maintainers
augmentation preset -- see CLAUDE.md future considerations) but deliberately
NOT included here: verification (composing it as a mixin, confirming it runs,
checking epoch-time cost) never finished before this run needed to be staged.
Left for a future run rather than shipped unverified.

Targets Dataset002_ATLAS (the corrected 1,453-case split, not the original
1,284-case Dataset001_ATLAS) -- pass --dataset-id 2 when training.
"""
from __future__ import annotations

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from .nnUNetTrainerLossVariants import nnUNetTrainerFocalTversky, nnUNetTrainerTverskyMild
from .nnUNetTrainerLesionAwareSampling import nnUNetTrainerLesionAwareSampling


class _Epochs500Mixin:
    """Rescales the poly-LR schedule to a 500-epoch budget. See module docstring
    for why 500 (not 250, not 1000) for this run.

    save_every=10, same as every other trainer in this project -- unattended-run
    safety margin, unchanged from the 250-epoch convention.

    __init__ must declare nnU-Net's exact named parameters, not *args/**kwargs --
    see the long comment on nnUNetTrainerLossVariants._Epochs250Mixin for why
    (this is the same bug class, avoided the same way).
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
        self.num_epochs = 500
        self.save_every = 10


class nnUNetTrainerBaseline_500epochs(_Epochs500Mixin, nnUNetTrainer):
    """Plain Dice+CE baseline at the 500-epoch budget, dataset002."""

    pass


class nnUNetTrainerFocalTversky_500epochs(_Epochs500Mixin, nnUNetTrainerFocalTversky):
    """Focal-Tversky loss at the 500-epoch budget, dataset002."""

    pass


class nnUNetTrainerTverskyMild_500epochs(_Epochs500Mixin, nnUNetTrainerTverskyMild):
    """Dice + softened-asymmetry Tversky (0.4/0.6) at the 500-epoch budget, dataset002.

    Added alongside baseline/focal-tversky/sampling, not instead of focal-tversky --
    tversky-mild's own 250-epoch/dataset001 run (nnUNetTrainerTverskyMild_250epochs,
    launched separately, already in progress on the GPU when this was staged) had no
    results yet at the time this run was assembled, so it couldn't be substituted in
    with any real evidence behind it (see CLAUDE.md) -- this is the fourth condition,
    run in addition, not a replacement.
    """

    pass


class nnUNetTrainerLesionAwareSampling_500epochs_full(_Epochs500Mixin, nnUNetTrainerLesionAwareSampling):
    """Lesion-aware (4:2:1 bin-level) sampling at the 500-epoch budget, dataset002.

    Named with a "_full" suffix (rather than reusing
    nnUNetTrainerLesionAwareSampling_500epochs, which would otherwise be the
    natural name) to keep it visually distinct from the existing 250-epoch
    dataset001 sampling trainer -- a naming collision risk given both differ
    only by epoch count and dataset otherwise. Own dedicated metadata CSV env
    var + default (workspace/case_metadata_full.csv, matching this session's
    naming agreement for the dataset002 metadata), same isolation pattern
    the Pow variant already established -- never reads the dataset001
    case_metadata.csv, regardless of what ISLES26_CASE_METADATA_CSV is set to.
    """

    CASE_METADATA_CSV_ENV_VAR = "ISLES26_CASE_METADATA_CSV_FULL"
    CASE_METADATA_CSV_DEFAULT = "workspace/case_metadata_full.csv"
