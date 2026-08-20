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
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerTopkLoss import nnUNetTrainerTopk10Loss

from .nnUNetTrainerLossVariants import nnUNetTrainerDCTopk10, nnUNetTrainerFocalTversky, nnUNetTrainerTverskyMild
from .nnUNetTrainerLesionAwareSampling import (
    nnUNetTrainerLesionAwareSampling,
    nnUNetTrainerLesionAwareSamplingPow,
    nnUNetTrainerLesionAwareSamplingPowCurriculum,
)


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
    var + default (workspace/case_metadata/case_metadata_full.csv, matching this session's
    naming agreement for the dataset002 metadata), same isolation pattern
    the Pow variant already established -- never reads the dataset001
    case_metadata.csv, regardless of what ISLES26_CASE_METADATA_CSV is set to.
    """

    CASE_METADATA_CSV_ENV_VAR = "ISLES26_CASE_METADATA_CSV_FULL"
    CASE_METADATA_CSV_DEFAULT = "workspace/case_metadata/case_metadata_full.csv"


class nnUNetTrainerLesionAwareSamplingPowCurriculum_500epochs_full(
    _Epochs500Mixin, nnUNetTrainerLesionAwareSamplingPowCurriculum
):
    """Power-law sampling with p annealed 0 -> 1 across training (in 20 steps of 25
    epochs each -- see nnUNetTrainerLesionAwareSamplingPowCurriculum's docstring for
    why it's stepped, not continuous), at the 500-epoch budget, dataset002.

    Additional condition alongside the fixed-p variants below, not a replacement for
    either -- keeps nnUNetTrainerLesionAwareSamplingPow_500epochs_full (fixed p=0.5)
    and nnUNetTrainerLesionAwareSampling_500epochs_full (fixed 4:2:1) as they are.
    Tests whether deferring the strong small-lesion correction to later in training
    (rather than applying it uniformly from epoch 0, as every other sampling variant
    in this project does) captures more of its benefit without the medium/large-bin
    cost seen in the fixed-p=0.5/250-epoch result (see CLAUDE.md).
    """

    pass


class nnUNetTrainerTopk10_500epochs(_Epochs500Mixin, nnUNetTrainerTopk10Loss):
    """Pure TopK(k=10) loss (nnU-Net's own stock nnUNetTrainerTopk10Loss, no Dice/CE)
    at the 500-epoch budget, dataset002.

    Modeled on the "DTK10" training scheme from MAPPING (Huo et al. 2022,
    arXiv:2211.15486) -- 1st place, 2022 MICCAI ATLAS Challenge. Their paper
    describes DTK10 as replacing the default Dice+CE compound loss with TopK10
    loss specifically because it "further improves the segmentation performance
    on small lesions in particular" -- concentrating gradient weight on the
    hardest 10% of voxels. Directly targets this project's own universal
    weak point (small-lesion Dice, worst on test_ood in every 500-epoch
    condition run so far -- see CLAUDE.md).

    Uses nnU-Net's own stock TopKLoss implementation unmodified (just adds the
    project's epoch-budget mixin) rather than reimplementing it -- MAPPING's own
    "we implement all models based on the nnU-Net framework" wording suggests
    they used this exact built-in trainer for their DTK10 scheme, not a custom
    loss class.

    **DO NOT RELAUNCH AS-IS (2026-08-19):** confirmed collapsed to predicting an
    entirely empty mask (0 nonzero voxels, checked directly on 2 real validation
    cases from a real launch's checkpoint_latest.pth at epoch ~26/500 --
    Pseudo Dice pinned at exactly 0.0 the whole time, train_loss plateaued
    rather than improving). Pure TopK/CE-family losses have no term punishing an
    empty prediction, unlike Dice -- see nnUNetTrainerDCTopk10_500epochs (Dice +
    TopK10 compound) for the hedged replacement.
    """

    pass


class nnUNetTrainerDCTopk10_500epochs(_Epochs500Mixin, nnUNetTrainerDCTopk10):
    """Dice + TopK(k=10) compound loss at the 500-epoch budget, dataset002.

    Hedged replacement for nnUNetTrainerTopk10_500epochs (pure TopK, confirmed
    collapsed to an empty-mask solution -- see that class's docstring). Same
    MAPPING-inspired hard-voxel-focus motivation, with Dice's overlap term as the
    anti-collapse anchor pure TopK was missing. See
    nnUNetTrainerLossVariants.nnUNetTrainerDCTopk10 for the loss construction.
    """

    pass


class nnUNetTrainerLesionAwareSamplingPow_500epochs_full(_Epochs500Mixin, nnUNetTrainerLesionAwareSamplingPow):
    """Power-law (p=0.5) bin-level sampling at the 500-epoch budget, dataset002.

    Re-tested here specifically on the test_ood angle, not the val-Dice angle that
    argued against the pow variant at 250 epochs/dataset001 (see CLAUDE.md): plain
    4:2:1 sampling's own 500-epoch/dataset002 result had the *worst* val Dice of the
    4 conditions run so far but the *best* test_ood Dice (0.6769, edging out baseline's
    0.6764) -- an odd but real split on exactly the metric this project is about.
    p=0.5 is a stronger size-bin correction than 4:2:1 (grounded in the real per-bin
    foreground-volume imbalance -- see metadata_utils.sampling_weights_from_size_bin_power
    and the 2026-08-17 CLAUDE.md entry); this asks whether a stronger correction pushes
    test_ood further, or whether 4:2:1's OOD result was noise.

    Metadata CSV isolation: reads workspace/case_metadata/case_metadata_pow_p05_full.csv (generated
    from case_metadata_full.csv via data_prep/generate_pow_sampling_metadata.py --p 0.5),
    never the plain 4:2:1 file above and never the 250-epoch/dataset001 pow file --
    three separate sampling-weight CSVs, three separate trainers, no shared state.
    """

    pass
