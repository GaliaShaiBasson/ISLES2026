"""Overfit-a-tiny-subset sanity gate: train tips checklist item 1, generic mixin.

**What it does.** A mixin that lets any real trainer class (loss / sampling /
augmentation variant, whatever) run long enough to *start clearly
memorizing* a tiny dataset, instead of the ~5-epoch wiring-only smoke test
``nnUNetTrainerDebugFast`` already provides. Combine with
``Dataset999_ATLASsample`` (the project's existing small sample dataset, ~6
cases/split -- see ``data_prep/split_dataset.py --sample-per-split`` and
CLAUDE.md's "Full pipeline smoke test" entry), not the real dataset -- this
is meant to run in a few minutes, not overnight.

**Non-obvious rationale.** Training-tips checklist item 1 (``General Training
Tips - Practical Checklist.pdf``, "Overfit a Tiny Subset First"): "if it can't
overfit a tiny subset, something is probably wrong" -- bug in labels, wrong
loss, wrong tensor shape, bad normalization, too much augmentation, learning
rate too low/high, model output not aligned with target. PROJECT_REVIEW.md
("Checklist review" / "two runs remain" entries) flags that this project had
no standing version of that check. The signal doesn't need full convergence
to be diagnostic: the 2026-08-17 sampling-collapse bug was visible from
Pseudo Dice across just its first 5-6 epochs (0, 0, 0, 0.004, 0.12, 0.38 --
climbing immediately once fixed, versus 96 epochs pinned at exactly 0.0 when
broken). This mixin is deliberately kept short (a few minutes, not the
15+ minutes a near-full memorization run would take) -- it only needs to show
a clear upward Pseudo-Dice trend / downward loss trend in that window, not
Dice -> ~1.0. Run this against ANY newly-touched trainer (new loss, new
sampling weights, new augmentation pipeline) before launching it for real --
see ``sanity_overfit_check.sh``.

Deliberately does NOT override ``do_split()`` to carve out its own N-case
subset: Dataset999_ATLASsample's existing train split is already small enough
(16 cases/split as of 2026-08-18, inside the deck's suggested 1/8/16/32/64
range) and reusing it means this mixin needs zero new data-prep machinery --
just point any trainer at ``--dataset-id 999 --dataset-name ATLASsample``.

**Calibrated against a real measurement, not guessed.** First cut used 40
epochs x 20 iterations/epoch (800 total). Verified live against two trainers
(stock-augmentation control and WideAug) on this dataset: both showed Pseudo
Dice flat at exactly 0.0 for the full 800 iterations while train_loss dropped
steadily and cleanly (0.3-0.4 -> negative) -- initially read as a possible
wiring bug in WideAug, until the *stock* control showed the identical flat
signature, which ruled that out. Cross-checked against this project's own
real-run history instead (CLAUDE.md, 2026-08-17): a healthy real condition's
Pseudo Dice was ALSO still 0 after 3 real epochs (750 iterations, at nnU-Net's
default 250 iterations/epoch) before climbing to 0.38 by epoch 5 (1250
iterations) -- i.e. 800 iterations is simply short of the real threshold
where this metric starts moving on a 3D full-res patch task, on ANY trainer,
not evidence of anything broken. Each epoch measured at ~2.45s here (this
tiny dataset, not the real one), so headroom was cheap: raised to 100 epochs
x 25 iterations/epoch (2500 total, comfortably past the ~1250-iteration
threshold above) for a rerun at ~5 minutes wall-clock, still well inside "a
few minutes," not a full memorization run.

Not a real experiment result -- same spirit as ``nnUNetTrainerDebugFast``,
just longer (enough epochs to see a real learning trend, not just verify the
pipeline runs at all).
"""
from __future__ import annotations

import torch

# See "Calibrated against a real measurement" above for where these numbers
# come from -- not a guess, and not a full memorization run.
OVERFIT_CHECK_NUM_EPOCHS = 100
OVERFIT_CHECK_ITERS_PER_EPOCH = 25


class _OverfitCheckMixin:
    """Rescales the schedule for the tiny-subset memorization-trend check. See module docstring.

    __init__ must declare nnU-Net's exact named parameters, not *args/**kwargs
    -- see the long comment on ``nnUNetTrainerLossVariants._Epochs250Mixin``
    for why (same bug class, avoided the same way).
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
        self.num_epochs = OVERFIT_CHECK_NUM_EPOCHS
        self.num_iterations_per_epoch = OVERFIT_CHECK_ITERS_PER_EPOCH
        self.num_val_iterations_per_epoch = min(self.num_val_iterations_per_epoch, 5)
        self.save_every = 50
        self.print_to_log_file(
            f"[OVERFIT-CHECK] {self.num_epochs} epochs x {self.num_iterations_per_epoch} "
            "iterations on a tiny sample dataset -- a few minutes, not a full memorization "
            "run. Expect Pseudo Dice trending up and train_loss trending down clearly by the "
            "end; if it's flat/stuck instead, something in this trainer's wiring is broken "
            "(see training-tips checklist item 1). Do not use this checkpoint for reported "
            "results."
        )


# --- Concrete combos: add one per trainer that needs gating before a real launch. ---

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer  # noqa: E402

from .nnUNetTrainerLesionAwareSampling import nnUNetTrainerLesionAwareSamplingPow  # noqa: E402
from .nnUNetTrainerWideAug import nnUNetTrainerWideAug  # noqa: E402


class nnUNetTrainerWideAugBaseline_OverfitCheck(_OverfitCheckMixin, nnUNetTrainerWideAug):
    """Gate for nnUNetTrainerWideAugBaseline_500epochs before tonight's real launch.

    Usage:
      python isles26.py train overfit-check-wideaug --dataset-id 999 --dataset-name ATLASsample
    """

    pass


class nnUNetTrainerBaseline_OverfitCheck(_OverfitCheckMixin, nnUNetTrainer):
    """Stock-augmentation control for the same tiny dataset/budget as the WideAug check.

    Exists to disambiguate a WideAug WARN: widened augmentation (especially newly-enabled
    elastic deformation) legitimately fights memorization by design -- the training-tips
    deck itself lists "too much augmentation" as a reason a model won't overfit a tiny
    subset, so a flat/low Pseudo Dice under widened augmentation isn't automatically a
    wiring bug. Run this stock-augmentation trainer through the identical short budget on
    the identical cases; if it ALSO shows a flat/near-zero Pseudo Dice trend, the flat
    result is explained by the short budget/tiny-n interaction, not by anything specific
    to nnUNetTrainerWideAug's overridden ``get_training_transforms``.
    """

    pass


class nnUNetTrainerLesionAwareSamplingPow_OverfitCheck(_OverfitCheckMixin, nnUNetTrainerLesionAwareSamplingPow):
    """Gate for nnUNetTrainerLesionAwareSamplingPow_500epochs_full before queuing it tonight.

    Tests the actually-new wiring (epoch mixin + dataset002-specific metadata CSV path),
    not the sampling-injection mechanism itself (already proven live in tonight's
    finished sampling-500 run). Needs a pow-weighted metadata CSV for Dataset999's own
    case IDs, which is NOT the dataset002 default this trainer normally reads --
    override at invocation time:

      ISLES26_SAMPLING_POW_METADATA_CSV_FULL=workspace/sample_run/case_metadata_pow_p05.csv \\
        python isles26.py train overfit-check-samplingpow --dataset-id 999 --dataset-name ATLASsample
    """

    pass
