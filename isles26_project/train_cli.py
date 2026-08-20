#!/usr/bin/env python3
"""Train stage: launch one trainer or a predefined experiment group via nnUNetv2_train.

Expected input layout: a preprocessed nnU-Net dataset (see data_prep_cli.py
preprocess) and, for lesion-aware-sampling trainers, the case-metadata CSV
each specific trainer reads (see core.SAMPLING_METADATA_BY_TRAINER).

What it produces: nnU-Net checkpoints under nnUNet_results/, plus an
``isles26_fingerprint.json`` marker per run folder recording exactly what
was trained (see core.compute_run_fingerprint) so a later hyperparameter
change can never silently reuse/overwrite an incompatible checkpoint folder.

Non-obvious rationale: split out of isles26.py (see core.py's docstring) so
this stage is runnable on its own (``python train_cli.py baseline-500
--dataset-id 2``, no "train" subcommand needed -- the experiment name is the
positional argument directly) as well as via the ``isles26.py`` umbrella
(where it's registered under the "train" subcommand name). Both paths share
the exact same argument definitions (``_add_train_arguments``) and
``cmd_train``, so behavior never drifts between them.

Usage:
    python train_cli.py baseline-500 --dataset-id 2
    python train_cli.py sampling-pow-500 --dataset-id 2 --print-only
"""
from __future__ import annotations

import argparse
from pathlib import Path

from core import (
    SAMPLING_METADATA_BY_TRAINER,
    build_environment,
    compute_run_fingerprint,
    config_value,
    ensure_workspace,
    run_cli,
    run_command,
    run_id_from_fingerprint,
    _find_existing_checkpoint,
    _find_latest_checkpoint,
    _guard_run_fingerprint,
    _output_folder,
)
from data_prep_cli import cmd_preprocess

TRAINER_GROUPS = {
    # All real-experiment trainers below share a 250-epoch budget (down from
    # nnU-Net's 1000-epoch default) so the study fits the available GPU time.
    # Every condition uses the same budget so the baseline/loss/sampling
    # comparison stays controlled. See PROJECT_PLAN.md.
    "baseline": ["nnUNetTrainerBaseline_250epochs"],
    "losses": [
        "nnUNetTrainerDiceOnly_250epochs",
        "nnUNetTrainerFocal_250epochs",
        "nnUNetTrainerTversky_250epochs",
        "nnUNetTrainerFocalTversky_250epochs",
    ],
    "dice": ["nnUNetTrainerDiceOnly_250epochs"],
    "focal": ["nnUNetTrainerFocal_250epochs"],
    "tversky": ["nnUNetTrainerTversky_250epochs"],
    "focal-tversky": ["nnUNetTrainerFocalTversky_250epochs"],
    "tversky-mild": ["nnUNetTrainerTverskyMild_250epochs"],
    "sampling": ["nnUNetTrainerLesionAwareSampling_250epochs"],
    "sampling-pow": ["nnUNetTrainerLesionAwareSamplingPow_250epochs"],
    "debug": ["nnUNetTrainerDebugFast"],

    # 500-epoch run against Dataset002_ATLAS (the corrected, 1,453-case full
    # split -- see CLAUDE.md). Requires --dataset-id 2. Sampling uses the
    # original 4:2:1 bin-level ratio, not the power-law variant -- the pow
    # variant's own 250-epoch results (val/test_id/test_ood Dice all lower
    # than plain 4:2:1) argued against it before this run was staged. Includes
    # tversky-mild alongside focal-tversky (in addition to, not instead of --
    # tversky-mild's own 250-epoch/dataset001 run had no results yet when this
    # was assembled). DA5 augmentation was investigated but not verified in
    # time; stock augmentation only for this run.
    "baseline-500": ["nnUNetTrainerBaseline_500epochs"],
    "focal-tversky-500": ["nnUNetTrainerFocalTversky_500epochs"],
    "tversky-mild-500": ["nnUNetTrainerTverskyMild_500epochs"],
    "sampling-500": ["nnUNetTrainerLesionAwareSampling_500epochs_full"],

    # Widened-intensity-augmentation variant, built on the plain baseline for a
    # clean A/B against baseline-500 (see custom_trainers/nnUNetTrainerWideAug.py
    # and CLAUDE.md/PROJECT_PLAN.md/PROJECT_REVIEW.md "two runs remain" entries).
    "baseline-wideaug-500": ["nnUNetTrainerWideAugBaseline_500epochs"],

    # Power-law (p=0.5) bin-level sampling at 500ep/dataset002 -- re-tested on the
    # test_ood angle specifically, not the val-Dice angle that argued against it at
    # 250ep/dataset001 (see nnUNetTrainerLesionAwareSamplingPow_500epochs_full's
    # docstring and PROJECT_PLAN.md). Queued to launch after baseline-wideaug-500.
    "sampling-pow-500": ["nnUNetTrainerLesionAwareSamplingPow_500epochs_full"],

    # Power-law sampling with p annealed 0 -> 1 across training (deferred
    # re-weighting), rather than a single fixed p -- additional condition
    # alongside sampling-500/sampling-pow-500, not a replacement for either.
    # See nnUNetTrainerLesionAwareSamplingPowCurriculum_500epochs_full's docstring.
    "sampling-pow-curriculum-500": ["nnUNetTrainerLesionAwareSamplingPowCurriculum_500epochs_full"],

    # 1000-epoch baseline, dataset002 -- the follow-up the 500-epoch baseline's
    # own docstring deferred to, if 500 still showed Dice climbing (see
    # nnUNetTrainer1000epochs.py). Baseline only, not a full re-run of every
    # condition at the new budget.
    "baseline-1000": ["nnUNetTrainerBaseline_1000epochs"],

    # Pure TopK(k=10) loss at 500ep/dataset002 -- modeled on MAPPING's (Huo et al.
    # 2022, arXiv:2211.15486, 1st place ATLAS'22 challenge) "DTK10" scheme, which
    # replaces the default Dice+CE compound loss with TopK10 specifically to improve
    # small-lesion segmentation -- this project's own universal weak point (see
    # nnUNetTrainerTopk10_500epochs's docstring).
    # nnUNetTrainerTopk10_500epochs (pure TopK, above) collapsed to an empty-mask
    # prediction on a real launch (2026-08-19, see its docstring) -- not relaunched.
    "topk10-500": ["nnUNetTrainerTopk10_500epochs"],
    # Hedged replacement: Dice + TopK10 compound, keeps Dice's anti-collapse anchor.
    "dctopk10-500": ["nnUNetTrainerDCTopk10_500epochs"],

    # Overfit-a-tiny-subset sanity gate (training-tips checklist item 1 / this
    # project's own bug history -- see PROJECT_REVIEW.md). Run against a new
    # trainer on Dataset999_ATLASsample (small sample dataset) BEFORE launching
    # it for real -- see sanity_overfit_check.sh. Not a real experiment result.
    "overfit-check-wideaug": ["nnUNetTrainerWideAugBaseline_OverfitCheck"],
    # Stock-augmentation control for the same check -- see
    # custom_trainers/nnUNetTrainerOverfitCheck.py:nnUNetTrainerBaseline_OverfitCheck.
    "overfit-check-baseline": ["nnUNetTrainerBaseline_OverfitCheck"],
    # Gate for sampling-pow-500 -- needs ISLES26_SAMPLING_POW_METADATA_CSV_FULL
    # overridden to a Dataset999-specific pow CSV, see
    # nnUNetTrainerOverfitCheck.py:nnUNetTrainerLesionAwareSamplingPow_OverfitCheck.
    "overfit-check-samplingpow": ["nnUNetTrainerLesionAwareSamplingPow_OverfitCheck"],
    # Gate for topk10-500 -- see nnUNetTrainerOverfitCheck.py:nnUNetTrainerTopk10_OverfitCheck.
    "overfit-check-topk10": ["nnUNetTrainerTopk10_OverfitCheck"],
    # Gate for dctopk10-500 -- non-negotiable given topk10-500's collapse, see
    # nnUNetTrainerOverfitCheck.py:nnUNetTrainerDCTopk10_OverfitCheck.
    "overfit-check-dctopk10": ["nnUNetTrainerDCTopk10_OverfitCheck"],
}


# TRAINER_GROUPS = {
#     "baseline": ["nnUNetTrainer"],
#     "losses": [
#         "nnUNetTrainerDiceOnly",
#         "nnUNetTrainerFocal",
#         "nnUNetTrainerTversky",
#         "nnUNetTrainerFocalTversky",
#     ],
#     "dice": ["nnUNetTrainerDiceOnly"],
#     "focal": ["nnUNetTrainerFocal"],
#     "tversky": ["nnUNetTrainerTversky"],
#     "focal-tversky": ["nnUNetTrainerFocalTversky"],
#     "sampling": ["nnUNetTrainerLesionAwareSampling"],
#     "debug": ["nnUNetTrainerDebugFast"],
# }


def _train_one(
    trainer: str,
    dataset_id: int,
    dataset_name: str,
    configuration: str,
    fold: str,
    device: str,
    num_gpus: int,
    args: argparse.Namespace,
    env: dict[str, str],
    plans_identifier: str = "nnUNetPlans",
) -> None:
    continue_training = args.continue_training
    if not continue_training and not args.overwrite and not args.validate_only:
        latest = _find_latest_checkpoint(trainer, dataset_id, dataset_name, configuration, fold, env, plans_identifier)
        if latest is not None:
            # Training was interrupted mid-run (crash, kill, machine restart) and left a
            # partial checkpoint_latest.pth. nnU-Net's own default (no --c) would silently
            # start over from epoch 0 and eventually overwrite it -- for unattended
            # multi-hour runs that's a real risk, not a hypothetical, so resume by default
            # instead. --overwrite explicitly opts back into a genuine restart.
            print(f"[resume] Found partial checkpoint, continuing from it instead of restarting: {latest}")
            continue_training = True

    command = ["nnUNetv2_train", str(dataset_id), configuration, fold]
    if trainer != "nnUNetTrainer":
        command += ["-tr", trainer]
    if plans_identifier != "nnUNetPlans":
        command += ["-p", plans_identifier]
    if num_gpus != 1:
        command += ["-num_gpus", str(num_gpus)]
    if device != "cuda":
        command += ["-device", device]
    if continue_training:
        command.append("--c")
    if args.validate_only:
        command.append("--val")
    if args.val_best:
        command.append("--val_best")
    if args.npz:
        command.append("--npz")
    if args.disable_checkpointing:
        command.append("--disable_checkpointing")
    run_command(command, env, args.print_only)


def cmd_train(args: argparse.Namespace) -> int:
    env = build_environment()
    ensure_workspace(env)
    dataset_id = args.dataset_id or config_value(env, "ISLES26_DATASET_ID", int)
    dataset_name = args.dataset_name or config_value(env, "ISLES26_DATASET_NAME")
    fold = str(args.fold if args.fold is not None else config_value(env, "ISLES26_FOLD"))

    if args.experiment == "debug":
        configuration = args.configuration or "2d"
        device = args.device or "cpu"
    else:
        configuration = args.configuration or config_value(env, "ISLES26_CONFIGURATION")
        device = args.device or config_value(env, "ISLES26_DEVICE")
    num_gpus = args.num_gpus or config_value(env, "ISLES26_NUM_GPUS", int)
    plans_identifier = args.plans or "nnUNetPlans"

    if args.preprocess:
        preprocess_args = argparse.Namespace(
            dataset_id=dataset_id,
            no_verify=False,
            print_only=args.print_only,
            overwrite=False,
        )
        cmd_preprocess(preprocess_args)

    trainers = [args.trainer] if args.trainer else TRAINER_GROUPS[args.experiment]
    # Each lesion-aware-sampling variant reads its own metadata CSV (distinct env var
    # + default path per trainer -- see custom_trainers/nnUNetTrainerLesionAwareSampling.py)
    # so concurrent variants never share or collide over one file. Check whichever file
    # the specific trainer(s) being launched will actually read, not always the original.
    for trainer_name in trainers:
        if trainer_name not in SAMPLING_METADATA_BY_TRAINER:
            continue
        env_var, default_path = SAMPLING_METADATA_BY_TRAINER[trainer_name]
        metadata = Path(env.get(env_var, default_path))
        if not metadata.is_file() and not args.print_only:
            raise SystemExit(
                f"Sampling metadata is missing for {trainer_name}: {metadata} "
                f"(set via {env_var}). Generate it first."
            )

    is_group_run = args.trainer is None and len(trainers) > 1
    skip_guard_active = (
        not args.continue_training and not args.validate_only and not args.overwrite and not args.print_only
    )

    for trainer in trainers:
        print(f"\n== {trainer} ==" + (f" (plans={plans_identifier})" if plans_identifier != "nnUNetPlans" else ""))
        output_folder = _output_folder(trainer, dataset_id, dataset_name, configuration, fold, env, plans_identifier)
        if not args.print_only:
            # Runs before the checkpoint skip-guard below: a hyperparameter change that
            # nnU-Net's own folder naming can't see (num_epochs, sampling weights
            # content, split fold count, ...) must be caught even when no
            # checkpoint_final.pth exists yet (e.g. a stale partial run from a since-
            # changed config). See compute_run_fingerprint/_guard_run_fingerprint.
            fingerprint = compute_run_fingerprint(
                trainer, dataset_id, dataset_name, configuration, fold, env, plans_identifier
            )
            _guard_run_fingerprint(output_folder, fingerprint, args.overwrite)
            print(f"[run_id] {run_id_from_fingerprint(fingerprint)}")
        if skip_guard_active:
            existing = _find_existing_checkpoint(
                trainer, dataset_id, dataset_name, configuration, fold, env, plans_identifier
            )
            if existing is not None:
                message = (
                    f"Training already completed: {existing}. Pass --overwrite to restart from "
                    "scratch, or --continue to resume/verify."
                )
                if is_group_run:
                    print(f"[skip] {message}")
                    continue
                raise SystemExit(message)
        _train_one(trainer, dataset_id, dataset_name, configuration, fold, device, num_gpus, args, env, plans_identifier)
    return 0


def _add_train_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("experiment", choices=sorted(TRAINER_GROUPS))
    p.add_argument("--trainer", help="Override the predefined trainer class")
    p.add_argument("--dataset-id", type=int)
    p.add_argument("--dataset-name", help="Override the dataset name (default: ISLES26_DATASET_NAME in .env)")
    p.add_argument("--fold")
    p.add_argument("--configuration")
    p.add_argument("--device", choices=["cuda", "cpu", "mps"])
    p.add_argument("--num-gpus", type=int)
    p.add_argument(
        "--plans",
        default=None,
        help=(
            "nnU-Net plans identifier, e.g. nnUNetResEncUNetMPlans (must already exist under "
            "nnUNet_preprocessed/<dataset>/, via nnUNetv2_plan_experiment -pl ...). Default: "
            "nnUNetPlans (the plain-U-Net default planner). A non-default value gets its own "
            "output folder and run_id -- see compute_run_fingerprint -- so it never collides "
            "with a same-trainer default-plans run."
        ),
    )
    p.add_argument("--preprocess", action="store_true", help="Run preprocessing before training")
    p.add_argument("--continue", dest="continue_training", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--val-best", action="store_true")
    p.add_argument("--npz", action="store_true")
    p.add_argument("--disable-checkpointing", action="store_true")
    p.add_argument("--print-only", action="store_true")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Restart training from scratch even if a completed checkpoint_final.pth already exists",
    )


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train", help="Train one experiment or a predefined experiment group")
    _add_train_arguments(p)
    p.set_defaults(func=cmd_train)


def build_standalone_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train stage: launch one trainer or an experiment group")
    _add_train_arguments(parser)
    parser.set_defaults(func=cmd_train)
    return parser


if __name__ == "__main__":
    raise SystemExit(run_cli(build_standalone_parser()))
