"""Case-level inverse-volume sampling for nnU-Net 2.8.1.

The original implementation patched the augmenter returned by
``super().get_dataloaders()``. At that point worker processes had already been
started and the object did not expose the case indices. This implementation
injects ``sampling_probabilities`` while the underlying nnUNetDataLoader is
being constructed, before augmentation workers start.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerLesionAwareSampling(nnUNetTrainer):
    # Overridden by subclasses (e.g. the power-law variant below) that need a
    # dedicated metadata CSV -- keeps concurrent sampling experiments from ever
    # reading/writing the same file, on top of already having distinct trainer
    # class names (-> distinct nnU-Net output folders, since nnU-Net namespaces
    # results by trainer name). See CLAUDE.md.
    CASE_METADATA_CSV_ENV_VAR = "ISLES26_CASE_METADATA_CSV"
    CASE_METADATA_CSV_DEFAULT = "workspace/case_metadata/case_metadata.csv"

    def _load_case_weights(self) -> dict[str, float]:
        csv_path = Path(os.environ.get(self.CASE_METADATA_CSV_ENV_VAR, self.CASE_METADATA_CSV_DEFAULT))
        if not csv_path.is_file():
            raise RuntimeError(
                f"Sampling metadata not found: {csv_path}. Run the project prepare command first."
            )
        frame = pd.read_csv(csv_path)
        required = {"case_id", "sampling_weight"}
        missing = required.difference(frame.columns)
        if missing:
            raise RuntimeError(f"Sampling metadata is missing columns: {sorted(missing)}")
        if frame["case_id"].duplicated().any():
            raise RuntimeError("Sampling metadata contains duplicate case_id values")
        weights = pd.to_numeric(frame["sampling_weight"], errors="coerce")
        if weights.isna().any() or not np.isfinite(weights.to_numpy(dtype=float)).all() or (weights <= 0).any():
            raise RuntimeError("sampling_weight values must all be finite and positive")
        return dict(zip(frame["case_id"].astype(str), weights.astype(float)))

    def get_dataloaders(self):
        weights_by_case = self._load_case_weights()
        trainer_module = importlib.import_module("nnunetv2.training.nnUNetTrainer.nnUNetTrainer")
        original_loader = trainer_module.nnUNetDataLoader
        construction_count = 0
        sampled_case_count = 0

        def weighted_loader(*args, **kwargs):
            nonlocal construction_count, sampled_case_count
            construction_count += 1
            # nnU-Net constructs the training loader first and validation loader second.
            if construction_count == 1:
                dataset = args[0] if args else kwargs["data"]
                identifiers = [str(case_id) for case_id in dataset.identifiers]
                missing = [case_id for case_id in identifiers if case_id not in weights_by_case]
                if missing:
                    preview = ", ".join(missing[:5])
                    raise RuntimeError(
                        f"Metadata has no sampling weights for {len(missing)} training cases: {preview}"
                    )
                raw = np.asarray([weights_by_case[case_id] for case_id in identifiers], dtype=np.float64)
                probabilities = raw / raw.sum()
                kwargs["sampling_probabilities"] = probabilities
                sampled_case_count = len(identifiers)
            return original_loader(*args, **kwargs)

        trainer_module.nnUNetDataLoader = weighted_loader
        try:
            train_loader, validation_loader = super().get_dataloaders()
        finally:
            trainer_module.nnUNetDataLoader = original_loader

        if construction_count < 2 or sampled_case_count == 0:
            raise RuntimeError("Could not inject lesion-aware probabilities into the nnU-Net training loader")
        self.print_to_log_file(
            f"[LesionAwareSampling] active for {sampled_case_count} training cases; validation remains uniform."
        )
        return train_loader, validation_loader


class nnUNetTrainerLesionAwareSampling_250epochs(nnUNetTrainerLesionAwareSampling):
    """Same lesion-aware sampling, rescaled to the project-wide 250-epoch budget.

    save_every=10 (down from nnU-Net's default 50) for the same unattended-run
    safety reason as the loss-variant trainers -- see
    nnUNetTrainerLossVariants._Epochs250Mixin.

    __init__ must declare nnU-Net's exact named parameters, not *args/**kwargs
    -- see the long comment on _Epochs250Mixin for why (this is the same bug,
    fixed the same way).
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


class nnUNetTrainerLesionAwareSamplingPow_250epochs(nnUNetTrainerLesionAwareSampling_250epochs):
    """Power-law (p=0.5) bin-level sampling weights, grounded in real per-bin volume mass,

    instead of the fixed 4:2:1 ratio nnUNetTrainerLesionAwareSampling_250epochs uses.
    4:2:1 was an arbitrary geometric default; p=0.5 is derived from this project's
    actual small/medium/large lesion-volume imbalance (see
    metadata_utils.sampling_weights_from_size_bin_power and CLAUDE.md decisions log).

    Deliberately isolated from the original sampling trainer in every dimension that
    could cause a collision if both are run: distinct trainer class name (nnU-Net
    namespaces output under nnUNetTrainerLesionAwareSamplingPow_250epochs__..., never
    touching the original's folder) and a distinct metadata CSV
    (workspace/case_metadata/case_metadata_pow_p05.csv, generated by
    data_prep/generate_pow_sampling_metadata.py -- never overwrites
    workspace/case_metadata/case_metadata.csv).
    """

    CASE_METADATA_CSV_ENV_VAR = "ISLES26_SAMPLING_POW_METADATA_CSV"
    CASE_METADATA_CSV_DEFAULT = "workspace/case_metadata/case_metadata_pow_p05.csv"


class nnUNetTrainerLesionAwareSamplingPowCurriculum(nnUNetTrainerLesionAwareSampling):
    """Power-law bin-level sampling with p annealed 0 -> 1 over training, instead of a
    single fixed p (contrast with nnUNetTrainerLesionAwareSamplingPow's fixed p=0.5).

    Rationale: early training on the natural (uniform) case distribution lets the
    network learn generic features before being pushed toward the rare small-lesion
    bin; p then ramps toward the full per-bin volume-mass correction later in training.
    This is the same idea as "deferred re-weighting" in the long-tailed-recognition
    literature (e.g. Cao et al. 2019 LDAM-DRW) -- reweight late, not from epoch 0.

    Engineering note this class exists to work around: nnU-Net's training dataloader
    runs inside worker processes spawned by NonDetMultiThreadedAugmenter (see
    nnUNetTrainer.get_dataloaders), so mutating ``sampling_probabilities`` on the
    in-process loader object would never reach the workers. Instead, p is recomputed
    and the whole train+val dataloader pair is torn down and rebuilt (fresh worker
    processes) every DATALOADER_REBUILD_EVERY_EPOCHS epochs, via on_train_epoch_start.
    This makes p a step function (one value per REBUILD window), not truly continuous
    -- 20 steps across a 500-epoch budget at the default cadence, which is a close
    enough approximation given the alternative (single-process, num_processes=0
    dataloader for genuinely continuous updates) would serialize data loading and
    likely slow training meaningfully.

    Reads case_id/size_bin/lesion_volume_mm3 directly (not a precomputed
    sampling_weight column -- unlike the base class/fixed-p subclass, the weight
    formula must be re-evaluated at a new p on every rebuild) from the same
    dataset002 "_full" metadata CSV nnUNetTrainerLesionAwareSampling_500epochs_full
    reads (workspace/case_metadata/case_metadata_full.csv) -- shared read-only input data, own
    dedicated env var so an override never leaks between trainers.

    Duplicates (deliberately, rather than importing) the power-law weight formula
    from data_prep/metadata_utils.py:sampling_weights_from_size_bin_power -- see the
    module docstring for why this file doesn't cross-import data_prep.
    """

    CASE_METADATA_CSV_ENV_VAR = "ISLES26_SAMPLING_POW_CURRICULUM_METADATA_CSV_FULL"
    CASE_METADATA_CSV_DEFAULT = "workspace/case_metadata/case_metadata_full.csv"

    P_START = 0.0
    P_END = 1.0
    DATALOADER_REBUILD_EVERY_EPOCHS = 25

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self._last_dataloader_rebuild_epoch: int | None = None

    def _current_target_p(self) -> float:
        rebuild_every = self.DATALOADER_REBUILD_EVERY_EPOCHS
        steps_total = max(1, self.num_epochs // rebuild_every)
        current_step = min(steps_total - 1, self.current_epoch // rebuild_every)
        frac = current_step / max(1, steps_total - 1)
        return self.P_START + frac * (self.P_END - self.P_START)

    def _load_case_weights(self) -> dict[str, float]:
        csv_path = Path(os.environ.get(self.CASE_METADATA_CSV_ENV_VAR, self.CASE_METADATA_CSV_DEFAULT))
        if not csv_path.is_file():
            raise RuntimeError(
                f"Sampling metadata not found: {csv_path}. Run the project prepare command first."
            )
        frame = pd.read_csv(csv_path)
        required = {"case_id", "size_bin", "lesion_volume_mm3"}
        missing = required.difference(frame.columns)
        if missing:
            raise RuntimeError(f"Sampling metadata is missing columns: {sorted(missing)}")
        if frame["case_id"].duplicated().any():
            raise RuntimeError("Sampling metadata contains duplicate case_id values")

        p = self._current_target_p()
        bins = frame["size_bin"].astype(str)
        volumes = pd.to_numeric(frame["lesion_volume_mm3"], errors="coerce")
        if volumes.isna().any() or not np.isfinite(volumes.to_numpy(dtype=float)).all():
            raise RuntimeError("lesion_volume_mm3 values must all be finite")
        bin_totals = volumes.groupby(bins).sum()
        bad_bins = bin_totals[(bin_totals <= 0) | bin_totals.isna()].index.tolist()
        if bad_bins:
            raise RuntimeError(f"Non-positive or missing total volume for size_bin(s): {bad_bins}")
        bin_weight = (1.0 / bin_totals) ** float(p)
        per_case = bins.map(bin_weight)
        total = float(per_case.sum())
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError("Could not derive finite sampling weights from size-bin power weighting")
        weights = (per_case / total).astype(float)

        self._last_dataloader_rebuild_epoch = self.current_epoch
        self.print_to_log_file(
            f"[LesionAwareSamplingPowCurriculum] epoch {self.current_epoch}: p={p:.4f}"
        )
        return dict(zip(frame["case_id"].astype(str), weights))

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        rebuild_every = self.DATALOADER_REBUILD_EVERY_EPOCHS
        due = (
            self.current_epoch % rebuild_every == 0
            and self.current_epoch != self._last_dataloader_rebuild_epoch
        )
        if due:
            self._rebuild_dataloaders()

    def _rebuild_dataloaders(self) -> None:
        from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
        from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
            NonDetMultiThreadedAugmenter,
        )

        for loader in (self.dataloader_train, self.dataloader_val):
            if isinstance(loader, (MultiThreadedAugmenter, NonDetMultiThreadedAugmenter)):
                loader._finish()
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()


class nnUNetTrainerLesionAwareSamplingPow(nnUNetTrainerLesionAwareSampling):
    """Power-law (p=0.5) bin-level sampling weights, with NO epoch-budget mixin baked in --
    combine with whichever epoch mixin the target run needs (mirrors how
    nnUNetTrainerLesionAwareSampling itself has no epoch mixin, and
    nnUNetTrainerLesionAwareSampling_500epochs_full combines it with
    _Epochs500Mixin in nnUNetTrainer500epochs.py).

    Exists as a separate class from nnUNetTrainerLesionAwareSamplingPow_250epochs
    (which bakes in the 250-epoch mixin directly) rather than reusing it, so a
    500-epoch/dataset002 variant doesn't have to fight an already-baked-in 250-epoch
    __init__. Own dedicated env var/default (case_metadata_pow_p05_full.csv, matching
    dataset002's "_full" metadata-file naming convention -- see
    nnUNetTrainerLesionAwareSampling_500epochs_full) so it can never collide with the
    250-epoch/dataset001 pow file above, even though both are the same p=0.5 formula.
    """

    CASE_METADATA_CSV_ENV_VAR = "ISLES26_SAMPLING_POW_METADATA_CSV_FULL"
    CASE_METADATA_CSV_DEFAULT = "workspace/case_metadata/case_metadata_pow_p05_full.csv"
