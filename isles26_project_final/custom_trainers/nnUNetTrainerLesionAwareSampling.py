"""Fixed and curriculum lesion-volume-aware case sampling for nnU-Net 2.8.1.

nnU-Net's training loader accepts per-case ``sampling_probabilities``. The
trainers below inject those probabilities before augmentation workers start.
Validation sampling remains unchanged.

Main experiment:
    nnUNetTrainerLesionAwareSampling
        Uses inverse lesion-volume probabilities for the entire training run.

Curriculum experiment:
    nnUNetTrainerCurriculumLesionAwareSampling
        First third: uniform case sampling
        Middle third: probability proportional to volume**-0.5
        Final third: probability proportional to volume**-1.0

At the two curriculum boundaries the augmenters are deliberately rebuilt so the
new probabilities reach multiprocessing workers on both Linux and Windows.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from .sampling_utils import curriculum_power, inverse_volume_probabilities


class _LesionVolumeSamplingBase(nnUNetTrainer):
    """Shared machinery for case-level sampling based on lesion volume."""

    def _load_volume_by_case(self) -> dict[str, float]:
        csv_path = Path(os.environ.get("ISLES26_CASE_METADATA_CSV", "case_metadata.csv"))
        if not csv_path.is_file():
            raise RuntimeError(
                f"Sampling metadata not found: {csv_path}. Run `python isles26.py prepare` first."
            )
        frame = pd.read_csv(csv_path)
        required = {"case_id", "lesion_volume_mm3"}
        missing = required.difference(frame.columns)
        if missing:
            raise RuntimeError(f"Sampling metadata is missing columns: {sorted(missing)}")
        if frame["case_id"].duplicated().any():
            raise RuntimeError("Sampling metadata contains duplicate case_id values")

        volumes = pd.to_numeric(frame["lesion_volume_mm3"], errors="coerce")
        if volumes.isna().any() or not np.isfinite(volumes.to_numpy(dtype=float)).all() or (volumes < 0).any():
            raise RuntimeError("lesion_volume_mm3 values must all be finite and non-negative")
        return dict(zip(frame["case_id"].astype(str), volumes.astype(float)))

    def _sampling_power(self) -> tuple[str, float]:
        raise NotImplementedError

    def _probabilities_for_identifiers(self, identifiers: list[str]) -> np.ndarray:
        volumes_by_case = self._load_volume_by_case()
        missing = [case_id for case_id in identifiers if case_id not in volumes_by_case]
        if missing:
            preview = ", ".join(missing[:5])
            raise RuntimeError(
                f"Metadata has no lesion volume for {len(missing)} training cases: {preview}"
            )
        _, power = self._sampling_power()
        volumes = [volumes_by_case[case_id] for case_id in identifiers]
        return inverse_volume_probabilities(volumes, power=power)

    def get_dataloaders(self):
        trainer_module = importlib.import_module("nnunetv2.training.nnUNetTrainer.nnUNetTrainer")
        original_loader = trainer_module.nnUNetDataLoader
        construction_count = 0
        sampled_case_count = 0

        def weighted_loader(*args, **kwargs):
            nonlocal construction_count, sampled_case_count
            construction_count += 1
            # nnU-Net 2.8.1 constructs the training loader first and validation
            # loader second. Only the training loader receives case probabilities.
            if construction_count == 1:
                dataset = args[0] if args else kwargs["data"]
                identifiers = [str(case_id) for case_id in dataset.identifiers]
                kwargs["sampling_probabilities"] = self._probabilities_for_identifiers(identifiers)
                sampled_case_count = len(identifiers)
            return original_loader(*args, **kwargs)

        trainer_module.nnUNetDataLoader = weighted_loader
        try:
            train_loader, validation_loader = super().get_dataloaders()
        finally:
            trainer_module.nnUNetDataLoader = original_loader

        if construction_count < 2 or sampled_case_count == 0:
            raise RuntimeError("Could not inject lesion-aware probabilities into the nnU-Net training loader")

        phase_name, power = self._sampling_power()
        self._active_sampling_signature = (phase_name, power)
        self.print_to_log_file(
            f"[{self.__class__.__name__}] case sampling active for {sampled_case_count} training cases: "
            f"phase={phase_name}, inverse-volume power={power:.2f}. Validation remains uniform."
        )
        return train_loader, validation_loader

    @staticmethod
    def _finish_augmenter(augmenter) -> None:
        if isinstance(augmenter, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
            old_stdout = sys.stdout
            try:
                with open(os.devnull, "w") as sink:
                    sys.stdout = sink
                    augmenter._finish()
            finally:
                sys.stdout = old_stdout


class nnUNetTrainerLesionAwareSampling(_LesionVolumeSamplingBase):
    """Fixed inverse-volume case sampling for all epochs (experiment 4)."""

    def _sampling_power(self) -> tuple[str, float]:
        return "fixed", 1.0


class nnUNetTrainerCurriculumLesionAwareSampling(_LesionVolumeSamplingBase):
    """Progressively increase small-lesion sampling emphasis (experiment 6)."""

    def _sampling_power(self) -> tuple[str, float]:
        return curriculum_power(self.current_epoch, self.num_epochs)

    def on_train_epoch_start(self):
        desired = self._sampling_power()
        active = getattr(self, "_active_sampling_signature", None)
        if active is not None and desired != active:
            self.print_to_log_file(
                f"[CurriculumLesionAwareSampling] switching sampling phase at epoch {self.current_epoch}: "
                f"{active[0]} -> {desired[0]} (power {desired[1]:.2f})."
            )
            # Worker processes hold their own loader copies, so mutating the parent
            # loader is not sufficient on Windows/Linux. Rebuild at phase changes.
            self._finish_augmenter(self.dataloader_train)
            self._finish_augmenter(self.dataloader_val)
            self.dataloader_train, self.dataloader_val = self.get_dataloaders()
        super().on_train_epoch_start()
