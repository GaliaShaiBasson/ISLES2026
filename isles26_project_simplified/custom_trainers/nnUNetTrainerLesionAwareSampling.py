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
    CASE_METADATA_CSV_DEFAULT = "case_metadata.csv"

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
    (workspace/case_metadata_pow_p05.csv, generated by
    data_prep/generate_pow_sampling_metadata.py -- never overwrites
    workspace/case_metadata.csv).
    """

    CASE_METADATA_CSV_ENV_VAR = "ISLES26_SAMPLING_POW_METADATA_CSV"
    CASE_METADATA_CSV_DEFAULT = "workspace/case_metadata_pow_p05.csv"
