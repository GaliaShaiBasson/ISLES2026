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
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerLesionAwareSampling(nnUNetTrainer):
    def _load_case_weights(self) -> dict[str, float]:
        csv_path = Path(os.environ.get("ISLES26_CASE_METADATA_CSV", "case_metadata.csv"))
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
