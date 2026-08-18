"""
Trainer variant for the lesion-aware sampling study (Phase 3).

Design choice: rather than reconstructing nnU-Net's dataloader pipeline from
scratch (patch size, transforms, oversample_foreground_percent, etc. --
version-specific internals I can't verify without a live install), this
takes the dataloaders nnU-Net *already builds correctly* via the parent
class and only intercepts case selection: which training case a given
sample in the batch comes from. Standard foreground-patch-center
oversampling (nnU-Net's `oversample_foreground_percent`) still applies
per-case as usual -- this adds a second, complementary layer: cases with
smaller lesions are simply picked more often in the first place.

Weights come from `case_metadata.csv` (built by
`data_prep/prepare_isles26_dataset.py`), specifically the
`sampling_weight` column (inverse lesion volume, pre-normalized).

Usage:
    nnUNetv2_train <DATASET_ID> 3d_fullres 0 -tr nnUNetTrainerLesionAwareSampling

Set the metadata CSV path via environment variable before training:
    export ISLES26_CASE_METADATA_CSV=/path/to/case_metadata.csv

VERIFY BEFORE TRUSTING: this monkeypatches `get_indices` on the training
dataloader instance, assuming it exposes `self.indices` (list of case keys)
and `self.batch_size` (standard in nnunetv2's `nnUNetDataLoaderBase` as of
2.4/2.5). Print `dl_tr.indices[:3]` and `dl_tr.batch_size` after building the
dataloaders once to confirm these exist in your installed version before a
long training run.
"""
import os
import types

import numpy as np
import pandas as pd

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _weighted_get_indices(self):
    """Replacement for the dataloader's default (uniform) case sampling."""
    return list(np.random.choice(self._case_weight_keys, self.batch_size,
                                  replace=True, p=self._case_weight_probs))


def _inject_lesion_aware_sampling(dataloader, weights_by_case: dict):
    """Attach normalized sampling weights to `dataloader` and swap in
    `_weighted_get_indices` for its `get_indices` method, aligned to the
    dataloader's own case key list so we never sample a key it doesn't have."""
    keys = list(dataloader.indices)
    default_weight = np.median(list(weights_by_case.values())) if weights_by_case else 1.0
    raw = np.array([weights_by_case.get(k, default_weight) for k in keys], dtype=np.float64)
    probs = raw / raw.sum()

    dataloader._case_weight_keys = keys
    dataloader._case_weight_probs = probs
    dataloader.get_indices = types.MethodType(_weighted_get_indices, dataloader)


class nnUNetTrainerLesionAwareSampling(nnUNetTrainer):

    def _load_case_weights(self) -> dict:
        csv_path = os.environ.get("ISLES26_CASE_METADATA_CSV", "case_metadata.csv")
        if not os.path.isfile(csv_path):
            self.print_to_log_file(
                f"[nnUNetTrainerLesionAwareSampling] WARNING: metadata CSV not found at "
                f"'{csv_path}' (set ISLES26_CASE_METADATA_CSV). Falling back to uniform "
                f"sampling -- this run will NOT test the lesion-aware hypothesis."
            )
            return {}
        df = pd.read_csv(csv_path)
        if "sampling_weight" not in df.columns or "case_id" not in df.columns:
            self.print_to_log_file(
                "[nnUNetTrainerLesionAwareSampling] WARNING: expected columns "
                "'case_id'/'sampling_weight' not found in metadata CSV. Falling back to uniform sampling."
            )
            return {}
        return dict(zip(df["case_id"], df["sampling_weight"]))

    def get_dataloaders(self):
        dl_tr, dl_val = super().get_dataloaders()
        weights = self._load_case_weights()
        if weights:
            _inject_lesion_aware_sampling(dl_tr, weights)
            self.print_to_log_file(
                f"[nnUNetTrainerLesionAwareSampling] lesion-aware sampling active "
                f"for {len(weights)} cases (validation loader left uniform, as usual)."
            )
        return dl_tr, dl_val
