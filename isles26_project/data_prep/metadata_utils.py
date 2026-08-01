"""
Shared utilities for computing lesion size statistics and joining them with
any provided clinical metadata (center, chronicity, days-post-stroke, ...).

This module is imported by both `prepare_isles26_dataset.py` (to build
`case_metadata.csv` once) and `evaluation/compute_metrics.py` (to stratify
results by lesion size / metadata later). Keeping this logic in one place
guarantees the size bins used for sampling and for evaluation are identical.
"""
from __future__ import annotations

import numpy as np
import nibabel as nib
import pandas as pd


def lesion_volume_mm3(mask_path: str) -> float:
    """Compute lesion volume in mm^3 from a binary/multi-label mask file.

    Any label > 0 is treated as lesion. If your dataset uses a specific
    background/ignore label other than 0, adjust `lesion_label` below.
    """
    img = nib.load(mask_path)
    data = img.get_fdata()
    voxel_vol = float(np.prod(img.header.get_zooms()[:3]))
    n_lesion_voxels = int(np.sum(data > 0))
    return n_lesion_voxels * voxel_vol


def assign_size_bin(volumes_mm3: pd.Series, n_bins: int = 3) -> pd.Series:
    """Assign each case to a small/medium/large lesion bin via tertiles.

    Using data-driven tertiles (rather than fixed mm^3 cutoffs) keeps the
    bins balanced regardless of the exact dataset you end up with, which
    matters for readable stratified plots and for the sampling weights.
    """
    labels = ["small", "medium", "large"][:n_bins]
    try:
        return pd.qcut(volumes_mm3, q=n_bins, labels=labels, duplicates="drop")
    except ValueError:
        # Not enough distinct values for clean quantile cuts (e.g. tiny
        # pilot dataset) -- fall back to equal-width bins.
        return pd.cut(volumes_mm3, bins=n_bins, labels=labels[: len(set(volumes_mm3))])


def sampling_weights_from_volume(volumes_mm3: pd.Series, floor: float = 1.0) -> pd.Series:
    """Inverse-volume sampling weights for lesion-aware oversampling.

    Cases with small lesions get higher weight. `floor` avoids division
    blowing up for near-zero-volume edge cases; weights are normalized to
    sum to 1 downstream (see nnUNetTrainerLesionAwareSampling).
    """
    inv = 1.0 / np.maximum(volumes_mm3.values, floor)
    return pd.Series(inv / inv.sum(), index=volumes_mm3.index)


def build_case_metadata(
    case_ids: list[str],
    label_paths: dict[str, str],
    clinical_csv: str | None = None,
    clinical_id_col: str = "subject_id",
) -> pd.DataFrame:
    """Build the master case_metadata.csv used across the whole project.

    Parameters
    ----------
    case_ids: list of nnU-Net case identifiers (e.g. "ISLES26_001")
    label_paths: dict mapping case_id -> path to its ground-truth label file
    clinical_csv: optional path to a CSV with clinical/site metadata to join
        (columns like center, chronicity, days_post_stroke). Joined on
        `clinical_id_col`, which must match your raw subject IDs -- adjust
        the mapping in `prepare_isles26_dataset.py` if your case IDs differ
        from the raw subject IDs.
    """
    rows = []
    for cid in case_ids:
        vol = lesion_volume_mm3(label_paths[cid])
        rows.append({"case_id": cid, "lesion_volume_mm3": vol})
    df = pd.DataFrame(rows)
    df["size_bin"] = assign_size_bin(df["lesion_volume_mm3"])
    df["sampling_weight"] = sampling_weights_from_volume(df["lesion_volume_mm3"])

    if clinical_csv is not None:
        clin = pd.read_csv(clinical_csv)
        # Best-effort join: assumes raw subject id is embedded in case_id.
        # Adjust this join key if your ID scheme differs.
        df["_raw_id"] = df["case_id"].str.extract(r"(\d+)$")[0]
        clin["_raw_id"] = clin[clinical_id_col].astype(str).str.extract(r"(\d+)$")[0]
        df = df.merge(clin.drop(columns=[clinical_id_col]), on="_raw_id", how="left")
        df = df.drop(columns=["_raw_id"])

    return df
