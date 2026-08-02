"""Utilities shared by data preparation, sampling, and evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SIZE_LABELS = ("small", "medium", "large")


def lesion_volume_mm3(mask_path: str | Path) -> float:
    """Compute foreground volume in mm³ from a NIfTI label map."""
    try:
        import nibabel as nib
    except ImportError as exc:  # keeps lightweight commands such as --help usable
        raise RuntimeError("nibabel is required for reading NIfTI files") from exc

    img = nib.load(str(mask_path))
    data = np.asanyarray(img.dataobj)
    voxel_volume = float(np.prod(img.header.get_zooms()[:3]))
    return int(np.count_nonzero(data > 0)) * voxel_volume


def assign_size_bin(volumes_mm3: pd.Series, n_bins: int = 3) -> pd.Series:
    """Assign balanced lesion-size bins, including for tiny pilot datasets.

    Ranking with ``method='first'`` avoids qcut failures when many lesions have
    identical volumes. Empty input and one-case input are handled explicitly.
    """
    values = pd.to_numeric(volumes_mm3, errors="coerce")
    result = pd.Series(index=values.index, dtype="object")
    valid = values.dropna()
    if valid.empty:
        return result

    bins = max(1, min(int(n_bins), len(valid), len(SIZE_LABELS)))
    labels = list(SIZE_LABELS[:bins])
    if bins == 1:
        result.loc[valid.index] = labels[0]
        return result

    ranked = valid.rank(method="first")
    result.loc[valid.index] = pd.qcut(ranked, q=bins, labels=labels).astype(str)
    return result


def sampling_weights_from_volume(volumes_mm3: pd.Series, floor: float = 1.0) -> pd.Series:
    """Return normalized inverse-volume weights for case-level sampling."""
    values = pd.to_numeric(volumes_mm3, errors="coerce").fillna(float(floor)).to_numpy(dtype=float)
    inverse = 1.0 / np.maximum(values, float(floor))
    total = float(inverse.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Could not derive finite sampling weights from lesion volumes")
    return pd.Series(inverse / total, index=volumes_mm3.index, dtype=float)


def build_case_metadata(
    case_ids: list[str],
    label_paths: dict[str, str],
    clinical_csv: str | None = None,
    clinical_id_col: str = "subject_id",
) -> pd.DataFrame:
    """Build a case-level metadata table from labels and optional clinical CSV."""
    rows = [
        {"case_id": case_id, "lesion_volume_mm3": lesion_volume_mm3(label_paths[case_id])}
        for case_id in case_ids
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["case_id", "lesion_volume_mm3", "size_bin", "sampling_weight"])

    df["size_bin"] = assign_size_bin(df["lesion_volume_mm3"])
    df["sampling_weight"] = sampling_weights_from_volume(df["lesion_volume_mm3"])

    if clinical_csv is not None:
        clinical = pd.read_csv(clinical_csv)
        if clinical_id_col not in clinical.columns:
            raise ValueError(f"Clinical CSV does not contain {clinical_id_col!r}")
        df["_raw_id"] = df["case_id"].str.extract(r"(\d+)$")[0]
        clinical["_raw_id"] = clinical[clinical_id_col].astype(str).str.extract(r"(\d+)$")[0]
        df = df.merge(clinical.drop(columns=[clinical_id_col]), on="_raw_id", how="left")
        df = df.drop(columns=["_raw_id"])

    return df
