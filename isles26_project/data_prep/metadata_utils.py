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
    """Return normalized inverse-volume weights for case-level sampling.

    DO NOT USE for lesion-aware sampling -- kept only for reference/comparison.
    A fixed floor is fragile against the real volume distribution: on the ATLAS
    R3.0 real dataset, 3 zero-volume (empty-mask) cases alone captured 66% of
    all sampling probability under this formula (floor=1.0mm3 is negligible
    next to a ~4600mm3 median lesion), collapsing training to a "predict
    nothing" degenerate solution (pseudo Dice pinned at 0.0 for 96+ epochs
    before the bug was caught -- see CLAUDE.md). Use
    sampling_weights_from_size_bin instead.
    """
    values = pd.to_numeric(volumes_mm3, errors="coerce").fillna(float(floor)).to_numpy(dtype=float)
    inverse = 1.0 / np.maximum(values, float(floor))
    total = float(inverse.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Could not derive finite sampling weights from lesion volumes")
    return pd.Series(inverse / total, index=volumes_mm3.index, dtype=float)


# small:medium:large -- geometric 4:2:1 spacing. Bins are already ~balanced in
# case count (tertile split), so this ratio translates directly into expected
# oversampling of small lesions relative to their natural frequency, which is
# the actual goal of lesion-aware sampling.
SIZE_BIN_SAMPLING_MULTIPLIERS = {"small": 4.0, "medium": 2.0, "large": 1.0}


def sampling_weights_from_size_bin(
    size_bins: pd.Series, multipliers: dict[str, float] | None = None
) -> pd.Series:
    """Return normalized per-case sampling weights from lesion-size bin membership.

    Bin-level (not continuous inverse-volume) reweighting: every case in a bin
    gets the same weight, so no single pathological case (e.g. a near-zero or
    exactly-zero lesion volume) can dominate sampling probability the way
    continuous 1/volume weighting did -- see sampling_weights_from_volume's
    docstring for what went wrong. Small lesions are still intentionally
    oversampled relative to their natural (already roughly balanced) frequency,
    which is the actual point of lesion-aware sampling.
    """
    mult = multipliers or SIZE_BIN_SAMPLING_MULTIPLIERS
    bins = size_bins.astype(str)
    unknown = set(bins.unique()) - set(mult)
    if unknown:
        raise ValueError(f"No sampling multiplier defined for size_bin value(s): {sorted(unknown)}")
    weights = bins.map(mult).astype(float)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Could not derive finite sampling weights from size bins")
    return weights / total


def sampling_weights_from_size_bin_power(
    size_bins: pd.Series, volumes_mm3: pd.Series, p: float = 0.5
) -> pd.Series:
    """Bin-level sampling weights via power-law interpolation, grounded in real per-bin volume mass.

    weight(bin) is proportional to (1 / total_lesion_volume_in_bin) ** p, and every
    case in a bin gets that same weight (bin-level, not per-case -- avoids the
    case-level pathology described in sampling_weights_from_volume's docstring).

    p is a single interpretable knob between two well-defined endpoints:
    - p=0: uniform: reproduces natural case-balanced sampling (equivalent to
      SIZE_BIN_SAMPLING_MULTIPLIERS all set to 1).
    - p=1: full correction: equalizes total foreground-voxel exposure per bin per
      epoch. On the real ATLAS R3.0 training pool this works out to roughly
      small:medium:large = 102:12:1 -- likely too extreme (undertrains "large").
    - p=0.5 (sqrt-dampened): a reasonable middle ground, works out to roughly
      10:3.5:1 on the real pool. Recommended starting point if replacing the
      original fixed-ratio SIZE_BIN_SAMPLING_MULTIPLIERS (4:2:1, chosen as an
      arbitrary geometric default, not derived from the data) with something
      grounded in the actual per-bin volume imbalance -- see CLAUDE.md decisions
      log for the derivation and the reasoning against p=1.
    """
    bins = size_bins.astype(str)
    volumes = pd.to_numeric(volumes_mm3, errors="coerce")
    bin_totals = volumes.groupby(bins).sum()
    bad_bins = bin_totals[(bin_totals <= 0) | bin_totals.isna()].index.tolist()
    if bad_bins:
        raise ValueError(f"Non-positive or missing total volume for size_bin(s): {bad_bins}")

    bin_weight = (1.0 / bin_totals) ** float(p)
    per_case = bins.map(bin_weight)
    total = float(per_case.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Could not derive finite sampling weights from size-bin power weighting")
    result = per_case / total
    result.index = size_bins.index
    return result


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
    df["sampling_weight"] = sampling_weights_from_size_bin(df["size_bin"])

    if clinical_csv is not None:
        clinical = pd.read_csv(clinical_csv)
        if clinical_id_col not in clinical.columns:
            raise ValueError(f"Clinical CSV does not contain {clinical_id_col!r}")
        df["_raw_id"] = df["case_id"].str.extract(r"(\d+)$")[0]
        clinical["_raw_id"] = clinical[clinical_id_col].astype(str).str.extract(r"(\d+)$")[0]
        df = df.merge(clinical.drop(columns=[clinical_id_col]), on="_raw_id", how="left")
        df = df.drop(columns=["_raw_id"])

    return df
