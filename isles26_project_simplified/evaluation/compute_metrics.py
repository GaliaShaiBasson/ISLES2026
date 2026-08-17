#!/usr/bin/env python3
"""Compute per-case Dice, HD95, and lesion-wise F1, then join case metadata."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure, label


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool, copy=False)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return mask & ~binary_erosion(mask)


def hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    pred_surface = _surface_voxels(pred)
    gt_surface = _surface_voxels(gt)
    if not pred_surface.any() or not gt_surface.any():
        return float("nan")
    from_gt = distance_transform_edt(~gt_surface, sampling=spacing)[pred_surface]
    from_pred = distance_transform_edt(~pred_surface, sampling=spacing)[gt_surface]
    return float(np.percentile(np.concatenate([from_gt, from_pred]), 95))


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool, copy=False)
    gt = gt.astype(bool, copy=False)
    denominator = int(pred.sum() + gt.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(pred, gt).sum() / denominator)


def _drop_small_components(labels: np.ndarray, n: int, min_voxels: int) -> tuple[np.ndarray, int]:
    """Relabel connected components 1..n, discarding any below min_voxels size."""
    if n == 0 or min_voxels <= 1:
        return labels, n
    counts = np.bincount(labels.ravel(), minlength=n + 1)
    keep = np.flatnonzero(counts[1:] >= min_voxels) + 1
    if len(keep) == n:
        return labels, n
    remap = np.zeros(n + 1, dtype=labels.dtype)
    remap[keep] = np.arange(1, len(keep) + 1, dtype=labels.dtype)
    return remap[labels], len(keep)


def lesion_wise_f1(pred: np.ndarray, gt: np.ndarray, connectivity: int = 3, min_lesion_voxels: int = 0) -> dict:
    """Instance-level (lesion-wise) F1, as opposed to dice_score's voxel-level overlap.

    Connected-component matching: a predicted lesion is a true positive if it shares at
    least one voxel with any ground-truth lesion (the "any-overlap" convention used by
    ISLES/BraTS-style lesion-wise scoring). TP is counted per matched ground-truth lesion,
    not per predicted component -- a single true lesion fragmented into several predicted
    blobs still counts as one TP, since each fragment did find its target; only predicted
    blobs with zero overlap anywhere count as FP. This rewards detecting a lesion's
    presence and penalizes phantom/missed lesions, without being derailed by how cleanly a
    detected lesion's boundary is drawn (that's what voxel dice_score is for).

    connectivity=3 (26-connected, i.e. face+edge+corner neighbors) is the default so
    diagonally-touching lesion voxels count as one lesion rather than splitting into
    several -- matches typical clinical lesion counting. min_lesion_voxels defaults to 0
    (no filtering); pass a small threshold to ignore single/few-voxel noise blobs that
    would otherwise inflate FP/FN counts without being clinically meaningful lesions.
    """
    structure = generate_binary_structure(3, connectivity)
    pred_labels, n_pred = label(pred, structure=structure)
    gt_labels, n_gt = label(gt, structure=structure)
    pred_labels, n_pred = _drop_small_components(pred_labels, n_pred, min_lesion_voxels)
    gt_labels, n_gt = _drop_small_components(gt_labels, n_gt, min_lesion_voxels)

    if n_gt == 0 and n_pred == 0:
        return {"lesion_f1": 1.0, "lesion_tp": 0, "lesion_fp": 0, "lesion_fn": 0,
                "n_gt_lesions": 0, "n_pred_lesions": 0}

    matched_gt_ids: set[int] = set()
    matched_pred_count = 0
    for p in range(1, n_pred + 1):
        overlapping_gt_ids = set(np.unique(gt_labels[pred_labels == p]).tolist()) - {0}
        if overlapping_gt_ids:
            matched_pred_count += 1
            matched_gt_ids.update(overlapping_gt_ids)

    tp = len(matched_gt_ids)
    fn = n_gt - tp
    fp = n_pred - matched_pred_count
    denominator = 2 * tp + fp + fn
    f1 = 1.0 if denominator == 0 else (2 * tp) / denominator
    return {"lesion_f1": f1, "lesion_tp": tp, "lesion_fp": fp, "lesion_fn": fn,
            "n_gt_lesions": n_gt, "n_pred_lesions": n_pred}


def evaluate_case(pred_path: Path, gt_path: Path, lesion_connectivity: int = 3,
                  min_lesion_voxels: int = 0) -> dict:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("SimpleITK is required for NIfTI evaluation") from exc

    # Read geometry via SimpleITK, not nibabel. nnU-Net's own SimpleITKIO reads/writes
    # every image (input, ground truth, prediction) through SimpleITK exclusively, which
    # prefers a NIfTI file's qform. nibabel's default `.affine` instead prefers sform when
    # sform_code > 0. Some ATLAS label files carry a stale/divergent qform-vs-sform pair
    # (e.g. left over from a registration step that updated one but not the other), so
    # comparing nibabel's affine against what nnU-Net actually used flags a false
    # mismatch -- verified directly against ATLAS_r028s012_ses1/r028s017_ses1: SimpleITK
    # reads byte-identical origin/direction for the image and the ground-truth label
    # (matching the prediction nnU-Net wrote), even though nibabel's sform-based affine
    # for the label disagrees. The voxel arrays genuinely correspond 1:1 in the frame
    # nnU-Net operates in; that's the frame this check needs to agree with.
    pred_image = sitk.ReadImage(str(pred_path))
    gt_image = sitk.ReadImage(str(gt_path))
    if pred_image.GetSize() != gt_image.GetSize():
        raise ValueError(f"Shape mismatch: prediction {pred_image.GetSize()}, ground truth {gt_image.GetSize()}")
    if not np.allclose(pred_image.GetSpacing(), gt_image.GetSpacing(), rtol=1e-4, atol=1e-3):
        raise ValueError("Prediction and ground truth spacing do not match")
    if not np.allclose(pred_image.GetOrigin(), gt_image.GetOrigin(), rtol=1e-4, atol=1e-3) or \
            not np.allclose(pred_image.GetDirection(), gt_image.GetDirection(), rtol=1e-4, atol=1e-3):
        raise ValueError("Prediction and ground truth geometry (origin/direction) do not match")

    pred = sitk.GetArrayFromImage(pred_image) > 0
    gt = sitk.GetArrayFromImage(gt_image) > 0
    spacing = gt_image.GetSpacing()[::-1]  # sitk spacing is (x,y,z); GetArrayFromImage returns (z,y,x)
    return {
        "dice": dice_score(pred, gt),
        "hd95_mm": hausdorff_distance_95(pred, gt, spacing),
        "gt_empty": bool(not gt.any()),
        "pred_empty": bool(not pred.any()),
        **lesion_wise_f1(pred, gt, connectivity=lesion_connectivity, min_lesion_voxels=min_lesion_voxels),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--case-metadata-csv", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--lesion-connectivity", type=int, default=3, choices=(1, 2, 3),
                        help="scipy connectivity for lesion-wise components (1=6-connected, "
                             "3=26-connected, default 3)")
    parser.add_argument("--min-lesion-voxels", type=int, default=0,
                        help="drop connected components smaller than this many voxels before "
                             "lesion-wise matching (default 0 = no filtering)")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    metadata_path = Path(args.case_metadata_csv)
    for label, path in (("prediction directory", pred_dir), ("ground-truth directory", gt_dir)):
        if not path.is_dir():
            parser.error(f"{label} not found: {path}")
    if not metadata_path.is_file():
        parser.error(f"metadata CSV not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    if "case_id" not in metadata.columns:
        parser.error("metadata CSV must contain a case_id column")
    if metadata["case_id"].duplicated().any():
        parser.error("metadata CSV contains duplicate case_id values")

    rows: list[dict] = []
    skipped = 0
    failed = 0
    for row in metadata.to_dict(orient="records"):
        case_id = str(row["case_id"])
        pred_path = pred_dir / f"{case_id}.nii.gz"
        gt_path = gt_dir / f"{case_id}.nii.gz"
        if not pred_path.is_file() or not gt_path.is_file():
            print(f"[skip] {case_id}: prediction or ground truth missing")
            skipped += 1
            continue
        try:
            metrics = evaluate_case(pred_path, gt_path, lesion_connectivity=args.lesion_connectivity,
                                    min_lesion_voxels=args.min_lesion_voxels)
        except Exception as exc:
            print(f"[error] {case_id}: {exc}")
            failed += 1
            continue
        rows.append({"case_id": case_id, "experiment": args.experiment_name, **metrics, **{k: v for k, v in row.items() if k != "case_id"}})

    if not rows:
        raise SystemExit(f"No cases were evaluated successfully (skipped={skipped}, failed={failed}).")

    output = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    print(f"Wrote {len(output)} rows to {out_path} (skipped={skipped}, failed={failed})")
    print(output[["dice", "hd95_mm", "lesion_f1"]].describe().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
