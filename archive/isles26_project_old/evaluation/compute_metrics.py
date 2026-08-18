#!/usr/bin/env python
"""
Compute per-case Dice and HD95 between nnU-Net predictions and ground truth,
then join with `case_metadata.csv` so results can be stratified by lesion
size / center / chronicity later.

Usage:
    python compute_metrics.py \
        --pred-dir /path/to/nnUNet_results/.../fold_0/validation \
        --gt-dir   /path/to/nnUNet_raw/Dataset001_ISLES26/labelsTr \
        --case-metadata-csv /path/to/case_metadata.csv \
        --experiment-name baseline \
        --out-csv results_baseline.csv

Run once per trained model/experiment, then combine everything with
`aggregate_results.py`.
"""
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import distance_transform_edt, binary_erosion


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Boolean array marking voxels on the boundary of the foreground mask."""
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask)
    return mask & ~eroded


def hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    """95th-percentile symmetric Hausdorff distance, in the units of `spacing`
    (mm if spacing comes from the NIfTI header, as used below).

    Standard approach: for each surface voxel of A, find the distance to the
    nearest surface voxel of B (via a distance transform seeded at B's
    surface), and vice versa; HD95 is the 95th percentile over the union of
    both directed distance sets. Returns NaN if either mask is empty (Dice
    with an empty ground truth / empty prediction should be inspected
    separately rather than folded into an HD95 average).
    """
    pred_surf = _surface_voxels(pred.astype(bool))
    gt_surf = _surface_voxels(gt.astype(bool))
    if pred_surf.sum() == 0 or gt_surf.sum() == 0:
        return float("nan")

    dt_from_gt = distance_transform_edt(~gt_surf, sampling=spacing)
    dt_from_pred = distance_transform_edt(~pred_surf, sampling=spacing)

    d_pred_to_gt = dt_from_gt[pred_surf]
    d_gt_to_pred = dt_from_pred[gt_surf]

    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_d, 95))


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0  # both empty: trivially perfect agreement
    intersection = np.logical_and(pred, gt).sum()
    return float(2 * intersection / (pred.sum() + gt.sum() + 1e-8))


def evaluate_case(pred_path: Path, gt_path: Path) -> dict:
    pred_img = nib.load(str(pred_path))
    gt_img = nib.load(str(gt_path))
    pred = (pred_img.get_fdata() > 0).astype(np.uint8)
    gt = (gt_img.get_fdata() > 0).astype(np.uint8)
    spacing = gt_img.header.get_zooms()[:3]

    return {
        "dice": dice_score(pred, gt),
        "hd95_mm": hausdorff_distance_95(pred, gt, spacing),
        "gt_empty": bool(gt.sum() == 0),
        "pred_empty": bool(pred.sum() == 0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", required=True, help="Folder of predicted .nii.gz masks (filenames matching case_id.nii.gz)")
    parser.add_argument("--gt-dir", required=True, help="Folder of ground-truth .nii.gz masks")
    parser.add_argument("--case-metadata-csv", required=True, help="case_metadata.csv from data_prep")
    parser.add_argument("--experiment-name", required=True, help="Label for this run, e.g. 'baseline', 'focal_tversky', 'lesion_aware_sampling'")
    parser.add_argument("--out-csv", required=True, help="Where to write per-case results for this experiment")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    meta = pd.read_csv(args.case_metadata_csv)

    rows = []
    for _, row in meta.iterrows():
        case_id = row["case_id"]
        pred_path = pred_dir / f"{case_id}.nii.gz"
        gt_path = gt_dir / f"{case_id}.nii.gz"
        if not pred_path.exists():
            print(f"[skip] no prediction for {case_id} at {pred_path}")
            continue
        if not gt_path.exists():
            print(f"[skip] no ground truth for {case_id} at {gt_path}")
            continue

        metrics = evaluate_case(pred_path, gt_path)
        result = {"case_id": case_id, "experiment": args.experiment_name, **metrics}
        for col in meta.columns:
            if col != "case_id":
                result[col] = row[col]
        rows.append(result)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_df)} case results to {args.out_csv}")
    print(out_df[["dice", "hd95_mm"]].describe())


if __name__ == "__main__":
    main()
