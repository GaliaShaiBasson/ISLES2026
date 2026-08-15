#!/usr/bin/env python3
"""Compute per-case Dice and HD95, then join case metadata."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt


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


def evaluate_case(pred_path: Path, gt_path: Path) -> dict:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required for NIfTI evaluation") from exc

    pred_image = nib.load(str(pred_path))
    gt_image = nib.load(str(gt_path))
    if pred_image.shape != gt_image.shape:
        raise ValueError(f"Shape mismatch: prediction {pred_image.shape}, ground truth {gt_image.shape}")
    if not np.allclose(pred_image.affine, gt_image.affine, rtol=1e-4, atol=1e-3):
        raise ValueError("Prediction and ground truth affines do not match")

    pred = np.asanyarray(pred_image.dataobj) > 0
    gt = np.asanyarray(gt_image.dataobj) > 0
    spacing = gt_image.header.get_zooms()[:3]
    return {
        "dice": dice_score(pred, gt),
        "hd95_mm": hausdorff_distance_95(pred, gt, spacing),
        "gt_empty": bool(not gt.any()),
        "pred_empty": bool(not pred.any()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--case-metadata-csv", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--out-csv", required=True)
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
            metrics = evaluate_case(pred_path, gt_path)
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
    print(output[["dice", "hd95_mm"]].describe().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
