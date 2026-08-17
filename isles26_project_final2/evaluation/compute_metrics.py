#!/usr/bin/env python3
"""Compute strict per-case Dice and HD95 metrics, then join case metadata.

The expected evaluation cohort is explicit. By default every metadata row is
required. For cross-validation, pass nnU-Net's ``splits_final.json`` and a fold.
Missing predictions are fatal unless ``--allow-missing`` is deliberately used.
"""
from __future__ import annotations

import argparse
import json
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
    """Return symmetric HD95, or NaN when exactly one surface is absent."""
    pred_surface = _surface_voxels(pred)
    gt_surface = _surface_voxels(gt)
    if not pred_surface.any() and not gt_surface.any():
        return 0.0
    if not pred_surface.any() or not gt_surface.any():
        return float("nan")
    from_gt = distance_transform_edt(~gt_surface, sampling=spacing)[pred_surface]
    from_pred = distance_transform_edt(~pred_surface, sampling=spacing)[gt_surface]
    return float(np.percentile(np.concatenate([from_gt, from_pred]), 95))


def physical_diagonal_mm(shape, spacing) -> float:
    """Maximum corner-to-corner distance in the physical image grid."""
    extents = (np.asarray(shape[:3], dtype=float) - 1.0) * np.asarray(spacing[:3], dtype=float)
    return float(np.linalg.norm(np.maximum(extents, 0.0)))


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
    raw_hd95 = hausdorff_distance_95(pred, gt, spacing)
    hd95_defined = bool(np.isfinite(raw_hd95))
    penalty = physical_diagonal_mm(gt_image.shape, spacing)
    return {
        "dice": dice_score(pred, gt),
        "hd95_mm": raw_hd95,
        # Empty-vs-nonempty masks have no finite surface distance. The image
        # diagonal is a transparent, case-specific worst-distance penalty.
        "hd95_penalized_mm": raw_hd95 if hd95_defined else penalty,
        "hd95_defined": hd95_defined,
        "hd95_penalty_mm": penalty,
        "gt_empty": bool(not gt.any()),
        "pred_empty": bool(not pred.any()),
    }


def _read_case_list(path: Path) -> list[str]:
    if not path.is_file():
        raise ValueError(f"Expected-case file not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("case_ids")
        if not isinstance(payload, list):
            raise ValueError("Expected-case JSON must be a list or contain a case_ids list")
        return [str(value) for value in payload]
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if "case_id" not in frame.columns:
            raise ValueError("Expected-case CSV must contain a case_id column")
        return frame["case_id"].astype(str).tolist()
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def expected_case_ids(
    metadata: pd.DataFrame,
    expected_cases_file: str | None = None,
    splits_json: str | None = None,
    fold: str | None = None,
) -> list[str]:
    """Resolve the exact cohort from a file, an nnU-Net fold, or all metadata."""
    if expected_cases_file and splits_json:
        raise ValueError("Use either --expected-cases-file or --splits-json, not both")
    if expected_cases_file:
        cases = _read_case_list(Path(expected_cases_file))
    elif splits_json:
        if fold is None:
            raise ValueError("--fold is required with --splits-json")
        try:
            fold_index = int(fold)
        except ValueError as exc:
            raise ValueError("Cross-validation fold must be an integer") from exc
        path = Path(splits_json)
        if not path.is_file():
            raise ValueError(f"nnU-Net split file not found: {path}")
        splits = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(splits, list) or not 0 <= fold_index < len(splits):
            raise ValueError(f"Fold {fold_index} is not present in {path}")
        cases = [str(value) for value in splits[fold_index].get("val", [])]
    else:
        cases = metadata["case_id"].astype(str).tolist()

    if not cases:
        raise ValueError("The expected evaluation cohort is empty")
    if len(cases) != len(set(cases)):
        raise ValueError("The expected evaluation cohort contains duplicate case IDs")
    metadata_cases = set(metadata["case_id"].astype(str))
    absent = sorted(set(cases).difference(metadata_cases))
    if absent:
        raise ValueError(f"Metadata is missing {len(absent)} expected cases, for example: {absent[:5]}")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--case-metadata-csv", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--out-csv", required=True)
    cohort = parser.add_mutually_exclusive_group()
    cohort.add_argument("--expected-cases-file")
    cohort.add_argument("--splits-json")
    parser.add_argument("--fold", help="Fold index used with --splits-json")
    parser.add_argument("--fold-label", help="Value recorded in the output fold column")
    parser.add_argument("--allow-missing", action="store_true", help="Skip missing expected files instead of failing")
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
    metadata["case_id"] = metadata["case_id"].astype(str)
    if metadata["case_id"].duplicated().any():
        parser.error("metadata CSV contains duplicate case_id values")
    metadata_by_case = metadata.set_index("case_id", drop=False)

    try:
        case_ids = expected_case_ids(metadata, args.expected_cases_file, args.splits_json, args.fold)
    except ValueError as exc:
        parser.error(str(exc))

    observed_predictions = {path.name[:-7] for path in pred_dir.glob("*.nii.gz")}
    extra_predictions = sorted(observed_predictions.difference(case_ids))
    if extra_predictions:
        parser.error(
            f"Prediction directory contains {len(extra_predictions)} cases outside the expected cohort; "
            f"examples: {extra_predictions[:5]}"
        )

    missing: list[str] = []
    for case_id in case_ids:
        if not (pred_dir / f"{case_id}.nii.gz").is_file() or not (gt_dir / f"{case_id}.nii.gz").is_file():
            missing.append(case_id)
    if missing and not args.allow_missing:
        parser.error(
            f"{len(missing)} of {len(case_ids)} expected cases are missing a prediction or ground truth; "
            f"examples: {missing[:5]}. Use --allow-missing only for debugging."
        )

    rows: list[dict] = []
    failures: list[str] = []
    for case_id in case_ids:
        pred_path = pred_dir / f"{case_id}.nii.gz"
        gt_path = gt_dir / f"{case_id}.nii.gz"
        if not pred_path.is_file() or not gt_path.is_file():
            continue
        try:
            metrics = evaluate_case(pred_path, gt_path)
        except Exception as exc:
            failures.append(f"{case_id}: {exc}")
            continue
        metadata_row = metadata_by_case.loc[case_id].to_dict()
        metadata_row.pop("case_id", None)
        result = {"case_id": case_id, "experiment": args.experiment_name, **metrics, **metadata_row}
        if args.fold_label is not None:
            result["fold"] = args.fold_label
        rows.append(result)

    if failures:
        raise SystemExit(f"Evaluation failed for {len(failures)} cases; examples: {failures[:3]}")
    if not rows:
        raise SystemExit("No cases were evaluated successfully")

    output = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    print(f"Wrote {len(output)}/{len(case_ids)} expected rows to {out_path}")
    print(output[["dice", "hd95_penalized_mm", "pred_empty"]].describe().to_string())
    if missing:
        print(f"WARNING: skipped {len(missing)} expected cases because --allow-missing was used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
