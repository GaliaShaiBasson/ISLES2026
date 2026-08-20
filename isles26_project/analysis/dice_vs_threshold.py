#!/usr/bin/env python3
"""Sweep the foreground-probability threshold and plot Dice vs. threshold.

Expected input layout: a directory of per-case nnU-Net probability exports
(``<case_id>.npz``, each holding a ``probabilities`` array of shape
``(num_classes, Z, Y, X)`` in the same voxel grid as the ground-truth label --
this is what ``nnUNetv2_train ... --val --npz`` / ``nnUNetv2_predict
--save_probabilities`` write) and a directory of matching ``<case_id>.nii.gz``
ground-truth labels.

Produces: ``<out-dir>/dice_vs_threshold_<experiment-name>.png`` (mean Dice +/-
std across cases at each threshold, with the current fixed 0.5 operating
point and the best-found threshold both marked) and a companion
``dice_vs_threshold_<experiment-name>.csv`` (one row per threshold: mean/std/
median Dice) so the sweep doesn't have to be rerun to check exact numbers.

Non-obvious rationale: nnU-Net's own argmax-based hard-label export is
equivalent to thresholding the foreground class's softmax probability at
0.5 (for the binary lesion/background case here) -- that fixed 0.5 was never
tuned against this project's data, it's just the implicit default. This
script re-derives per-case Dice at a grid of alternative thresholds from the
already-computed probability maps, without re-running inference, to check
whether 0.5 is actually where Dice peaks on this split.

Usage:
    python analysis/dice_vs_threshold.py \\
        --prob-dir /path/to/validation \\
        --gt-dir /path/to/labelsTr \\
        --case-metadata-csv workspace/splits_dataset002/manifest.csv \\
        --split val \\
        --experiment-name baseline_val \\
        --out-dir workspace/figures/threshold_analysis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool, copy=False)
    gt = gt.astype(bool, copy=False)
    denom = int(pred.sum() + gt.sum())
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(pred, gt).sum() / denom)


def per_case_dice_curve(prob_path: Path, gt_path: Path, thresholds: np.ndarray) -> np.ndarray:
    probs = np.load(prob_path)["probabilities"][1]  # foreground class
    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))) > 0
    if probs.shape != gt.shape:
        raise ValueError(f"Shape mismatch: probabilities {probs.shape}, ground truth {gt.shape}")
    return np.array([dice_score(probs >= t, gt) for t in thresholds])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prob-dir", required=True, help="Directory of <case_id>.npz probability exports")
    parser.add_argument("--gt-dir", required=True, help="Directory of <case_id>.nii.gz ground-truth labels")
    parser.add_argument("--case-metadata-csv", required=True, help="CSV with case_id (+ optional split) columns")
    parser.add_argument("--split", default=None, help="Restrict to this value of the metadata's split column")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--out-dir", default="workspace/figures/threshold_analysis")
    parser.add_argument("--thresholds", default="0.05:0.95:0.05",
                        help="start:stop:step for the probability grid (default 0.05:0.95:0.05)")
    args = parser.parse_args()

    prob_dir = Path(args.prob_dir)
    gt_dir = Path(args.gt_dir)
    for label, path in (("probability directory", prob_dir), ("ground-truth directory", gt_dir)):
        if not path.is_dir():
            parser.error(f"{label} not found: {path}")

    metadata = pd.read_csv(args.case_metadata_csv)
    if "case_id" not in metadata.columns:
        parser.error("metadata CSV must contain a case_id column")
    if args.split:
        if "split" not in metadata.columns:
            parser.error("--split given but metadata CSV has no split column")
        metadata = metadata[metadata["split"] == args.split]
    case_ids = metadata["case_id"].astype(str).tolist()

    start, stop, step = (float(x) for x in args.thresholds.split(":"))
    thresholds = np.round(np.arange(start, stop + step / 2, step), 4)

    curves = []
    used, skipped, failed = 0, 0, 0
    for case_id in case_ids:
        prob_path = prob_dir / f"{case_id}.npz"
        gt_path = gt_dir / f"{case_id}.nii.gz"
        if not prob_path.is_file() or not gt_path.is_file():
            skipped += 1
            continue
        try:
            curves.append(per_case_dice_curve(prob_path, gt_path, thresholds))
            used += 1
        except Exception as exc:
            print(f"[error] {case_id}: {exc}")
            failed += 1
    if not curves:
        raise SystemExit(f"No cases evaluated (skipped={skipped}, failed={failed}).")

    curves = np.stack(curves)  # (n_cases, n_thresholds)
    mean_dice = curves.mean(axis=0)
    std_dice = curves.std(axis=0)
    median_dice = np.median(curves, axis=0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame({
        "threshold": thresholds, "mean_dice": mean_dice, "std_dice": std_dice, "median_dice": median_dice,
    })
    csv_path = out_dir / f"dice_vs_threshold_{args.experiment_name}.csv"
    summary.to_csv(csv_path, index=False)

    best_idx = int(np.argmax(mean_dice))
    default_idx = int(np.argmin(np.abs(thresholds - 0.5)))

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(thresholds, mean_dice, marker="o", color="tab:blue", label="Mean Dice")
    axis.fill_between(thresholds, mean_dice - std_dice, mean_dice + std_dice, alpha=0.15, color="tab:blue")
    axis.axvline(0.5, color="gray", linestyle="--", linewidth=1,
                 label=f"Current default (0.5): {mean_dice[default_idx]:.3f}")
    axis.axvline(thresholds[best_idx], color="tab:red", linestyle=":", linewidth=1.5,
                 label=f"Best ({thresholds[best_idx]:.2f}): {mean_dice[best_idx]:.3f}")
    axis.set_xlabel("Foreground probability threshold")
    axis.set_ylabel("Dice")
    axis.set_title(f"Dice vs. threshold -- {args.experiment_name} (n={used} cases)")
    axis.legend()
    figure.tight_layout()
    png_path = out_dir / f"dice_vs_threshold_{args.experiment_name}.png"
    figure.savefig(png_path, dpi=180)
    plt.close(figure)

    print(f"Used {used} cases (skipped={skipped}, failed={failed})")
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Default threshold 0.5: mean Dice {mean_dice[default_idx]:.4f}")
    print(f"Best threshold {thresholds[best_idx]:.2f}: mean Dice {mean_dice[best_idx]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
