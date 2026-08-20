#!/usr/bin/env python3
"""Plot the distribution of predicted foreground probability, split by GT class.

Expected input layout: same as ``analysis/dice_vs_threshold.py`` -- a
directory of per-case ``<case_id>.npz`` probability exports (``probabilities``
array, shape ``(num_classes, Z, Y, X)``) and a directory of matching
``<case_id>.nii.gz`` ground-truth labels.

Produces: ``<out-dir>/probability_histogram_<experiment-name>.png``, two
overlaid normalized histograms of the predicted foreground probability --
one over voxels that are truly lesion (GT=1), one over voxels that are truly
background (GT=0) -- plus a companion
``probability_histogram_<experiment-name>.csv`` of the raw bin counts.

Non-obvious rationale: with ~12M voxels/case x ~100+ cases, concatenating raw
voxel probabilities into one array is not memory-safe. Each case's voxels are
histogrammed into a fixed bin grid immediately and only the (tiny) per-bin
counts are accumulated across cases -- a running sum, not a growing array.
Y-axis is log-scale by default because background vastly outnumbers lesion
voxels; a linear axis would make the lesion-class histogram invisible.

Usage:
    python analysis/probability_histogram.py \\
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prob-dir", required=True, help="Directory of <case_id>.npz probability exports")
    parser.add_argument("--gt-dir", required=True, help="Directory of <case_id>.nii.gz ground-truth labels")
    parser.add_argument("--case-metadata-csv", required=True, help="CSV with case_id (+ optional split) columns")
    parser.add_argument("--split", default=None, help="Restrict to this value of the metadata's split column")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--out-dir", default="workspace/figures/threshold_analysis")
    parser.add_argument("--num-bins", type=int, default=50)
    parser.add_argument("--linear-yscale", action="store_true", help="Use a linear y-axis instead of log")
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

    bin_edges = np.linspace(0.0, 1.0, args.num_bins + 1)
    lesion_counts = np.zeros(args.num_bins, dtype=np.int64)
    background_counts = np.zeros(args.num_bins, dtype=np.int64)

    used, skipped, failed = 0, 0, 0
    for case_id in case_ids:
        prob_path = prob_dir / f"{case_id}.npz"
        gt_path = gt_dir / f"{case_id}.nii.gz"
        if not prob_path.is_file() or not gt_path.is_file():
            skipped += 1
            continue
        try:
            probs = np.load(prob_path)["probabilities"][1]  # foreground class
            gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))) > 0
            if probs.shape != gt.shape:
                raise ValueError(f"shape mismatch: probabilities {probs.shape}, ground truth {gt.shape}")
            lesion_counts += np.histogram(probs[gt], bins=bin_edges)[0]
            background_counts += np.histogram(probs[~gt], bins=bin_edges)[0]
            used += 1
        except Exception as exc:
            print(f"[error] {case_id}: {exc}")
            failed += 1
    if used == 0:
        raise SystemExit(f"No cases evaluated (skipped={skipped}, failed={failed}).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    summary = pd.DataFrame({
        "bin_low": bin_edges[:-1], "bin_high": bin_edges[1:], "bin_center": bin_centers,
        "lesion_voxel_count": lesion_counts, "background_voxel_count": background_counts,
    })
    csv_path = out_dir / f"probability_histogram_{args.experiment_name}.csv"
    summary.to_csv(csv_path, index=False)

    lesion_density = lesion_counts / max(lesion_counts.sum(), 1)
    background_density = background_counts / max(background_counts.sum(), 1)
    bin_width = bin_edges[1] - bin_edges[0]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(bin_centers, background_density, width=bin_width * 0.9, alpha=0.55,
             color="tab:gray", label=f"Background voxels (GT=0), n={background_counts.sum():,}")
    axis.bar(bin_centers, lesion_density, width=bin_width * 0.9, alpha=0.65,
             color="tab:red", label=f"Lesion voxels (GT=1), n={lesion_counts.sum():,}")
    if not args.linear_yscale:
        axis.set_yscale("log")
    axis.set_xlabel("Predicted foreground probability")
    axis.set_ylabel("Fraction of voxels" + ("" if args.linear_yscale else " (log scale)"))
    axis.set_title(f"Predicted probability by true class -- {args.experiment_name} (n={used} cases)")
    axis.legend()
    figure.tight_layout()
    png_path = out_dir / f"probability_histogram_{args.experiment_name}.png"
    figure.savefig(png_path, dpi=180)
    plt.close(figure)

    print(f"Used {used} cases (skipped={skipped}, failed={failed})")
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
