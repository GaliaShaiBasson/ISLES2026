#!/usr/bin/env python3
"""Connected-component size filtering for held-out predictions.

Expected input layout: --pred-dir is a flat directory of `<case_id>.nii.gz`
binary/label NIfTI predictions, e.g. nnU-Net's `predTs` output. No other
input required beyond the images themselves -- this does not need ground
truth, so it can run on any prediction directory as soon as inference is
done.

Produces: one filtered `<case_id>.nii.gz` per input case under --out-dir,
byte-identical geometry (spacing/origin/direction) to the input, with any
foreground connected component smaller than --min-voxels voxels zeroed out.
Also writes a `postprocess_summary.csv` (one row per case: voxels before/
after, components before/after, voxels/components removed) to --out-dir --
this is the input for a before/after report table or figure, not just a log.

Non-obvious rationale: nnU-Net's raw argmax output tends to include a small
number of few-voxel false-positive blobs. These barely move Dice (their
voxel count is negligible against a real lesion) but can badly inflate
HD95, since Hausdorff distance is a max/percentile-of-distances metric --
one stray voxel far from the true lesion contributes a large distance no
matter how small it is. Filtering by absolute component size, not by
"keep only the largest component," is deliberate: ISLES/ATLAS stroke cases
are frequently multifocal (multiple real infarct territories), so a
largest-only heuristic would delete genuine lesions, not just noise. This
mirrors `compute_metrics.py`'s `_drop_small_components` (used there only to
adjust lesion-wise F1 scoring in memory) but is a distinct, disk-writing
step: it actually mutates the predictions that get evaluated/reported,
rather than filtering only inside one metric's computation. Connectivity
defaults to 26-connected (3), matching `lesion_wise_f1`'s default so a
"lesion" here means the same voxel-adjacency convention project-wide.

--min-voxels has no built-in default on purpose -- pick it by inspecting
the removed-component size distribution on a val-set run first (rerun with
a candidate threshold, check `postprocess_summary.csv` and/or eyeball a
few `components_removed > 0` cases) rather than guessing a number that
happens to look reasonable. A threshold tuned on val, then applied frozen
to test_id/test_ood, keeps the comparison honest -- do not re-tune per
test split.

Usage:
    python evaluation/postprocess_predictions.py \\
        --pred-dir workspace/results/runs/<run_id>/predTs \\
        --out-dir workspace/results/runs/<run_id>/predTs_cc10 \\
        --min-voxels 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure, label


def drop_small_components(mask: np.ndarray, min_voxels: int, connectivity: int = 3) -> tuple[np.ndarray, dict]:
    """Zero out foreground connected components smaller than min_voxels.

    Returns the filtered boolean mask plus a stats dict (voxel/component counts
    before and after) for the summary CSV.
    """
    structure = generate_binary_structure(3, connectivity)
    labels, n_components = label(mask, structure=structure)
    voxels_before = int(mask.sum())

    if n_components == 0 or min_voxels <= 1:
        return mask.astype(bool, copy=False), {
            "voxels_before": voxels_before,
            "voxels_after": voxels_before,
            "components_before": n_components,
            "components_after": n_components,
            "components_removed": 0,
        }

    counts = np.bincount(labels.ravel(), minlength=n_components + 1)
    keep_ids = np.flatnonzero(counts[1:] >= min_voxels) + 1
    filtered = np.isin(labels, keep_ids)
    return filtered, {
        "voxels_before": voxels_before,
        "voxels_after": int(filtered.sum()),
        "components_before": n_components,
        "components_after": len(keep_ids),
        "components_removed": n_components - len(keep_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pred-dir", required=True, help="Directory of <case_id>.nii.gz predictions to filter")
    parser.add_argument("--out-dir", required=True, help="Directory to write filtered predictions + summary CSV")
    parser.add_argument("--min-voxels", type=int, required=True,
                        help="Drop connected components smaller than this many voxels. Tune on val "
                             "first (see module docstring) -- no default is provided deliberately.")
    parser.add_argument("--connectivity", type=int, default=3, choices=(1, 2, 3),
                        help="scipy connectivity for components (1=6-connected, 3=26-connected, "
                             "default 3, matching compute_metrics.py's lesion_wise_f1 default)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if --out-dir already has a postprocess_summary.csv")
    args = parser.parse_args()

    try:
        import SimpleITK as sitk
    except ImportError as exc:
        parser.error(f"SimpleITK is required: {exc}")

    pred_dir = Path(args.pred_dir).expanduser().resolve()
    if not pred_dir.is_dir():
        parser.error(f"prediction directory not found: {pred_dir}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    # Hard guard, not just a convention: filtered output must never land in (or contain)
    # the input directory, so a mistyped --out-dir can't silently overwrite the raw
    # predictions the report's "already safe" held-out-predictions section relies on.
    if out_dir == pred_dir or pred_dir in out_dir.parents or out_dir in pred_dir.parents:
        parser.error(
            f"--out-dir ({out_dir}) must not be the same as, or nested with, "
            f"--pred-dir ({pred_dir}) -- raw predictions are never overwritten in place."
        )
    summary_path = out_dir / "postprocess_summary.csv"
    if summary_path.is_file() and not args.overwrite:
        print(f"[skip] {summary_path} already exists -- nothing to redo. Pass --overwrite to recompute.")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_paths = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_paths:
        raise SystemExit(f"No *.nii.gz files found in {pred_dir}")

    rows: list[dict] = []
    for pred_path in pred_paths:
        case_id = pred_path.name[: -len(".nii.gz")]
        image = sitk.ReadImage(str(pred_path))
        # >0 mirrors compute_metrics.py's evaluate_case binarization -- multi-class label
        # maps aren't expected here (ISLES/ATLAS lesion masks are binary), but this keeps
        # the convention consistent if that ever changes.
        mask = sitk.GetArrayFromImage(image) > 0
        filtered, stats = drop_small_components(mask, args.min_voxels, connectivity=args.connectivity)

        out_image = sitk.GetImageFromArray(filtered.astype(np.uint8))
        out_image.CopyInformation(image)
        sitk.WriteImage(out_image, str(out_dir / pred_path.name))

        rows.append({"case_id": case_id, **stats})

    summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, index=False)
    cases_touched = int((summary["components_removed"] > 0).sum())
    print(f"Filtered {len(summary)} cases (min_voxels={args.min_voxels}, connectivity={args.connectivity}) -> {out_dir}")
    print(f"{cases_touched}/{len(summary)} cases had at least one component removed")
    print(f"Total voxels removed: {int((summary['voxels_before'] - summary['voxels_after']).sum())}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
