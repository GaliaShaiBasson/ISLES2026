#!/usr/bin/env python3
"""ne / voxel-
level TP-FP-FN error map, for a handful of representative and failure cases.

Expected input layout: nnU-Net's own raw-dataset naming --
``--image-dir/<case_id>_0000.nii.gz`` (single-channel T1w), ``--gt-dir/
<case_id>.nii.gz``, ``--pred-dir/<case_id>.nii.gz`` (binary/argmax masks --
NOT softmax probability .npz), all on the same 1mm isotropic grid. Case
selection is automatic from ``--results-csv`` (a compute_metrics.py output,
needs ``case_id``, ``dice``, ``gt_vol_mm3``, ``pred_vol_mm3`` columns) unless
``--case-ids``/``--case-labels`` are passed explicitly.

Produces a single ``overlay_grid_<model-label>[_by_size_bin].png`` under
``--out-dir`` -- one figure, one row per selected case, 3 subplot columns
(image+GT contour, image+prediction contour, image+TP/FP/FN color map).
Deliberately one dense multi-subplot file rather than N+1 files (a separate
PNG per case plus the grid) -- easier to browse and to drop straight into
the report as a single figure.

Non-obvious rationale:
- Auto-selected categories are ``best`` (highest Dice -- a clean, mostly-TP
  success case), ``fn_dominant`` (real lesion, model predicts far less than
  the GT volume -- a missed-lesion failure), and ``fp_dominant`` (model
  predicts far more than the (small/near-empty) GT volume -- a phantom-lesion
  failure). There is deliberately no auto-selected "TN" case: this project's
  test_id/test_ood split has zero empty-lesion cases (all 3 dataset-wide
  landed in train, see CLAUDE.md's "Split finalized" entry) -- a whole-case
  true negative doesn't exist to show on held-out test. Every rendered
  slice's background is itself a true-negative region (correctly-quiet, not
  colored in the error map), which is the only place voxel-level TN actually
  shows up here.
- The rendered axial slice is picked per-case as the z-index with the most
  (GT union prediction) foreground area -- not a fixed slice index -- since
  lesion location varies case to case and a fixed index would often miss the
  lesion entirely.
- ATLAS is raw/native-space, multi-site data (see CLAUDE.md "Data source:
  ATLAS R3.0") -- voxel spacing is NOT uniform across cases (e.g. one real
  case here is 1x3x1mm, thick-slice, vs another's 1x1x1mm isotropic). Each
  panel is rendered with imshow's `aspect` set from that case's own
  spacing -- without it, a thick-slice case's slice silently renders
  squashed/undersized relative to isotropic cases, which looks like a bug
  but is actually a display bug (fixed here), not a data problem.
- Read-only against every input directory; only ``--out-dir`` is written.

Usage:
    python analysis/save_qualitative_overlays.py \\
        --image-dir /home/galia/ISLES2026/nnUNet_raw/Dataset002_ATLAS/imagesTs \\
        --gt-dir /home/galia/ISLES2026/nnUNet_raw/Dataset002_ATLAS/labelsTs \\
        --pred-dir /path/to/fold_0/predTs \\
        --results-csv workspace/results/runs/<run_id>/results_test.csv \\
        --model-label wideaug_resencm \\
        --out-dir workspace/figures/qualitative_overlays
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

CATEGORY_TITLES = {
    "best": "Best Dice (mostly TP)",
    "fn_dominant": "False-negative dominated (missed lesion)",
    "fp_dominant": "False-positive dominated (phantom lesion)",
    "small_best": "Small lesion -- best Dice in bin",
    "small_average": "Small lesion -- closest to mean Dice in bin",
    "small_worst": "Small lesion -- worst Dice in bin",
    "medium_best": "Medium lesion -- best Dice in bin",
    "medium_average": "Medium lesion -- closest to mean Dice in bin",
    "medium_worst": "Medium lesion -- worst Dice in bin",
    "large_best": "Large lesion -- best Dice in bin",
    "large_average": "Large lesion -- closest to mean Dice in bin",
    "large_worst": "Large lesion -- worst Dice in bin",
}


def load_volume(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Returns (array in (z,y,x) index order, spacing in (x,y,z) mm) --
    spacing is needed to render anisotropic-voxel cases (real ATLAS raw data,
    not every case is 1mm isotropic) at their correct physical aspect ratio."""
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), img.GetSpacing()


def select_cases(results_csv: Path) -> dict[str, str]:
    df = pd.read_csv(results_csv)
    missing = [c for c in ("case_id", "dice", "gt_vol_mm3", "pred_vol_mm3") if c not in df.columns]
    if missing:
        raise SystemExit(f"{results_csv}: missing expected column(s) {missing}")

    selected: dict[str, str] = {}
    best = df.sort_values("dice", ascending=False).iloc[0]
    selected["best"] = best["case_id"]

    fn_dom = df[(df["gt_vol_mm3"] > 1000) & (df["pred_vol_mm3"] < df["gt_vol_mm3"] * 0.3)]
    if not fn_dom.empty:
        selected["fn_dominant"] = fn_dom.sort_values("dice").iloc[0]["case_id"]
    else:
        print("[warn] no false-negative-dominated case found (gt>1000mm3, pred<30% of gt) -- skipping")

    fp_dom = df[(df["pred_vol_mm3"] > df["gt_vol_mm3"] * 3) & (df["pred_vol_mm3"] > 200)]
    if not fp_dom.empty:
        selected["fp_dominant"] = fp_dom.sort_values("dice").iloc[0]["case_id"]
    else:
        print("[warn] no false-positive-dominated case found (pred>3x gt, pred>200mm3) -- skipping")

    if (df["gt_vol_mm3"] <= 0).any():
        print("[note] a gt_empty case exists in this split but is not auto-selected -- pass "
              "--case-ids/--case-labels explicitly to add a true-negative example.")
    return selected


def select_cases_by_size_bin(results_csv: Path) -> dict[str, str]:
    """Best-, average-, and worst-Dice case per lesion-size bin (small/medium/
    large) -- up to nine cases total, so each size shows a clean success, a
    real failure, AND what's actually typical for that bin (not just the two
    extremes, which alone can misrepresent a bin whose mean Dice is mediocre
    even though its best case looks great). "Average" is the case whose own
    Dice is closest to the bin's mean Dice -- an actual case, not a synthetic
    blend, so it renders like any other row -- e.g. if small-lesion mean Dice
    is 0.5, the small_average row is a real case scoring closest to 0.5.
    Mirrors analysis/plot_results.py:save_by_size's bin ordering/grouping."""
    df = pd.read_csv(results_csv)
    missing = [c for c in ("case_id", "dice", "size_bin") if c not in df.columns]
    if missing:
        raise SystemExit(f"{results_csv}: missing expected column(s) {missing}")

    selected: dict[str, str] = {}
    for size_bin in ("small", "medium", "large"):
        bin_df = df[df["size_bin"] == size_bin]
        if bin_df.empty:
            print(f"[warn] no cases found for size_bin={size_bin!r} -- skipping")
            continue
        ranked = bin_df.sort_values("dice", ascending=False)
        best_id = ranked.iloc[0]["case_id"]
        worst_id = ranked.iloc[-1]["case_id"]
        has_worst = len(ranked) > 1 and worst_id != best_id
        if not has_worst:
            print(f"[warn] size_bin={size_bin!r} has only 1 case -- no separate worst example")

        mean_dice = bin_df["dice"].mean()
        exclude = {best_id, worst_id} if has_worst else {best_id}
        remaining = bin_df[~bin_df["case_id"].isin(exclude)]
        pool = remaining if not remaining.empty else bin_df
        avg_id = pool.iloc[(pool["dice"] - mean_dice).abs().argsort().iloc[0]]["case_id"]
        print(f"[info] size_bin={size_bin!r} mean dice={mean_dice:.3f}, "
              f"closest case={avg_id} (dice={pool.set_index('case_id').loc[avg_id, 'dice']:.3f})")

        # best -> average -> worst, so each bin's rows read as a clean gradient
        selected[f"{size_bin}_best"] = best_id
        selected[f"{size_bin}_average"] = avg_id
        if has_worst:
            selected[f"{size_bin}_worst"] = worst_id
    return selected


def best_slice(gt: np.ndarray, pred: np.ndarray) -> int:
    union = (gt > 0) | (pred > 0)
    areas = union.sum(axis=(1, 2))
    if areas.max() == 0:
        return gt.shape[0] // 2  # no foreground anywhere -- fall back to mid-volume
    return int(np.argmax(areas))


def render_grid(image_dir: Path, gt_dir: Path, pred_dir: Path, cases: dict[str, str],
                 dice_by_case: dict[str, float], model_label: str, out_path: Path) -> None:
    """One figure, one row per case -- the only rendering path (no more
    one-PNG-per-case): fewer files, denser subplots, same content."""
    n = len(cases)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(13.5, 4.5 * n), squeeze=False)
    for row, (category, case_id) in enumerate(cases.items()):
        image, spacing = load_volume(image_dir / f"{case_id}_0000.nii.gz")
        gt, _ = load_volume(gt_dir / f"{case_id}.nii.gz")
        pred, _ = load_volume(pred_dir / f"{case_id}.nii.gz")
        gt = gt > 0
        pred = pred > 0
        if image.shape != gt.shape or gt.shape != pred.shape:
            raise ValueError(f"{case_id}: shape mismatch image={image.shape} gt={gt.shape} pred={pred.shape}")

        z = best_slice(gt, pred)
        img_slice, gt_slice, pred_slice = image[z], gt[z], pred[z]
        tp = gt_slice & pred_slice
        fp = pred_slice & ~gt_slice
        fn = gt_slice & ~pred_slice
        vmin, vmax = np.percentile(image[image > 0], [1, 99]) if (image > 0).any() else (image.min(), image.max())

        # spacing is (x, y, z) mm; a slice on array axis 0 (z) has rows=y, cols=x --
        # imshow's `aspect` must be spacing_y/spacing_x or an anisotropic-voxel case
        # (real ATLAS raw data, e.g. 1x3x1mm thick-slice acquisitions) renders
        # squashed relative to isotropic cases, purely a display artifact.
        aspect = spacing[1] / spacing[0]

        ax0, ax1, ax2 = axes[row]
        for ax in (ax0, ax1, ax2):
            ax.imshow(img_slice, cmap="gray", vmin=vmin, vmax=vmax, aspect=aspect)
            ax.set_xticks([])
            ax.set_yticks([])
        ax0.contour(gt_slice, levels=[0.5], colors=["#2ecc71"], linewidths=1.5)
        ax1.contour(pred_slice, levels=[0.5], colors=["#3498db"], linewidths=1.5)
        error_rgba = np.zeros((*tp.shape, 4))
        error_rgba[tp] = (0.18, 0.80, 0.44, 0.55)
        error_rgba[fp] = (0.90, 0.30, 0.24, 0.55)
        error_rgba[fn] = (0.20, 0.60, 0.86, 0.55)
        ax2.imshow(error_rgba, aspect=aspect)

        dice = dice_by_case.get(case_id, float("nan"))
        ax0.set_ylabel(f"{CATEGORY_TITLES.get(category, category)}\n{case_id} (Dice={dice:.3f})",
                        fontsize=9, rotation=0, ha="right", va="center", labelpad=60)
        if row == 0:
            ax0.set_title("Ground truth", fontsize=10)
            ax1.set_title(f"Prediction ({model_label})", fontsize=10)
            ax2.set_title("TP (green) / FP (red) / FN (blue)", fontsize=10)

    fig.suptitle(f"Qualitative overlays -- {model_label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--results-csv", type=Path, help="Used for auto case selection (needs case_id/dice/"
                         "gt_vol_mm3/pred_vol_mm3). Ignored if --case-ids is given.")
    parser.add_argument("--case-ids", nargs="+", help="Explicit case_id list, overrides auto-selection.")
    parser.add_argument("--case-labels", nargs="+", help="Category label per --case-ids entry (same length).")
    parser.add_argument("--by-size-bin", action="store_true",
                         help="Select best+worst Dice case per lesion-size bin (small/medium/large, 6 cases "
                              "total) instead of the default best/fn_dominant/fp_dominant selection. Needs "
                              "a size_bin column in --results-csv.")
    parser.add_argument("--model-label", required=True, help="Used in titles/filenames, e.g. 'wideaug_resencm'.")
    parser.add_argument("--out-dir", default="workspace/figures/qualitative_overlays", type=Path)
    args = parser.parse_args()

    for label, path in (("--image-dir", args.image_dir), ("--gt-dir", args.gt_dir), ("--pred-dir", args.pred_dir)):
        if not path.is_dir():
            parser.error(f"{label} not found: {path}")

    if args.case_ids:
        if not args.case_labels or len(args.case_labels) != len(args.case_ids):
            parser.error("--case-ids requires --case-labels of the same length")
        cases = dict(zip(args.case_labels, args.case_ids))
        dice_by_case = {}
        if args.results_csv and args.results_csv.is_file():
            df = pd.read_csv(args.results_csv)
            dice_by_case = dict(zip(df["case_id"], df["dice"]))
    else:
        if not args.results_csv:
            parser.error("either --case-ids/--case-labels or --results-csv is required")
        cases = select_cases_by_size_bin(args.results_csv) if args.by_size_bin else select_cases(args.results_csv)
        dice_by_case = dict(zip(pd.read_csv(args.results_csv)["case_id"], pd.read_csv(args.results_csv)["dice"]))

    if not cases:
        raise SystemExit("No cases selected -- nothing to render.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for category, case_id in cases.items():
        for label, path in (
            ("image", args.image_dir / f"{case_id}_0000.nii.gz"),
            ("gt", args.gt_dir / f"{case_id}.nii.gz"),
            ("pred", args.pred_dir / f"{case_id}.nii.gz"),
        ):
            if not path.is_file():
                raise SystemExit(f"{case_id}: {label} file not found: {path}")
        print(f"[{category}] {case_id} (Dice={dice_by_case.get(case_id, float('nan')):.3f})")

    if args.by_size_bin:
        # One figure per size bin (average -> best -> worst rows) rather than
        # one 9-row figure -- each bin's own success/typical/failure story is
        # easier to read as its own compact 3-row plot.
        for size_bin in ("small", "medium", "large"):
            bin_cases = {
                category: case_id for category, case_id in cases.items()
                if category.startswith(f"{size_bin}_")
            }
            row_order = [f"{size_bin}_average", f"{size_bin}_best", f"{size_bin}_worst"]
            bin_cases = {k: bin_cases[k] for k in row_order if k in bin_cases}
            if not bin_cases:
                continue
            grid_path = args.out_dir / f"overlay_grid_{args.model_label}_{size_bin}.png"
            render_grid(args.image_dir, args.gt_dir, args.pred_dir, bin_cases, dice_by_case,
                        args.model_label, grid_path)
            print(f"Wrote {grid_path}")
    else:
        grid_path = args.out_dir / f"overlay_grid_{args.model_label}.png"
        render_grid(args.image_dir, args.gt_dir, args.pred_dir, cases, dice_by_case, args.model_label, grid_path)
        print(f"Wrote {grid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
