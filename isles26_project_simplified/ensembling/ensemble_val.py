#!/usr/bin/env python3
"""Score real softmax-averaged ensemble candidates on val -- not a proxy
(the per-case-Dice correlation/PCA in analysis/), the actual averaged
prediction, evaluated the same way every single model was.

Expected input: `predVal_prob/` directories written by
`export_val_probabilities.sh` under each shortlisted trainer's normal
nnU-Net results path (`.../fold_0/predVal_prob/*.npz` + `.pkl`) -- run that
script first if they don't exist yet; this script never launches inference
itself. Also reads `workspace/evaluation/runs/*/results_val.csv` (via
`select_finalist_from_val`'s loader) for the best single-model val scores
to compare ensembles against, and `--gt-dir` (default
`nnUNet_raw/Dataset002_ATLAS/labelsTr`) for ground truth.

What it produces (under `--out-dir`, default
`workspace/evaluation/finalist_selection/`):
- `voxel_probability_correlation.png`/`.csv` -- per-voxel Pearson r of the
  foreground-probability channel between every pair of shortlisted
  trainers, averaged over the shared val cases. A finer-grained
  complement to the per-case Dice correlation already computed in
  `analysis/plot_finalist_selection.py`: two trainers can land the same
  Dice on a case while disagreeing heavily on which voxels they're unsure
  about, which is exactly the disagreement ensembling exploits and a
  case-level Dice correlation can't see.
- `ensemble_summary_val.csv` -- one row per evaluated ensemble combo
  (mean Dice/HD95/lesion-F1 + paired-bootstrap comparison against the
  best single model on the same val cases) -- the real ensemble-vs-single
  decision, not inferred from a proxy.

Method: averages softmax probabilities voxel-wise via nnU-Net's own
`nnunetv2.ensembling.ensemble.ensemble_folders` (not reimplemented here --
it correctly handles the crop/resample-back-to-original-space bookkeeping
via each case's saved .pkl properties, which a hand-rolled average would
have to duplicate exactly to be trustworthy), then scores the resulting
masks with `evaluation/compute_metrics.py:evaluate_case`, identically to
every other prediction in this project.

Which combos are tried: by default, every pair among the shortlist (cheap:
5 trainers -> 10 pairs) plus the full shortlist averaged together --
override with `--combos "A+B,C+D+E"` (trainer short names, '+'-joined,
comma-separated) to target specific groups, e.g. ones flagged as
complementary by the voxel-correlation output above.

Usage:
    python ensembling/ensemble_val.py
    python ensembling/ensemble_val.py --combos "FocalTversky+LesionAwareSamplingPow"
"""
from __future__ import annotations

import argparse
import shutil
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from select_finalist_from_val import discover_runs, load_all, check_paired  # noqa: E402
from plot_finalist_selection import short_name, trainer_colors, BLUE_SEQUENTIAL  # noqa: E402
from compute_metrics import evaluate_case  # noqa: E402

from nnunetv2.ensembling.ensemble import ensemble_folders  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap

NNUNET_RAW = Path("/home/galia/ISLES2026/nnUNet_raw")
NNUNET_PREPROCESSED = Path("/home/galia/ISLES2026/nnUNet_preprocessed")
NNUNET_RESULTS = Path("/home/galia/ISLES2026/nnUNet_results")
DATASET_NAME = "Dataset002_ATLAS"

DEFAULT_TRAINERS = [
    "nnUNetTrainerWideAugBaseline_500epochs",
    "nnUNetTrainerFocalTversky_500epochs",
    "nnUNetTrainerTverskyMild_500epochs",
    "nnUNetTrainerLesionAwareSamplingPow_500epochs_full",
    "nnUNetTrainerLesionAwareSamplingPowCurriculum_500epochs_full",
]


def predval_dir(trainer: str) -> Path:
    return NNUNET_RESULTS / DATASET_NAME / f"{trainer}__nnUNetPlans__3d_fullres" / "fold_0" / "predVal_prob"


def check_predval_dirs(trainers: list[str]) -> dict[str, Path]:
    missing = [t for t in trainers if not predval_dir(t).exists()]
    if missing:
        raise SystemExit(
            "Missing predVal_prob/ for: " + ", ".join(missing) +
            "\nRun ensembling/export_val_probabilities.sh for these trainers first."
        )
    return {t: predval_dir(t) for t in trainers}


def voxel_probability_correlation(dirs: dict[str, Path], case_ids: list[str]) -> pd.DataFrame:
    trainers = list(dirs)
    sums = {(a, b): [] for a, b in combinations(trainers, 2)}
    for case_id in case_ids:
        arrays = {}
        shape = None
        for t in trainers:
            npz_path = dirs[t] / f"{case_id}.npz"
            if not npz_path.exists():
                break
            arr = np.load(npz_path)["probabilities"][1]  # foreground channel
            if shape is None:
                shape = arr.shape
            elif arr.shape != shape:
                raise SystemExit(
                    f"{case_id}: probability shape mismatch between trainers "
                    f"({t} has {arr.shape}, expected {shape}) -- refusing to correlate "
                    "mismatched grids rather than silently reshaping."
                )
            arrays[t] = arr.ravel()
        if len(arrays) != len(trainers):
            continue  # case missing for some trainer, skip rather than guess
        for a, b in combinations(trainers, 2):
            r, _ = pearsonr(arrays[a], arrays[b])
            sums[(a, b)].append(r)

    trainers_sorted = trainers
    corr = pd.DataFrame(np.eye(len(trainers_sorted)), index=trainers_sorted, columns=trainers_sorted)
    for (a, b), vals in sums.items():
        mean_r = float(np.mean(vals)) if vals else np.nan
        corr.loc[a, b] = mean_r
        corr.loc[b, a] = mean_r
    return corr


def plot_voxel_correlation(corr: pd.DataFrame, out_dir: Path) -> None:
    labels = [short_name(t) for t in corr.columns]
    vmin = max(0.0, float(np.nanmin(corr.values[~np.eye(len(corr), dtype=bool)])) - 0.05)
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_SEQUENTIAL)

    fig, ax = plt.subplots(figsize=(0.75 * len(labels) + 2, 0.75 * len(labels) + 2))
    im = ax.imshow(corr.values, cmap=cmap, vmin=vmin, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.values[i, j]
            text_color = "#ffffff" if val > (vmin + 1.0) / 2 else "#0b0b0b"
            ax.text(j, i, f"{val:.2f}" if not np.isnan(val) else "n/a", ha="center", va="center",
                    fontsize=7, color=text_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r, per-voxel foreground probability")
    ax.set_title("Per-voxel probability correlation between trainers (val)\nlower = more complementary confidence, finer than the per-case Dice view", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "voxel_probability_correlation.png", dpi=180)
    plt.close(fig)
    corr.to_csv(out_dir / "voxel_probability_correlation.csv")


def default_combos(trainers: list[str]) -> list[tuple[str, ...]]:
    combos = list(combinations(trainers, 2))
    combos.append(tuple(trainers))
    return combos


def score_combo(dirs: dict[str, Path], combo: tuple[str, ...], gt_dir: Path, work_dir: Path) -> pd.DataFrame:
    input_folders = [str(dirs[t]) for t in combo]
    out_folder = work_dir / ("+".join(short_name(t) for t in combo))
    if out_folder.exists():
        shutil.rmtree(out_folder)
    dataset_json = NNUNET_RAW / DATASET_NAME / "dataset.json"
    plans_json = NNUNET_PREPROCESSED / DATASET_NAME / "nnUNetPlans.json"
    ensemble_folders(
        input_folders, str(out_folder), save_merged_probabilities=False,
        dataset_json_file_or_dict=str(dataset_json), plans_json_file_or_dict=str(plans_json),
    )
    rows = []
    for pred_path in sorted(out_folder.glob("*.nii.gz")):
        case_id = pred_path.stem.replace(".nii", "")
        gt_path = gt_dir / f"{case_id}.nii.gz"
        if not gt_path.exists():
            continue
        metrics = evaluate_case(pred_path, gt_path)
        metrics["case_id"] = case_id
        rows.append(metrics)
    return pd.DataFrame(rows)


def paired_bootstrap_vs_top(ensemble_scores: pd.Series, top_scores: pd.Series, n_boot: int, seed: int, alpha: float, lower_better: bool) -> dict:
    common = ensemble_scores.index.intersection(top_scores.index)
    diffs = (ensemble_scores.loc[common] - top_scores.loc[common]).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    boot_means = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(n_boot)])
    ci_low, ci_high = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    significant = (ci_low > 0) or (ci_high < 0)
    mean_diff = diffs.mean()
    better = (mean_diff < 0) if lower_better else (mean_diff > 0)
    return {
        "mean_diff_vs_top_single": mean_diff, "ci_low": ci_low, "ci_high": ci_high,
        "significant": significant, "ensemble_better_than_top_single": bool(significant and better),
        "n_paired_cases": len(diffs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trainers", nargs="+", default=DEFAULT_TRAINERS)
    ap.add_argument("--runs-dir", default="workspace/evaluation/runs", type=Path)
    ap.add_argument("--gt-dir", default=str(NNUNET_RAW / DATASET_NAME / "labelsTr"), type=Path)
    ap.add_argument("--out-dir", default="workspace/evaluation/finalist_selection", type=Path)
    ap.add_argument("--work-dir", default="workspace/evaluation/ensemble_tmp", type=Path)
    ap.add_argument("--combos", default=None, help='e.g. "FocalTversky+TverskyMild,WideAugBaseline+FocalTversky"')
    ap.add_argument("--primary-metric", default="hd95_mm", choices=["dice", "hd95_mm", "lesion_f1"])
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dirs = check_predval_dirs(args.trainers)

    runs = discover_runs(args.runs_dir)
    runs = {t: p for t, p in runs.items() if t in args.trainers}
    single_df = load_all(runs)
    check_paired(single_df)
    case_ids = sorted(single_df["case_id"].unique())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing per-voxel probability correlation over {len(case_ids)} shared val cases...")
    corr = voxel_probability_correlation(dirs, case_ids)
    plot_voxel_correlation(corr, args.out_dir)
    off_diag = corr.where(~np.eye(len(corr), dtype=bool))
    print(f"Wrote: {args.out_dir / 'voxel_probability_correlation.png'}")
    print(f"Lowest voxel-probability correlation pair (most complementary): "
          f"{off_diag.stack().idxmin()} = {off_diag.stack().min():.3f}")

    if args.combos:
        by_short = {short_name(t): t for t in args.trainers}
        combos = [tuple(by_short[x] for x in group.split("+")) for group in args.combos.split(",")]
    else:
        combos = default_combos(args.trainers)

    lower_better = args.primary_metric in {"hd95_mm"}
    top_trainer = single_df.groupby("trainer")[args.primary_metric].mean().sort_values(ascending=lower_better).index[0]
    top_scores = single_df[single_df["trainer"] == top_trainer].set_index("case_id")[args.primary_metric]
    print(f"\nBest single model on val ({args.primary_metric}): {short_name(top_trainer)}")

    rows = []
    for combo in combos:
        label = "+".join(short_name(t) for t in combo)
        print(f"\nScoring ensemble: {label} ...")
        scored = score_combo(dirs, combo, args.gt_dir, args.work_dir)
        if scored.empty:
            print(f"  [skip] no cases scored for {label}")
            continue
        scored = scored.set_index("case_id")
        stats = paired_bootstrap_vs_top(
            scored[args.primary_metric], top_scores, args.n_bootstrap, args.seed, args.alpha, lower_better
        )
        rows.append({
            "combo": label, "n_cases": len(scored),
            "dice_mean": scored["dice"].mean(), "hd95_mean": scored["hd95_mm"].mean(),
            "lesion_f1_mean": scored["lesion_f1"].mean(), **stats,
        })

    summary = pd.DataFrame(rows).sort_values("mean_diff_vs_top_single", ascending=lower_better)
    summary.to_csv(args.out_dir / "ensemble_summary_val.csv", index=False)
    print(f"\n=== Ensemble candidates vs. best single model ({short_name(top_trainer)}) on val ===")
    print(summary.to_string(index=False))

    winners = summary[summary["ensemble_better_than_top_single"]]
    if not winners.empty:
        print(f"\nEnsemble(s) significantly BETTER than the best single model on {args.primary_metric}: "
              f"{winners['combo'].tolist()}")
    else:
        print(f"\nNo ensemble combo significantly beats {short_name(top_trainer)} alone on {args.primary_metric} "
              "-- the single model may be the simpler, equally-good choice.")

    print(f"\nWrote: {args.out_dir / 'ensemble_summary_val.csv'}")
    print(f"Wrote: {args.out_dir / 'voxel_probability_correlation.csv'}")


if __name__ == "__main__":
    main()
