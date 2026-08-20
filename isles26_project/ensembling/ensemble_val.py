#!/usr/bin/env python3
"""Score real softmax-averaged ensemble candidates on val -- not a proxy
(the per-case-Dice correlation/PCA, or the voxel-probability-correlation/PCA
in `analysis/finalist_selection/`), the actual averaged prediction, evaluated
the same way every single model was.

This script only builds and scores real ensembles -- it has no correlation
computation and no plotting code. Selecting *which* trainers to combo over
in the first place is `analysis/finalist_selection/voxel_probability_
correlation.py` + `pca_probability_redundancy.py`'s job (read-only, over the
full candidate set); this script is the next, deliberate step: given a
chosen combo, actually build it and score it for real. Kept separate on
purpose -- this script does real, sometimes slow CPU work (softmax
averaging + affine-aware resampling + metric computation) that `analysis/`
scripts by convention never do.

Expected input: real val-set probabilities per shortlisted trainer, resolved
by `predval_dirs.check_predval_dirs()` -- prefers `predVal_prob/` (written by
`export_val_probabilities.sh`/its queue-script variants) but falls back to
nnU-Net's own automatic `fold_0/validation/*.npz` if that's where the real
probabilities actually are (e.g. standard-plans `nnUNetTrainerBaseline_
500epochs`, which was never in `export_val_probabilities.sh`'s trainer list
but already has real val-set probabilities from an earlier `--val --npz`
validate-only pass -- see PROJECT_PLAN.md "Finalist-selection" entry).
Trainer identifiers may include a plans suffix, `"trainer (PlansName)"`
(matching `select_finalist_from_val.discover_runs`'s key format exactly),
to address a non-default-plans run such as `nnUNetResEncUNetMPlans` -- run
`export_val_probabilities.sh` first if neither location has `.npz` files yet
for a given trainer; this script never launches inference itself. Also
reads `workspace/results/runs/*/results_val.csv` (via
`select_finalist_from_val`'s loader) for the best single-model val scores
to compare ensembles against, and `--gt-dir` (default
`nnUNet_raw/Dataset002_ATLAS/labelsTr`) for ground truth.

What it produces (under `--out-dir`, default
`workspace/results/finalist_selection/`):
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
4 trainers -> 6 pairs) plus the full shortlist averaged together --
override with `--combos "A+B,C+D+E"` (trainer short names, '+'-joined,
comma-separated) to target specific groups, e.g. ones flagged as
complementary by `analysis/finalist_selection/voxel_probability_
correlation.py`'s output. That selection script should run over the full
candidate set, not this one -- scoring O(n^2) real ensemble combos for many
trainers is exactly the expensive step worth avoiding until the candidate
list is already narrowed by correlation/PCA.

Once a combo is chosen here, see `ensembling/ensemble_test.py` for the
one-time, deliberate held-out (test_id/test_ood) scoring step -- this
script only ever touches val.

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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis" / "finalist_selection"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from select_finalist_from_val import discover_runs, load_all, check_paired  # noqa: E402
from plot_finalist_selection import short_name  # noqa: E402
from predval_dirs import check_predval_dirs  # noqa: E402
from compute_metrics import evaluate_case  # noqa: E402

from nnunetv2.ensembling.ensemble import ensemble_folders  # noqa: E402

NNUNET_RAW = Path("/home/galia/ISLES2026/nnUNet_raw")
NNUNET_PREPROCESSED = Path("/home/galia/ISLES2026/nnUNet_preprocessed")
DATASET_NAME = "Dataset002_ATLAS"

# Updated 2026-08-20 to match the finalist-selection/PCA redundancy analysis
# (PROJECT_PLAN.md "Finalist-selection / ensembling-candidate analysis"): one
# representative per distinct error-pattern cluster, not the earlier
# pre-ResEncM 5. `(nnUNetResEncUNetMPlans)` uses the same key format
# `select_finalist_from_val.discover_runs` produces.
#
# Re-derived same day after the primary metric switched hd95_mm -> dice (see
# CLAUDE.md "Primary metric switched to Dice"): the redundancy clusters'
# "best-ranked in cluster" pick flips in BOTH clusters under dice --
# WideAugBaseline (ResEncM) now beats Baseline (ResEncM) (was the reverse
# under hd95_mm), and Baseline now beats WideAugBaseline in the standard-plans
# cluster (also reversed). The 4th slot also flips: LesionAwareSamplingPow-
# Curriculum (dice 0.6173) now ranks above LesionAwareSamplingPow (dice
# 0.6127) -- consistent with the voxel-probability-correlation finding that
# Curriculum was already the more complementary of the two (see PROJECT_PLAN.md).
DEFAULT_TRAINERS = [
    "nnUNetTrainerBaseline_500epochs",
    "nnUNetTrainerWideAugBaseline_500epochs (nnUNetResEncUNetMPlans)",
    "nnUNetTrainerFocalTversky_500epochs",
    "nnUNetTrainerLesionAwareSamplingPowCurriculum_500epochs_full",
]


def default_combos(trainers: list[str]) -> list[tuple[str, ...]]:
    combos = list(combinations(trainers, 2))
    combos.append(tuple(trainers))
    return combos


def build_ensemble(dirs: dict[str, Path], combo: tuple[str, ...], work_dir: Path) -> Path:
    """Softmax-average `combo`'s probability folders into `work_dir/<label>/`
    via nnU-Net's own `ensemble_folders` (real ensembling, no scoring) --
    factored out of `score_combo` so `ensembling/ensemble_test.py` can reuse
    the exact same ensembling step but score the result with
    `evaluation/compute_metrics.py`'s CLI (manifest-merged columns: split,
    center, size_bin) instead of `score_combo`'s bare per-case loop, which
    only needs the raw metrics for the paired-bootstrap-vs-top comparison."""
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
    return out_folder


def score_combo(dirs: dict[str, Path], combo: tuple[str, ...], gt_dir: Path, work_dir: Path) -> pd.DataFrame:
    out_folder = build_ensemble(dirs, combo, work_dir)
    rows = []
    failed = 0
    for pred_path in sorted(out_folder.glob("*.nii.gz")):
        case_id = pred_path.stem.replace(".nii", "")
        gt_path = gt_dir / f"{case_id}.nii.gz"
        if not gt_path.exists():
            continue
        try:
            metrics = evaluate_case(pred_path, gt_path)
        except Exception as exc:
            # Same per-case error handling as evaluation/compute_metrics.py's own CLI --
            # a real geometry mismatch on one case (e.g. ensembling voxel-grids that
            # don't resample back identically) shouldn't crash scoring for every other
            # case in the combo. Verified against a real case: 2026-08-20,
            # WideAugBaseline+Baseline(ResEncM) crashed the whole run on exactly this
            # exception before this fix existed.
            print(f"  [error] {case_id}: {exc}")
            failed += 1
            continue
        metrics["case_id"] = case_id
        rows.append(metrics)
    if failed:
        print(f"  [warn] {failed} case(s) failed evaluation for this combo (see [error] lines above) -- excluded, not crashed on.")
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
    # Standard two-sided bootstrap p-value: the fraction of resampled means on
    # the "wrong" side of zero, doubled (two-sided) and capped at 1.0 -- direction-
    # agnostic (doesn't need lower_better), consistent with `significant` above
    # (which is also just "does the 95% CI exclude zero", regardless of direction).
    p_value = min(1.0, 2 * min((boot_means <= 0).mean(), (boot_means >= 0).mean()))
    return {
        "mean_diff_vs_top_single": mean_diff, "ci_low": ci_low, "ci_high": ci_high,
        "p_value": p_value,
        "significant": significant, "ensemble_better_than_top_single": bool(significant and better),
        "n_paired_cases": len(diffs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trainers", nargs="+", default=DEFAULT_TRAINERS)
    ap.add_argument("--runs-dir", default="workspace/results/runs", type=Path)
    ap.add_argument("--gt-dir", default=str(NNUNET_RAW / DATASET_NAME / "labelsTr"), type=Path)
    ap.add_argument("--out-dir", default="workspace/results/finalist_selection", type=Path)
    ap.add_argument("--work-dir", default="workspace/results/ensemble_tmp", type=Path)
    ap.add_argument("--combos", default=None, help='e.g. "FocalTversky+LesionAwareSamplingPow,WideAugBaseline+Baseline (ResEncM)"')
    ap.add_argument("--primary-metric", default="dice", choices=["dice", "hd95_mm", "lesion_f1"])
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


if __name__ == "__main__":
    main()
