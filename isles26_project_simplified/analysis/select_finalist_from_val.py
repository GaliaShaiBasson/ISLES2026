#!/usr/bin/env python3
"""Rank trained conditions on val ONLY, to pick a short finalist list before
anything touches test_id/test_ood or final_holdout.

Expected input layout: `workspace/evaluation/runs/<run_id>/results_val.csv`
(one row per val case, written by `isles26.py evaluate --split val`), each
with at least `case_id, dice, hd95_mm, lesion_f1, size_bin`. `run_manifest.json`
in the same directory supplies the human-readable trainer name. Auto-discovers
every run dir under `--runs-dir` that has a `results_val.csv` -- rerun this
script unmodified as new conditions (e.g. tonight's `curriculum`/`resencm`
runs) finish; nothing here is hardcoded to today's 7 trainers.

What it produces (under `--out-dir`, default
`workspace/evaluation/finalist_selection/`):
- `summary_val.csv` -- one row per trainer: n cases, mean/std/median for
  dice, hd95_mm, lesion_f1, ranked by the primary selection rule.
- `summary_by_size_bin_val.csv` -- same, faceted by small/medium/large.
- `pairwise_vs_top_val.csv` -- paired bootstrap comparison of every other
  trainer against the current #1, on the primary metric. Paired (not just
  two independent means) because every run here was scored against the
  identical 119 val cases -- confirmed via case_id set equality before any
  stats are computed; a mismatch is treated as an error, not silently
  downgraded to an unpaired comparison, since silently switching statistical
  tests would hide the exact thing this script exists to catch.
- Printed ranked table + a plain-language flag for whether the "winner" is
  statistically distinguishable from the runner-up, or a near-tie.

Non-obvious rationale: default primary selection rule is lowest mean
hd95_mm, dice as tiebreaker -- deliberately matching the convention already
used by `postprocess_predictions.py`'s grid search (see PROJECT_PLAN.md /
CLAUDE.md), so model selection and post-processing selection apply the same
rule rather than silently picking whichever metric favors a preferred
model. Override with `--primary-metric dice` if a different rule is
intended for a given call. This script deliberately never reads
`results_test.csv` -- val is for choosing among conditions; test_id/
test_ood/final_holdout must stay untouched by this step (see CLAUDE.md
"Split finalized" and the finalist-selection discussion in project chat).

Usage:
    python analysis/select_finalist_from_val.py
    python analysis/select_finalist_from_val.py --primary-metric dice --top-n 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["dice", "hd95_mm", "lesion_f1"]
LOWER_IS_BETTER = {"hd95_mm"}
# raw-metric name -> summary-column prefix (results_val.csv's hd95_mm collides
# with the "_mean" suffix pattern otherwise, e.g. hd95_mm_mean vs. hd95_mean).
SUMMARY_PREFIX = {"dice": "dice", "hd95_mm": "hd95", "lesion_f1": "lesion_f1"}


def discover_runs(runs_dir: Path, filename: str = "results_val.csv") -> dict[str, Path]:
    """Map a human-readable trainer name -> its `filename` path (default
    `results_val.csv`; pass `filename="results_test.csv"` to discover
    already-scored held-out results instead -- e.g. `ensembling/
    ensemble_test.py` reuses this to read existing single-model test_id/
    test_ood scores for comparison, without recomputing them).

    Keyed by trainer name *and* plans identifier, not trainer name alone --
    two runs can share a trainer class (e.g. `nnUNetTrainerBaseline_500epochs`
    trained under the default plans vs. under `nnUNetResEncUNetMPlans`) while
    being genuinely different models. Keying by trainer name only would make
    the second overwrite the first in this dict silently, dropping it from
    every downstream comparison with no error. Non-default plans get a
    `(plans_name)` suffix; the default (`plans` missing/None/"nnUNetPlans")
    keeps the plain trainer name so existing output/CSVs stay unchanged."""
    out = {}
    for csv_path in sorted(runs_dir.glob(f"*/{filename}")):
        run_dir = csv_path.parent
        manifest_path = run_dir / "run_manifest.json"
        name = run_dir.name
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                name = manifest.get("trainer", name)
                plans = manifest.get("plans")
                if plans and plans != "nnUNetPlans":
                    name = f"{name} ({plans})"
            except (json.JSONDecodeError, OSError):
                pass
        if name in out:
            raise SystemExit(
                f"discover_runs: duplicate trainer+plans key '{name}' from both "
                f"{out[name].parent} and {run_dir} -- refusing to silently drop one."
            )
        out[name] = csv_path
    return out


def load_all(runs: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for trainer, path in runs.items():
        df = pd.read_csv(path)
        missing = [c for c in ["case_id", "dice", "hd95_mm", "lesion_f1", "size_bin"] if c not in df.columns]
        if missing:
            raise SystemExit(f"{path}: missing expected column(s) {missing}")
        df = df.copy()
        df["trainer"] = trainer
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def check_paired(df: pd.DataFrame) -> list[str]:
    """Confirm every trainer was scored against the identical val case set.
    Returns the common case_id list (sorted) to align on; raises loudly if
    trainers don't actually share a case set, rather than silently falling
    back to an unpaired comparison."""
    by_trainer = {t: set(g["case_id"]) for t, g in df.groupby("trainer")}
    sets = list(by_trainer.values())
    common = set.intersection(*sets)
    union = set.union(*sets)
    if common != union:
        mismatched = {t: sorted(s - common) for t, s in by_trainer.items() if s - common}
        raise SystemExit(
            "val case sets differ across trainers -- paired comparison would be invalid.\n"
            f"common={len(common)} union={len(union)}\nmismatched case_ids per trainer: {mismatched}"
        )
    return sorted(common)


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    agg = df.groupby(group_cols).agg(
        n_cases=("case_id", "nunique"),
        dice_mean=("dice", "mean"),
        dice_std=("dice", "std"),
        dice_median=("dice", "median"),
        hd95_mean=("hd95_mm", "mean"),
        hd95_std=("hd95_mm", "std"),
        hd95_n_nan=("hd95_mm", lambda s: s.isna().sum()),
        lesion_f1_mean=("lesion_f1", "mean"),
        lesion_f1_std=("lesion_f1", "std"),
    ).reset_index()
    return agg


def rank_trainers(summary: pd.DataFrame, primary_metric: str) -> pd.DataFrame:
    primary_col = f"{SUMMARY_PREFIX[primary_metric]}_mean"
    ascending = primary_metric in LOWER_IS_BETTER
    tiebreak_metric = "dice" if primary_metric != "dice" else "hd95_mm"
    tiebreak_col = f"{SUMMARY_PREFIX[tiebreak_metric]}_mean"
    tiebreak_ascending = tiebreak_metric in LOWER_IS_BETTER
    return summary.sort_values(
        [primary_col, tiebreak_col], ascending=[ascending, tiebreak_ascending]
    ).reset_index(drop=True)


def paired_bootstrap(
    df: pd.DataFrame, common_cases: list[str], top_trainer: str, metric: str, n_boot: int, seed: int, alpha: float
) -> pd.DataFrame:
    """For every trainer != top_trainer, bootstrap the paired case-level
    difference (trainer - top_trainer) on `metric` over `common_cases`, using
    the same resampled case indices for every trainer in a given draw (a
    proper paired bootstrap, not independent resamples per trainer)."""
    pivot = df.pivot(index="case_id", columns="trainer", values=metric).loc[common_cases]
    lower_better = metric in LOWER_IS_BETTER
    rng = np.random.default_rng(seed)
    n = len(common_cases)
    rows = []
    top_vals = pivot[top_trainer].to_numpy()
    for trainer in pivot.columns:
        if trainer == top_trainer:
            continue
        vals = pivot[trainer].to_numpy()
        diffs = vals - top_vals
        valid = ~np.isnan(diffs)
        diffs = diffs[valid]
        if len(diffs) == 0:
            continue
        mean_diff = diffs.mean()
        boot_means = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, len(diffs), len(diffs))
            boot_means[i] = diffs[idx].mean()
        ci_low, ci_high = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        significant = (ci_low > 0) or (ci_high < 0)
        # "trainer beats top" means: trainer is better than top on this metric
        trainer_better = (mean_diff < 0) if lower_better else (mean_diff > 0)
        rows.append(
            {
                "trainer": trainer,
                "metric": metric,
                "mean_diff_vs_top": mean_diff,
                f"ci_low_{alpha}": ci_low,
                f"ci_high_{alpha}": ci_high,
                "significant": significant,
                "trainer_better_than_top": bool(trainer_better and significant),
                "n_paired_cases": len(diffs),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="workspace/evaluation/runs", type=Path)
    ap.add_argument("--out-dir", default="workspace/evaluation/finalist_selection", type=Path)
    ap.add_argument("--primary-metric", default="hd95_mm", choices=METRICS)
    ap.add_argument("--top-n", type=int, default=3, help="How many finalists to flag for the next (test_id/ood) stage.")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trainer-filter", default=None, help="Substring filter, e.g. 500epochs, to exclude stale/legacy runs.")
    args = ap.parse_args()

    runs = discover_runs(args.runs_dir)
    if args.trainer_filter:
        runs = {t: p for t, p in runs.items() if args.trainer_filter in t}
    if not runs:
        raise SystemExit(f"No results_val.csv found under {args.runs_dir}")

    print(f"Discovered {len(runs)} run(s) with val results:")
    for t in sorted(runs):
        print(f"  - {t}")

    df = load_all(runs)
    common_cases = check_paired(df)
    print(f"\nAll {len(runs)} trainers share the same {len(common_cases)} val cases -- paired comparison is valid.")

    overall = rank_trainers(summarize(df, ["trainer"]), args.primary_metric)
    by_size = summarize(df, ["trainer", "size_bin"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(args.out_dir / "summary_val.csv", index=False)
    by_size.to_csv(args.out_dir / "summary_by_size_bin_val.csv", index=False)

    top_trainer = overall.iloc[0]["trainer"]
    pairwise = paired_bootstrap(
        df, common_cases, top_trainer, args.primary_metric, args.n_bootstrap, args.seed, args.alpha
    )
    pairwise = pairwise.sort_values("mean_diff_vs_top", ascending=(args.primary_metric not in LOWER_IS_BETTER))
    pairwise.to_csv(args.out_dir / "pairwise_vs_top_val.csv", index=False)

    print(f"\n=== Ranked on val by {args.primary_metric} (primary), dice/hd95 as tiebreak ===")
    cols = ["trainer", "n_cases", "dice_mean", "dice_std", "hd95_mean", "hd95_std", "lesion_f1_mean"]
    print(overall[cols].to_string(index=False))

    print(f"\n=== Paired bootstrap vs. top ({top_trainer}), {args.n_bootstrap} resamples, alpha={args.alpha} ===")
    if pairwise.empty:
        print("(only one trainer -- nothing to compare)")
    else:
        print(pairwise.to_string(index=False))
        indistinguishable = pairwise[~pairwise["significant"]]["trainer"].tolist()
        if indistinguishable:
            print(
                f"\nNOT statistically distinguishable from the top trainer on {args.primary_metric}: "
                f"{indistinguishable} -- treat as ties with '{top_trainer}', not as clear losses."
            )
        beats_top = pairwise[pairwise["trainer_better_than_top"]]["trainer"].tolist()
        if beats_top:
            print(
                f"\nActually beats the naive top-ranked trainer with significance: {beats_top} "
                "-- ranking by mean alone missed this, re-check the primary metric choice."
            )

    finalists = overall.head(args.top_n)["trainer"].tolist()
    # near-ties with the #1 (indistinguishable pairwise) belong in the shortlist too,
    # even if they rank below top_n by raw mean.
    tie_trainers = pairwise[~pairwise["significant"]]["trainer"].tolist() if not pairwise.empty else []
    finalist_set = list(dict.fromkeys(finalists + [top_trainer] + tie_trainers))
    print(f"\n=== Suggested finalist shortlist (top {args.top_n} + any near-ties with #1) ===")
    for t in finalist_set:
        print(f"  - {t}")
    print(
        "\nNext step per project convention: this shortlist -- not the full trainer list -- "
        "gets ONE evaluation round on test_id/test_ood to make the final pick. "
        "final_holdout stays untouched until that pick is frozen."
    )

    print(f"\nWrote: {args.out_dir / 'summary_val.csv'}")
    print(f"Wrote: {args.out_dir / 'summary_by_size_bin_val.csv'}")
    print(f"Wrote: {args.out_dir / 'pairwise_vs_top_val.csv'}")


if __name__ == "__main__":
    main()
