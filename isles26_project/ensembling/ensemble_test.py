#!/usr/bin/env python3
"""One-time, deliberate held-out (test_id/test_ood) scoring for a CHOSEN
ensemble combo, plus a side-by-side comparison against 1-2 already-scored
single models. No retraining and (usually) no new prediction -- reuses
whatever's already on disk wherever possible; the only thing this script
may need to newly compute is missing `predTs_prob/` exports for the
ensemble's own members (never for the comparison single model(s), whose
`results_test.csv` already exists and is read as-is).

Expected input:
- `--combo` member trainers, real val-set probabilities not required here --
  their `predTs_prob/` (test-set softmax probabilities), resolved by
  `predtest_dir()` (plans-aware, same key format as `predval_dirs.py`, but
  no `validation/`-folder fallback: nnU-Net has no automatic equivalent for
  the test set, only an explicit `--save_probabilities` predict run). Refuses
  with the exact `nnUNetv2_predict` command to run for any member missing it
  -- never launches inference itself.
- `--compare-against` trainer(s) already evaluated on test_id/test_ood --
  their existing `workspace/results/runs/<run_id>/results_test.csv` is read
  and reused directly, never recomputed. Default: the single best val
  performer (same rule `select_finalist_from_val.py` uses).
- ground truth: `nnUNet_raw/Dataset002_ATLAS/labelsTs` (test_id + test_ood
  combined; split apart via the `split` column already in every
  `results_test.csv`, joined from `workspace/splits_dataset002/manifest.csv`).

What it produces:
- `workspace/results/runs/ensemble_<combo-label>/results_test.csv` -- same
  schema as every single-model `results_test.csv` (dice/hd95_mm/lesion_f1/
  split/center/size_bin/...), written via `evaluation/compute_metrics.py`'s
  own CLI (not reimplemented -- this is exactly what every other
  `evaluate --split test` call in this project already uses, including its
  per-case error handling and case-metadata join), so the ensemble drops
  straight into `aggregate`'s existing glob alongside every single-model run.
- `workspace/results/runs/ensemble_<combo-label>/run_manifest.json` -- notes
  the ensemble's member trainers instead of a real checkpoint fingerprint
  (there is no single checkpoint for an ensemble).
- `workspace/results/runs/ensemble_<combo-label>/report_table.csv` (+ printed) --
  one row per (breakdown, metric): overall, per split (test_id/test_ood),
  AND per size_bin (small/medium/large) -- not just split. Size_bin matters
  as much as split here: per `docs/reference/`'s "Match Architecture to
  Pathology Size", small lesions are the case a generic pipeline is most
  likely to lose signal on, so "which model wins on pooled Dice" can hide
  "which model actually wins where it's hardest." Each row shows every
  compared model's mean, plus a paired bootstrap (same method as
  `ensemble_val.py`'s val-side comparison) -- mean difference, 95% CI, and a
  two-sided p-value -- for the ensemble vs. exactly one anchor model (the
  best single performer by `--primary-metric`, always included even if not
  requested via `--compare-against`), not one p-value per model shown, to
  avoid an uncontrolled multiple-comparisons problem.

Non-obvious rationale: deliberately separate from `ensemble_val.py` (which
only ever touches val) -- test_id/test_ood should be touched once, after the
combo is already chosen on val, not iteratively while still searching for
one. This script assumes a combo has already been chosen (e.g. via
`analysis/voxel_probability_correlation.py` + `pca_probability_redundancy.py`
+ a confirming `ensemble_val.py` run) and is not meant to be run
speculatively for many candidate combos.

Usage:
    python ensembling/ensemble_test.py
    python ensembling/ensemble_test.py --combo "nnUNetTrainerWideAugBaseline_500epochs" "nnUNetTrainerFocalTversky_500epochs" --compare-against "nnUNetTrainerWideAugBaseline_500epochs"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis" / "finalist_selection"))
from select_finalist_from_val import discover_runs, load_all, check_paired, LOWER_IS_BETTER  # noqa: E402
from plot_finalist_selection import short_name  # noqa: E402
from predval_dirs import parse_trainer_key  # noqa: E402

from ensemble_val import DEFAULT_TRAINERS, build_ensemble, paired_bootstrap_vs_top  # noqa: E402

NNUNET_RAW = Path("/home/galia/ISLES2026/nnUNet_raw")
NNUNET_RESULTS = Path("/home/galia/ISLES2026/nnUNet_results")
DATASET_NAME = "Dataset002_ATLAS"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS = ["dice", "hd95_mm", "lesion_f1"]


def predtest_dir(key: str) -> Path:
    """Resolve a trainer(+plans) key to its test-set probability directory.

    Same fallback shape as `predval_dirs.predval_dir()` for val: prefers the
    dedicated `predTs_prob/` export, but falls back to the plain `predTs/`
    folder if that one has the real `.npz` probabilities instead --
    concretely, both ResEncM-plans runs (`Baseline`, `WideAugBaseline`) were
    predicted with `--save_probabilities` on their original `predTs` run
    (250/250 cases each), so they never needed a separate `predTs_prob/`
    export at all -- confirmed 2026-08-20, not stale (same checkpoint,
    written together with the masks in one predict call)."""
    trainer, plans = parse_trainer_key(key)
    base = NNUNET_RESULTS / DATASET_NAME / f"{trainer}__{plans}__3d_fullres" / "fold_0"
    exported = base / "predTs_prob"
    if any(exported.glob("*.npz")):
        return exported
    fallback = base / "predTs"
    if any(fallback.glob("*.npz")):
        return fallback
    return exported  # neither has real data; check_predtest_dirs reports this as missing


def check_predtest_dirs(trainers: list[str]) -> dict[str, Path]:
    dirs = {}
    missing = []
    for t in trainers:
        d = predtest_dir(t)
        if any(d.glob("*.npz")):
            dirs[t] = d
            source = "predTs_prob/ (dedicated export)" if d.name == "predTs_prob" \
                else "predTs/ (probabilities saved on the original predict run, no separate export exists)"
            print(f"  {t}: using {source} -> {d}")
        else:
            missing.append((t, d))
    if missing:
        lines = ["Missing predTs_prob/ for:"]
        for t, d in missing:
            trainer, plans = parse_trainer_key(t)
            plans_flag = f" -p {plans}" if plans != "nnUNetPlans" else ""
            lines.append(
                f"  - {t}\n"
                f"    mkdir -p {d} && nnUNetv2_predict "
                f"-i {NNUNET_RAW / DATASET_NAME / 'imagesTs'} -o {d} "
                f"-d 2 -c 3d_fullres -tr {trainer}{plans_flag} -f 0 --save_probabilities -device cuda"
            )
        raise SystemExit("\n".join(lines) + "\n\nRun the command(s) above first -- this script never launches inference itself.")
    return dirs


def run_compute_metrics(pred_dir: Path, gt_dir: Path, case_metadata_csv: Path, experiment_name: str, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, str(PROJECT_ROOT / "evaluation" / "compute_metrics.py"),
            "--pred-dir", str(pred_dir), "--gt-dir", str(gt_dir),
            "--case-metadata-csv", str(case_metadata_csv),
            "--experiment-name", experiment_name, "--out-csv", str(out_csv),
        ],
        check=True,
    )


def build_report_table(
    dfs: dict[str, pd.DataFrame], ensemble_label: str, p_value_against: str,
    n_boot: int, seed: int, alpha: float,
) -> pd.DataFrame:
    """Report-ready table: one row per (breakdown, metric), one column per
    model's mean -- overall, per split (test_id/test_ood), AND per size_bin
    (small/medium/large), not just split. The size_bin breakdown matters as
    much as split here: per "General Training Tips" (`docs/reference/`)
    "Match Architecture to Pathology Size" -- small lesions are the case a
    generic pipeline is most likely to lose signal on, and the same deck's
    "best-loss checkpoint is not always the best task-metric checkpoint"
    point applies just as much to "best pooled Dice is not always best on
    the hardest subgroup" -- a model/ensemble that only wins on pooled
    Dice (dominated by the largest lesions, which contribute the most
    voxels) could still be worse specifically where it matters most.

    Statistical significance (paired bootstrap, same method as
    `ensemble_val.py`'s val-side comparison) is computed for exactly one
    comparison -- ensemble vs. `p_value_against` -- not one per model
    shown, to avoid an uncontrolled multiple-comparisons problem; every
    other model in `dfs` still gets its own mean columns for context.
    `dfs` values must already be restricted to the same shared case_id set
    (paired comparison requires it -- see `check_paired`/the case-id
    intersection in `main()`)."""
    ensemble_df = dfs[ensemble_label]
    compare_df = dfs[p_value_against]
    breakdowns = (
        [("overall", "pooled", df) for df in [ensemble_df]]
        + [("split", name, group) for name, group in ensemble_df.groupby("split")]
        + [("size_bin", name, group) for name, group in ensemble_df.groupby("size_bin")]
    )
    rows = []
    for breakdown_type, breakdown_value, group_e in breakdowns:
        case_ids = set(group_e["case_id"])
        for metric in METRICS:
            row = {"breakdown_type": breakdown_type, "breakdown_value": breakdown_value,
                   "metric": metric, "n_cases": len(case_ids)}
            for label, df in dfs.items():
                subset = df[df["case_id"].isin(case_ids)]
                row[f"{label}_mean"] = subset[metric].mean()
            lower_better = metric in LOWER_IS_BETTER
            e_idx = group_e.set_index("case_id")[metric]
            c_idx = compare_df[compare_df["case_id"].isin(case_ids)].set_index("case_id")[metric]
            stats = paired_bootstrap_vs_top(e_idx, c_idx, n_boot, seed, alpha, lower_better)
            row.update({
                "vs_trainer": p_value_against, "mean_diff": stats["mean_diff_vs_top_single"],
                "ci_low": stats["ci_low"], "ci_high": stats["ci_high"], "p_value": stats["p_value"],
                "significant": stats["significant"], "ensemble_better": stats["ensemble_better_than_top_single"],
            })
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combo", nargs="+", default=DEFAULT_TRAINERS)
    ap.add_argument("--compare-against", nargs="+", default=None,
                     help="Trainer(s) with an existing results_test.csv to compare against. "
                          "Default: the single best val performer (--primary-metric).")
    ap.add_argument("--primary-metric", default="dice", choices=METRICS,
                     help="Used to pick the default --compare-against trainer (highest val dice, "
                          "or lowest val hd95_mm) when --compare-against isn't given explicitly.")
    ap.add_argument("--runs-dir", default="workspace/results/runs", type=Path)
    ap.add_argument("--gt-dir", default=str(NNUNET_RAW / DATASET_NAME / "labelsTs"), type=Path)
    ap.add_argument("--case-metadata-csv", default="workspace/splits_dataset002/manifest.csv", type=Path)
    ap.add_argument("--work-dir", default="workspace/results/ensemble_test_tmp", type=Path)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    label = "+".join(short_name(t) for t in args.combo)
    ensemble_name = f"ensemble_{label}"
    run_dir = args.runs_dir / ensemble_name

    print(f"Resolving test-set probability directories for combo: {label}")
    dirs = check_predtest_dirs(args.combo)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nBuilding ensemble ({len(args.combo)} members) ...")
    out_folder = build_ensemble(dirs, tuple(args.combo), args.work_dir)

    out_csv = run_dir / "results_test.csv"
    print(f"\nScoring against {args.gt_dir} ...")
    run_compute_metrics(out_folder, args.gt_dir, args.case_metadata_csv, ensemble_name, out_csv)

    manifest = {
        "run_id": ensemble_name, "trainer": ensemble_name, "is_ensemble": True,
        "ensemble_members": list(args.combo), "split": "test",
        "gt_dir": str(args.gt_dir), "case_metadata_csv": str(args.case_metadata_csv),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    val_runs = discover_runs(args.runs_dir)  # default filename=results_val.csv
    val_df = load_all(val_runs)
    ascending = args.primary_metric in LOWER_IS_BETTER
    primary_trainer = val_df.groupby("trainer")[args.primary_metric].mean().sort_values(ascending=ascending).index[0]

    if not args.compare_against:
        args.compare_against = [primary_trainer]
        print(f"\n--compare-against not given -- defaulting to the best val performer "
              f"({args.primary_metric}): {short_name(primary_trainer)}")
    elif primary_trainer not in args.compare_against:
        # The p-value is always computed against the best single model by
        # --primary-metric, even if it wasn't explicitly requested -- otherwise
        # a report table could end up with no significance test against the
        # one model that actually matters most for "did ensembling help".
        args.compare_against = list(args.compare_against) + [primary_trainer]
        print(f"\nAdding {short_name(primary_trainer)} (best val performer by {args.primary_metric}) "
              f"to --compare-against -- it's always the p-value anchor.")

    test_runs = discover_runs(args.runs_dir, filename="results_test.csv")
    ensemble_df = pd.read_csv(out_csv)
    dfs = {ensemble_name: ensemble_df}
    missing = []
    for compare_trainer in args.compare_against:
        if compare_trainer not in test_runs:
            missing.append(compare_trainer)
            continue
        dfs[short_name(compare_trainer)] = pd.read_csv(test_runs[compare_trainer])
    if missing:
        print(f"\n[skip] no existing results_test.csv found for: {', '.join(missing)} -- "
              "this script never re-predicts single models, only reuses existing scores.")

    common = set.intersection(*(set(df["case_id"]) for df in dfs.values()))
    dropped = max(len(df) for df in dfs.values()) - len(common)
    if dropped:
        print(f"\n[note] {dropped} case(s) aren't shared across every model being compared -- "
              f"the report table below uses the {len(common)} shared cases only.")
    dfs = {label: df[df["case_id"].isin(common)] for label, df in dfs.items()}

    p_value_against = short_name(primary_trainer)
    table = build_report_table(dfs, ensemble_name, p_value_against, args.n_bootstrap, args.seed, args.alpha)
    report_csv = run_dir / "report_table.csv"
    table.to_csv(report_csv, index=False)
    print(f"\n=== Report table: {ensemble_name} vs. {', '.join(l for l in dfs if l != ensemble_name)} "
          f"(p-value anchored on {p_value_against}) ===")
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))

    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {run_dir / 'run_manifest.json'}")
    print(f"Wrote: {report_csv}")


if __name__ == "__main__":
    main()
