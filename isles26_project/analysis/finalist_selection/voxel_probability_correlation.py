#!/usr/bin/env python3
"""Per-voxel foreground-probability correlation between every trained
condition on val -- the probability-level twin of `select_finalist_from_val.py`
(which does the same job on per-case Dice). Read-only: computes and writes
numbers only, no plotting (see `plot_voxel_probability_correlation.py`) and
no real ensembling (see `ensembling/ensemble_val.py`).

Expected input layout: `workspace/results/runs/<run_id>/results_val.csv`
(for trainer auto-discovery + the shared val case_id set, via
`select_finalist_from_val.discover_runs`/`load_all`/`check_paired` -- reused
directly rather than reimplemented, so this and the case-Dice analysis
always agree on which runs/cases are in scope) plus each trainer's real
val-set probability `.npz` files, resolved via `predval_dirs.check_predval_dirs`.

What it produces (under `--out-dir`, default
`workspace/results/finalist_selection/`):
- `voxel_probability_correlation.csv` -- one row/column per trainer, Pearson
  r of the foreground-probability channel, averaged over the shared val
  cases. A finer-grained complement to `dice_correlation_heatmap.csv`
  (`plot_finalist_selection.py`): two trainers can land the same Dice on a
  case while disagreeing heavily on which voxels they're unsure about --
  exactly the disagreement ensembling exploits, and a case-level Dice
  correlation can't see it. Also the direct input to
  `pca_probability_redundancy.py`.

Non-obvious rationale: runs over the FULL candidate set by default (every
trainer with a `results_val.csv`), not a pre-chosen shortlist -- this is the
selection step (which trainers look complementary enough to be worth
ensembling), so restricting it to an already-chosen shortlist would be
circular. `ensembling/ensemble_val.py`'s `DEFAULT_TRAINERS` shortlist is
downstream of this script's output, not an input to it.

Usage:
    python analysis/finalist_selection/voxel_probability_correlation.py
    python analysis/finalist_selection/voxel_probability_correlation.py --trainer-filter 500epochs
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_finalist_from_val import discover_runs, load_all, check_paired  # noqa: E402
from predval_dirs import check_predval_dirs  # noqa: E402


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

    corr = pd.DataFrame(np.eye(len(trainers)), index=trainers, columns=trainers)
    for (a, b), vals in sums.items():
        mean_r = float(np.mean(vals)) if vals else np.nan
        corr.loc[a, b] = mean_r
        corr.loc[b, a] = mean_r
    return corr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="workspace/results/runs", type=Path)
    ap.add_argument("--out-dir", default="workspace/results/finalist_selection", type=Path)
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

    single_df = load_all(runs)
    common_cases = check_paired(single_df)
    print(f"\nAll {len(runs)} trainers share the same {len(common_cases)} val cases.")

    print(f"\nResolving val-set probability directories for {len(runs)} trainer(s)...")
    dirs = check_predval_dirs(list(runs.keys()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nComputing per-voxel probability correlation over {len(common_cases)} shared val cases...")
    corr = voxel_probability_correlation(dirs, common_cases)
    corr.to_csv(args.out_dir / "voxel_probability_correlation.csv")

    off_diag = corr.where(~np.eye(len(corr), dtype=bool))
    print(f"\nLowest voxel-probability correlation pair (most complementary): "
          f"{off_diag.stack().idxmin()} = {off_diag.stack().min():.3f}")
    print(f"Highest voxel-probability correlation pair (most redundant): "
          f"{off_diag.stack().idxmax()} = {off_diag.stack().max():.3f}")

    print(f"\nWrote: {args.out_dir / 'voxel_probability_correlation.csv'}")
    print("Run plot_voxel_probability_correlation.py for the heatmap figure, "
          "or pca_probability_redundancy.py for the PCA/clustering view.")


if __name__ == "__main__":
    main()
