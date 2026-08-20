#!/usr/bin/env python3
"""PCA/clustering view of voxel-probability redundancy -- the probability-
level twin of `pca_model_redundancy.py` (which does the same job on
per-case Dice). Answers the same question ("how many genuinely distinct
models are there, and which are redundant enough to drop") from a richer
signal: per-voxel confidence agreement instead of a single per-case Dice
scalar.

Expected input: `voxel_probability_correlation.csv` (written by
`voxel_probability_correlation.py`), an NxN Pearson-r matrix of per-voxel
foreground-probability correlation, indexed/columned by trainer -- plus the
same `workspace/evaluation/runs/*/results_val.csv` files `pca_model_
redundancy.py` reads (via `select_finalist_from_val`'s loader), used only to
rank which member of a redundant cluster to keep (the official
hd95_mm/dice ranking, same rule used everywhere else in this project --
not re-derived from probabilities).

Method: unlike `pca_model_redundancy.py` (which builds a fresh trainer x case
Dice matrix and correlates it), this script starts from the correlation
matrix directly -- it does NOT recompute a trainer x case matrix from raw
probabilities. Reason: the raw `.npz` arrays are in nnU-Net's internal
cropped/resampled space, not necessarily the original image grid; turning
them into a proper per-case scalar against ground truth would require
redoing the same crop/resample-back bookkeeping `ensemble_folders` already
handles correctly for real ensembling (see `ensembling/ensemble_val.py`) --
reimplementing that here would risk silently diverging from the trusted
path. The already-computed, already-validated trainer x trainer correlation
matrix carries the same "how similar do these trainers behave" signal
without that risk: each trainer is embedded via its full row of pairwise
correlations with every other trainer (eigendecomposition of the
correlation matrix itself, via `sklearn.decomposition.PCA`).

What it produces (under `--out-dir`, default
`workspace/evaluation/finalist_selection/probability/` -- a subdirectory, so
it never collides with `pca_model_redundancy.py`'s case-Dice outputs of the
same filenames):
- `pca_scree.png` -- explained variance per component + cumulative line.
- `pca_scatter.png` -- trainers projected onto PC1/PC2 of the correlation
  matrix, same fixed per-trainer colors as the rest of this project's
  finalist-selection figures.
- `dendrogram.png` -- hierarchical clustering (average linkage) on
  1 - Pearson-r voxel-probability distance between trainers.
- `redundancy_recommendation.csv` -- flat clusters cut at
  `--redundancy-corr-threshold` (default 0.95); within any cluster of
  size > 1, the member ranked best on val by the official primary-metric
  rule is marked "keep", the rest "drop (redundant with <best-in-cluster>)".
  A recommendation to sanity-check against `pca_model_redundancy.py`'s own
  (case-Dice-based) recommendation, not an automatic decision -- the two
  can legitimately disagree (a pair can share overall Dice while still
  disagreeing heavily at the voxel level, or vice versa), and a disagreement
  between them is itself informative.

Usage:
    python analysis/pca_probability_redundancy.py
    python analysis/pca_probability_redundancy.py --redundancy-corr-threshold 0.9
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_finalist_from_val import discover_runs, load_all, check_paired  # noqa: E402
from plot_finalist_selection import short_name, trainer_colors  # noqa: E402
from pca_model_redundancy import plot_scree, plot_scatter, recommend  # noqa: E402


def cluster_and_plot_dendrogram(corr: pd.DataFrame, out_dir: Path, corr_threshold: float) -> pd.DataFrame:
    """Same method as `pca_model_redundancy.cluster_and_plot_dendrogram`, but
    starting from an already-computed correlation matrix instead of deriving
    one from a trainer x case matrix."""
    dist = 1 - corr
    np.fill_diagonal(dist.values, 0.0)
    dist_condensed = squareform(dist.values, checks=False)
    Z = linkage(dist_condensed, method="average")

    labels = [short_name(t) for t in corr.index]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(labels) + 2.2))
    dendrogram(
        Z, labels=labels, ax=ax, orientation="right",
        color_threshold=1 - corr_threshold, above_threshold_color="#c3c2b7",
    )
    ax.axvline(1 - corr_threshold, color="#e34948", linestyle="--", linewidth=1)
    ax.text(1 - corr_threshold, 0.98, f"cut at r={corr_threshold}", fontsize=8, color="#e34948",
            transform=ax.get_xaxis_transform(), ha="left", va="top", rotation=90,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.set_xlabel("Distance (1 − Pearson r, per-voxel val probability)")
    ax.set_title("Hierarchical clustering of trainers by voxel-probability similarity", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "dendrogram.png", dpi=180)
    plt.close(fig)

    cluster_ids = fcluster(Z, t=1 - corr_threshold, criterion="distance")
    return pd.DataFrame({"trainer": corr.index, "cluster_id": cluster_ids})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-csv", default="workspace/evaluation/finalist_selection/voxel_probability_correlation.csv", type=Path)
    ap.add_argument("--runs-dir", default="workspace/evaluation/runs", type=Path)
    ap.add_argument("--out-dir", default="workspace/evaluation/finalist_selection/probability", type=Path)
    ap.add_argument("--primary-metric", default="hd95_mm")
    ap.add_argument("--redundancy-corr-threshold", type=float, default=0.95,
                     help="Trainers whose voxel-probability correlation exceeds this are clustered as redundant.")
    ap.add_argument("--trainer-filter", default=None)
    args = ap.parse_args()

    if not args.in_csv.exists():
        raise SystemExit(f"{args.in_csv} not found -- run voxel_probability_correlation.py first.")
    corr = pd.read_csv(args.in_csv, index_col=0)
    if args.trainer_filter:
        keep = [t for t in corr.index if args.trainer_filter in t]
        corr = corr.loc[keep, keep]
    if len(corr) < 3:
        raise SystemExit(f"Need at least 3 trainers for a meaningful PCA/clustering view, found {len(corr)}")

    runs = discover_runs(args.runs_dir)
    runs = {t: p for t, p in runs.items() if t in corr.index}
    df = load_all(runs)
    check_paired(df)
    colors = trainer_colors(list(corr.index))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_components = min(len(corr) - 1, len(corr))
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(corr.values)

    plot_scree(pca, args.out_dir,
               title="How many independent axes of voxel-probability behavior exist among the trainers?")
    plot_scatter(scores, list(corr.index), pca, colors, args.out_dir,
                 title="Trainers embedded by voxel-probability correlation")
    clusters = cluster_and_plot_dendrogram(corr, args.out_dir, args.redundancy_corr_threshold)
    recommend(clusters, df, args.primary_metric, args.out_dir)

    cum2 = pca.explained_variance_ratio_[:2].sum() * 100
    print(f"\nPC1+PC2 (of the voxel-probability correlation matrix) explain {cum2:.1f}% "
          f"of the trainer-to-trainer variance (n={len(corr)} trainers).")

    for name in ["pca_scree.png", "pca_scatter.png", "dendrogram.png", "redundancy_recommendation.csv"]:
        p = args.out_dir / name
        if p.exists():
            print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
