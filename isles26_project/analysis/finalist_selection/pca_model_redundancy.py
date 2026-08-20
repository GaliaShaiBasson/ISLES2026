#!/usr/bin/env python3
"""PCA/clustering view of the val-set finalist question: how many
*genuinely distinct* models are actually among these trainers, and which
ones are redundant enough to drop?

Complements (does not replace) `select_finalist_from_val.py`'s per-metric
ranking and `plot_finalist_selection.py`'s pairwise correlation heatmap.
The heatmap only shows pairwise redundancy; PCA/clustering can catch a
3+-way redundant cluster where no single pair crosses a correlation
threshold but the group still collectively spans just one axis of real
variation.

Expected input: same `workspace/results/runs/*/results_val.csv` files as
the other two scripts (imports their loaders directly).

Method: build a (trainer x case) matrix of per-case val Dice, treat each
*trainer* as one point in the n_cases-dimensional space of its per-case
scores (not the usual "samples=cases" PCA framing -- here every case is a
feature/axis and every trainer is the thing being embedded). PCA centers
each case-column by its mean across trainers first, which is deliberate:
it strips out per-case difficulty (a case that's hard for everyone stays
hard for everyone) and leaves only *how a trainer's errors deviate from
the group average, case by case* -- that deviation pattern is what
"redundant" vs. "complementary" actually means for ensembling.

What it produces (under `--out-dir`, default
`workspace/figures/finalist_selection/`):
- `pca_scree.png` -- explained variance per component + cumulative line.
  If 1-2 components already cover most of the variance, that's a direct
  quantitative answer to "how many effectively-different models do we
  have" -- e.g. 90% in 2 PCs among 7 trainers means 5 of them are mostly
  redundant with each other along the same one or two axes of behavior.
- `pca_scatter.png` -- trainers projected onto PC1/PC2, same fixed
  per-trainer colors as `plot_finalist_selection.py`. Tight clusters =
  redundant; isolated points = distinct error behavior, worth keeping for
  diversity even if their raw metric rank is middling.
- `dendrogram.png` -- hierarchical clustering (average linkage) on
  1 - Pearson-r per-case-Dice distance between trainers -- a structural
  view of the same correlation heatmap, showing which trainers merge into
  a cluster before any given threshold.
- `redundancy_recommendation.csv` -- flat clusters cut at
  `--redundancy-corr-threshold` (default 0.95 correlation, i.e. distance
  0.05); within any cluster of size > 1, the member ranked best on val by
  `select_finalist_from_val.py`'s primary-metric rule is marked "keep",
  the rest "drop (redundant with <best-in-cluster>)". This is a
  recommendation to sanity-check against the correlation heatmap and the
  by-size-bin breakdown, not an automatic decision -- a statistically
  tied-but-cheap model may still be worth keeping for reasons outside
  this script's view (e.g. training cost, interpretability).

Usage:
    python analysis/pca_model_redundancy.py
    python analysis/pca_model_redundancy.py --redundancy-corr-threshold 0.9
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
from select_finalist_from_val import (  # noqa: E402
    check_paired,
    discover_runs,
    load_all,
    rank_trainers,
    summarize,
)
from plot_finalist_selection import short_name, trainer_colors  # noqa: E402


def build_trainer_case_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Rows = trainer, columns = case_id, values = per-case val Dice."""
    pivot = df.pivot(index="case_id", columns="trainer", values="dice")
    return pivot.T  # trainer x case


def plot_scree(pca: PCA, out_dir: Path) -> None:
    ratios = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(ratios)
    x = np.arange(1, len(ratios) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x, ratios, color="#2a78d6", width=0.6, label="per-component")
    ax.plot(x, cumulative, color="#e34948", marker="o", markersize=5, linewidth=1.5, label="cumulative")
    ax.set_xticks(x)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_ylim(0, 105)
    ax.axhline(90, color="#52514e", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(x[-1], 91, "90%", fontsize=8, color="#52514e", ha="right")
    ax.set_title("How many independent axes of per-case behavior exist among the trainers?")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "pca_scree.png", dpi=180)
    plt.close(fig)


def plot_scatter(scores: np.ndarray, trainers: list[str], pca: PCA, colors: dict[str, str], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6.5))
    pc1_pct = pca.explained_variance_ratio_[0] * 100
    pc2_pct = pca.explained_variance_ratio_[1] * 100 if scores.shape[1] > 1 else 0.0
    for i, t in enumerate(trainers):
        ax.scatter(scores[i, 0], scores[i, 1] if scores.shape[1] > 1 else 0, s=140, color=colors[t], zorder=3,
                   edgecolor="white", linewidth=1)
        ax.annotate(short_name(t), (scores[i, 0], scores[i, 1] if scores.shape[1] > 1 else 0),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.axhline(0, color="#c3c2b7", linewidth=1, zorder=1)
    ax.axvline(0, color="#c3c2b7", linewidth=1, zorder=1)
    ax.set_xlabel(f"PC1 ({pc1_pct:.1f}% of variance)")
    ax.set_ylabel(f"PC2 ({pc2_pct:.1f}% of variance)")
    ax.set_title("Trainers embedded by per-case Dice deviation pattern\n(close together = redundant, far apart = complementary errors)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "pca_scatter.png", dpi=180)
    plt.close(fig)


def cluster_and_plot_dendrogram(
    matrix: pd.DataFrame, out_dir: Path, corr_threshold: float
) -> pd.DataFrame:
    corr = matrix.T.corr()  # trainer x trainer, since matrix is trainer x case
    dist = 1 - corr
    np.fill_diagonal(dist.values, 0.0)
    dist_condensed = squareform(dist.values, checks=False)
    Z = linkage(dist_condensed, method="average")

    labels = [short_name(t) for t in matrix.index]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(labels) + 2.2))
    dendrogram(
        Z, labels=labels, ax=ax, orientation="right",
        color_threshold=1 - corr_threshold, above_threshold_color="#c3c2b7",
    )
    ax.axvline(1 - corr_threshold, color="#e34948", linestyle="--", linewidth=1)
    ax.text(1 - corr_threshold, 0.98, f"cut at r={corr_threshold}", fontsize=8, color="#e34948",
            transform=ax.get_xaxis_transform(), ha="left", va="top", rotation=90,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.set_xlabel("Distance (1 − Pearson r, per-case val Dice)")
    ax.set_title("Hierarchical clustering of trainers by per-case Dice similarity", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "dendrogram.png", dpi=180)
    plt.close(fig)

    cluster_ids = fcluster(Z, t=1 - corr_threshold, criterion="distance")
    return pd.DataFrame({"trainer": matrix.index, "cluster_id": cluster_ids})


def recommend(clusters: pd.DataFrame, df: pd.DataFrame, primary_metric: str, out_dir: Path) -> None:
    ranked = rank_trainers(summarize(df, ["trainer"]), primary_metric)
    rank_order = {t: i for i, t in enumerate(ranked["trainer"])}
    clusters = clusters.copy()
    clusters["val_rank"] = clusters["trainer"].map(rank_order)

    rows = []
    for cluster_id, group in clusters.groupby("cluster_id"):
        members = group.sort_values("val_rank")
        best = members.iloc[0]["trainer"]
        cluster_members = ", ".join(short_name(t) for t in members["trainer"])
        for _, row in members.iterrows():
            if row["trainer"] == best:
                action = "keep (best-ranked in its redundancy cluster)" if len(members) > 1 else "keep (no redundant peer found)"
                reason = "" if len(members) == 1 else f"cluster: {cluster_members}"
            else:
                action = f"drop candidate -- redundant with {short_name(best)}"
                reason = f"cluster: {cluster_members}"
            rows.append({
                "trainer": row["trainer"], "cluster_id": cluster_id, "cluster_size": len(members),
                "val_rank_in_cluster": int(row["val_rank"]), "recommended_action": action, "reason": reason,
            })
    out = pd.DataFrame(rows).sort_values(["cluster_id", "val_rank_in_cluster"])
    out.to_csv(out_dir / "redundancy_recommendation.csv", index=False)

    print("\n=== Redundancy clusters (correlation-threshold cut) ===")
    for cluster_id, group in out.groupby("cluster_id"):
        if len(group) > 1:
            print(f"Cluster {cluster_id}: {', '.join(short_name(t) for t in group['trainer'])}")
            for _, row in group.iterrows():
                print(f"    {short_name(row['trainer']):40s} -> {row['recommended_action']}")
        else:
            t = group.iloc[0]["trainer"]
            print(f"Cluster {cluster_id}: {short_name(t)} (singleton, distinct from all others)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="workspace/results/runs", type=Path)
    ap.add_argument("--out-dir", default="workspace/figures/finalist_selection", type=Path)
    ap.add_argument("--primary-metric", default="hd95_mm")
    ap.add_argument("--redundancy-corr-threshold", type=float, default=0.95,
                     help="Trainers whose per-case Dice correlation exceeds this are clustered as redundant.")
    ap.add_argument("--trainer-filter", default=None)
    args = ap.parse_args()

    runs = discover_runs(args.runs_dir)
    if args.trainer_filter:
        runs = {t: p for t, p in runs.items() if args.trainer_filter in t}
    if len(runs) < 3:
        raise SystemExit(f"Need at least 3 trainers with results_val.csv for a meaningful PCA/clustering view, found {len(runs)}")

    df = load_all(runs)
    check_paired(df)
    colors = trainer_colors(list(runs.keys()))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    matrix = build_trainer_case_matrix(df)  # trainer x case
    n_components = min(matrix.shape[0] - 1, matrix.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(matrix.values)

    plot_scree(pca, args.out_dir)
    plot_scatter(scores, list(matrix.index), pca, colors, args.out_dir)
    clusters = cluster_and_plot_dendrogram(matrix, args.out_dir, args.redundancy_corr_threshold)
    recommend(clusters, df, args.primary_metric, args.out_dir)

    cum2 = pca.explained_variance_ratio_[:2].sum() * 100
    print(f"\nPC1+PC2 explain {cum2:.1f}% of the trainer-to-trainer variance "
          f"(n={matrix.shape[0]} trainers over {matrix.shape[1]} shared val cases).")

    for name in ["pca_scree.png", "pca_scatter.png", "dendrogram.png", "redundancy_recommendation.csv"]:
        p = args.out_dir / name
        if p.exists():
            print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
