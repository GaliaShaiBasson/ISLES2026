#!/usr/bin/env python3
"""Decision-support figures for picking a val-based finalist shortlist.

Expected input: the same `workspace/evaluation/runs/*/results_val.csv` files
`select_finalist_from_val.py` reads (imports its loader directly rather than
re-implementing it, so both scripts always agree on which runs/cases are in
scope), plus that script's `pairwise_vs_top_val.csv` output for the forest
plot -- run `select_finalist_from_val.py` first if it's missing.

What it produces (under `--out-dir`, default
`workspace/figures/finalist_selection/`):
- `ranked_metrics.png` -- one bar-per-trainer, sorted, for dice/HD95/lesion-F1
  each in its own panel (never on a shared/dual axis -- the three metrics
  have different units and "better" directions, so overlaying them on one
  axis would be misleading), error bars = 95% CI of the per-case mean via
  normal approximation (SEM * 1.96).
- `dice_distribution.png` -- per-case Dice as a box+strip plot per trainer,
  sorted by median. The ranked-bar plot alone can't show that two trainers
  with similar means have very different spread/shape; this can.
- `dice_by_size_bin.png` -- grouped bars, one panel per size_bin
  (small/medium/large), bars = trainers. Answers "does the val ranking
  hold up within each difficulty tier, or is one trainer just winning on
  the easy (large) cases".
- `forest_plot_vs_top.png` -- paired bootstrap mean-difference-vs-top-trainer
  with its 95% CI as a horizontal errorbar per trainer (a forest plot) --
  the direct picture of "which trainers are actually distinguishable from
  the leader," matching `pairwise_vs_top_val.csv`'s numbers.
- `dice_correlation_heatmap.png` -- NxN Pearson correlation of *per-case*
  Dice between every pair of trainers (over the shared val case set).
  Rationale (why this is a real question, not just a check): if two
  trainers are highly correlated case-by-case, they're failing/succeeding
  on the same cases -- averaging their softmax outputs (ensembling) buys
  little, since there's no complementary error pattern to cancel out. A
  lower-correlation pair, even with similar mean Dice, is a better
  ensembling candidate because their errors are more independent. This
  reframes "which trainer is best" (the other plots) into "which *pair* is
  most worth ensembling" -- a different, complementary question the mean
  metrics alone can't answer.

Color: every trainer gets one fixed hue, assigned once by alphabetical
trainer-name order and reused identically across every panel/plot in this
script (ranked bars, box plot, grouped bars, forest plot) -- so a trainer's
color means the same thing everywhere, and swapping which metric is "on
top" never repaints who's who.

Usage:
    python analysis/plot_finalist_selection.py
    python analysis/plot_finalist_selection.py --primary-metric dice
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_finalist_from_val import (  # noqa: E402
    LOWER_IS_BETTER,
    SUMMARY_PREFIX,
    check_paired,
    discover_runs,
    load_all,
    rank_trainers,
    summarize,
)

# Fixed 8-hue categorical palette (validated colorblind-safe order) -- reused
# for every panel in this script, assigned once per trainer alphabetically so
# a given trainer keeps the same color in every figure.
PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
BLUE_SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]
SHORT_LABEL_STRIP = ("nnUNetTrainer", "_500epochs_full", "_500epochs")
# Cosmetic renames applied after stripping, for the "(plans)" suffix
# discover_runs() appends when a trainer was run under non-default plans.
SHORT_LABEL_RENAME = {"(nnUNetResEncUNetMPlans)": "(ResEncM)"}


def short_name(trainer: str) -> str:
    name = trainer
    for tok in SHORT_LABEL_STRIP:
        name = name.replace(tok, "")
    for old, new in SHORT_LABEL_RENAME.items():
        name = name.replace(old, new)
    return name or trainer


def trainer_colors(trainers: list[str]) -> dict[str, str]:
    ordered = sorted(trainers)
    return {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(ordered)}


def plot_ranked_metrics(df: pd.DataFrame, colors: dict[str, str], out_dir: Path) -> None:
    metrics = [("dice", "Dice", False), ("hd95_mm", "HD95 (mm)", True), ("lesion_f1", "Lesion-wise F1", False)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (metric, label, lower_better) in zip(axes, metrics):
        stats = df.groupby("trainer")[metric].agg(["mean", "std", "count"]).reset_index()
        stats["sem"] = stats["std"] / np.sqrt(stats["count"])
        stats["ci95"] = stats["sem"] * 1.96
        stats = stats.sort_values("mean", ascending=lower_better)
        bar_colors = [colors[t] for t in stats["trainer"]]
        y_pos = np.arange(len(stats))
        ax.barh(y_pos, stats["mean"], xerr=stats["ci95"], color=bar_colors, edgecolor="none", height=0.65, capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([short_name(t) for t in stats["trainer"]], fontsize=8)
        ax.invert_yaxis()
        direction = "lower is better" if lower_better else "higher is better"
        ax.set_title(f"{label}\n({direction})", fontsize=10)
        ax.set_xlabel(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Val-set ranking per metric, mean ± 95% CI (n cases per trainer shown in summary_val.csv)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "ranked_metrics.png", dpi=180)
    plt.close(fig)


def plot_dice_distribution(df: pd.DataFrame, colors: dict[str, str], out_dir: Path) -> None:
    order = df.groupby("trainer")["dice"].median().sort_values(ascending=False).index.tolist()
    data = [df.loc[df["trainer"] == t, "dice"].to_numpy() for t in order]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    positions = np.arange(len(order))
    bp = ax.boxplot(
        data, positions=positions, widths=0.5, patch_artist=True, showfliers=False,
        medianprops={"color": "#0b0b0b", "linewidth": 1.5},
    )
    for patch, t in zip(bp["boxes"], order):
        patch.set_facecolor(colors[t])
        patch.set_alpha(0.55)
        patch.set_edgecolor(colors[t])
    rng = np.random.default_rng(0)
    for pos, vals, t in zip(positions, data, order):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(pos + jitter, vals, s=10, color=colors[t], alpha=0.5, linewidths=0)

    ax.set_xticks(positions)
    ax.set_xticklabels([short_name(t) for t in order], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Per-case Dice")
    ax.set_title("Per-case val Dice distribution, sorted by median (same 119 cases every trainer)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "dice_distribution.png", dpi=180)
    plt.close(fig)


def plot_by_size_bin(df: pd.DataFrame, colors: dict[str, str], out_dir: Path) -> None:
    if "size_bin" not in df.columns:
        print("[skip] size_bin column absent")
        return
    bins = [b for b in ["small", "medium", "large"] if b in df["size_bin"].unique()]
    if not bins:
        print("[skip] no small/medium/large cases in val")
        return
    overall_order = df.groupby("trainer")["dice"].mean().sort_values(ascending=False).index.tolist()

    fig, axes = plt.subplots(1, len(bins), figsize=(5.5 * len(bins), 5), sharey=True)
    if len(bins) == 1:
        axes = [axes]
    for ax, size_bin in zip(axes, bins):
        sub = df[df["size_bin"] == size_bin]
        stats = sub.groupby("trainer")["dice"].agg(["mean", "std", "count"]).reindex(overall_order)
        stats["ci95"] = 1.96 * stats["std"] / np.sqrt(stats["count"])
        bar_colors = [colors[t] for t in stats.index]
        x_pos = np.arange(len(stats))
        ax.bar(x_pos, stats["mean"], yerr=stats["ci95"], color=bar_colors, capsize=3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([short_name(t) for t in stats.index], rotation=45, ha="right", fontsize=7)
        n = int(stats["count"].iloc[0]) if len(stats) else 0
        ax.set_title(f"{size_bin} (n={n})")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Mean Dice")
    fig.suptitle("Val Dice by lesion-size bin -- does the overall ranking hold within each tier?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "dice_by_size_bin.png", dpi=180)
    plt.close(fig)


def plot_forest(pairwise_csv: Path, colors: dict[str, str], out_dir: Path, primary_metric: str) -> None:
    if not pairwise_csv.exists():
        print(f"[skip] {pairwise_csv} not found -- run select_finalist_from_val.py first")
        return
    pw = pd.read_csv(pairwise_csv)
    if pw.empty:
        print("[skip] pairwise_vs_top_val.csv is empty (only one trainer)")
        return
    ci_low_col = [c for c in pw.columns if c.startswith("ci_low_")][0]
    ci_high_col = [c for c in pw.columns if c.startswith("ci_high_")][0]
    pw = pw.sort_values("mean_diff_vs_top")

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(pw) + 1.5))
    y_pos = np.arange(len(pw))
    for y, (_, row) in zip(y_pos, pw.iterrows()):
        color = colors.get(row["trainer"], "#52514e")
        err_low = row["mean_diff_vs_top"] - row[ci_low_col]
        err_high = row[ci_high_col] - row["mean_diff_vs_top"]
        alpha = 1.0 if row["significant"] else 0.4
        ax.errorbar(
            row["mean_diff_vs_top"], y, xerr=[[err_low], [err_high]],
            fmt="o", color=color, ecolor=color, alpha=alpha, capsize=4, markersize=7,
        )
    ax.axvline(0, color="#52514e", linewidth=1, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([short_name(t) for t in pw["trainer"]], fontsize=8)
    ax.set_xlabel(f"{primary_metric}: (trainer − top), 95% bootstrap CI  |  faded = not significant (tie)")
    ax.set_title(f"Paired difference vs. top trainer on val ({primary_metric})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "forest_plot_vs_top.png", dpi=180)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    pivot = df.pivot(index="case_id", columns="trainer", values="dice")
    corr = pivot.corr()
    labels = [short_name(t) for t in corr.columns]

    vmin = max(0.0, float(np.floor(corr.values[~np.eye(len(corr), dtype=bool)].min() * 20) / 20 - 0.05))
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
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=text_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r, per-case Dice")
    ax.set_title(
        "Per-case Dice correlation between trainers (val)\nlower = more complementary errors = better ensemble candidate",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "dice_correlation_heatmap.png", dpi=180)
    plt.close(fig)

    corr.to_csv(out_dir / "dice_correlation_heatmap.csv")
    off_diag = corr.where(~np.eye(len(corr), dtype=bool))
    min_pair = off_diag.stack().idxmin()
    print(f"\nLowest per-case Dice correlation pair (most complementary, best ensemble candidate): "
          f"{min_pair[0]} vs {min_pair[1]} = {off_diag.stack().min():.3f}")
    max_pair = off_diag.stack().idxmax()
    print(f"Highest correlation pair (most redundant): {max_pair[0]} vs {max_pair[1]} = {off_diag.stack().max():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="workspace/evaluation/runs", type=Path)
    ap.add_argument("--selection-dir", default="workspace/evaluation/finalist_selection", type=Path,
                     help="Where select_finalist_from_val.py wrote pairwise_vs_top_val.csv.")
    ap.add_argument("--out-dir", default="workspace/figures/finalist_selection", type=Path)
    ap.add_argument("--primary-metric", default="hd95_mm", choices=list(SUMMARY_PREFIX))
    ap.add_argument("--trainer-filter", default=None)
    args = ap.parse_args()

    runs = discover_runs(args.runs_dir)
    if args.trainer_filter:
        runs = {t: p for t, p in runs.items() if args.trainer_filter in t}
    if not runs:
        raise SystemExit(f"No results_val.csv found under {args.runs_dir}")

    df = load_all(runs)
    check_paired(df)
    colors = trainer_colors(list(runs.keys()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_ranked_metrics(df, colors, args.out_dir)
    plot_dice_distribution(df, colors, args.out_dir)
    plot_by_size_bin(df, colors, args.out_dir)
    plot_forest(args.selection_dir / "pairwise_vs_top_val.csv", colors, args.out_dir, args.primary_metric)
    plot_correlation_heatmap(df, args.out_dir)

    for name in ["ranked_metrics.png", "dice_distribution.png", "dice_by_size_bin.png",
                 "forest_plot_vs_top.png", "dice_correlation_heatmap.png"]:
        p = args.out_dir / name
        if p.exists():
            print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
