#!/usr/bin/env python3
"""Generate report figures from the aggregated results CSV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import build_palette, short_label  # noqa: E402


def _grouped_boxplot(plot_data: pd.DataFrame, group_col: str, group_order: list[str],
                      title: str, out_path: Path) -> None:
    """One box per (group, experiment), grouped by `group_col` on the x-axis
    and colored by experiment -- replaces the old layout of one box per
    combo with a two-line rotated label crammed under it (illegible past
    ~4 experiments, and the label text alone dwarfed the actual plot at 9+).
    Group names appear once each on the x-axis; experiment identity comes
    from color + one shared legend outside the axes, using the project-wide
    palette (plot_style.py) so a trainer is the same color in every figure.
    """
    experiments = sorted(plot_data["experiment"].unique())
    palette = build_palette(experiments)
    n_exp = len(experiments)
    box_width = min(0.8 / n_exp, 0.18)

    # Legend goes below the axes, multi-column, with its row count reserved
    # explicitly via tight_layout's rect -- an outside-right legend (fine for
    # dice_by_center's already-wide figure) crushed these narrow 3-4-group
    # plots down to a sliver, since bbox_inches='tight' only expands the
    # saved canvas around whatever tight_layout already computed for the
    # axes; it doesn't leave the axes themselves more room.
    legend_ncol = min(3, n_exp)
    legend_nrows = -(-n_exp // legend_ncol)
    legend_height_in = 0.28 * legend_nrows + 0.15  # ~row height + top/bottom padding, in inches
    fig_height_in = 5.5 + legend_height_in
    figure, axis = plt.subplots(figsize=(max(8, len(group_order) * 2.2 + 2), fig_height_in))
    group_centers = np.arange(len(group_order))
    for exp_idx, experiment in enumerate(experiments):
        offset = (exp_idx - (n_exp - 1) / 2) * box_width
        positions = []
        values = []
        for group_idx, group_name in enumerate(group_order):
            data = plot_data.loc[
                (plot_data[group_col] == group_name) & (plot_data["experiment"] == experiment), "dice"
            ].dropna()
            if not data.empty:
                positions.append(group_centers[group_idx] + offset)
                values.append(data.to_numpy())
        if not values:
            continue
        bp = axis.boxplot(values, positions=positions, widths=box_width * 0.9, patch_artist=True,
                           showmeans=True, manage_ticks=False)
        color = palette[experiment]
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.75)
        for element in ("whiskers", "caps", "medians"):
            for artist in bp[element]:
                artist.set_color(color)

    axis.set_xticks(group_centers)
    axis.set_xticklabels(group_order)
    axis.set_title(title)
    axis.set_ylabel("Dice")
    axis.grid(axis="y", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=palette[e], alpha=0.75) for e in experiments]
    fig_bottom = legend_height_in / fig_height_in  # matches the reserved space to the legend's real size
    figure.legend(handles, [short_label(e) for e in experiments],
                  loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=legend_ncol, fontsize=9)
    figure.tight_layout(rect=(0, fig_bottom, 1, 1))
    figure.savefig(out_path, dpi=180)
    plt.close(figure)


def _save_overall_bar(summary: pd.Series, palette: dict, title: str, ylabel: str, out_path: Path) -> None:
    labels = [short_label(e) for e in summary.index]
    colors = [palette[e] for e in summary.index]
    figure, axis = plt.subplots(figsize=(max(8, len(summary) * 0.85), 5))
    axis.bar(labels, summary.to_numpy(), color=colors)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=40)
    for tick in axis.get_xticklabels():
        tick.set_ha("right")
    figure.tight_layout()
    figure.savefig(out_path, dpi=180)
    plt.close(figure)


def save_overall(df: pd.DataFrame, out_dir: Path) -> None:
    metric_cols = [c for c in ("dice", "hd95_mm", "lesion_f1") if c in df.columns]
    summary = df.groupby("experiment", observed=True)[metric_cols].mean()
    palette = build_palette(summary.index)

    _save_overall_bar(summary["dice"], palette, "Mean Dice by experiment", "Dice",
                       out_dir / "overall_dice.png")
    _save_overall_bar(summary["hd95_mm"], palette, "Mean HD95 by experiment (lower is better)", "HD95 (mm)",
                       out_dir / "overall_hd95.png")
    if "lesion_f1" in summary.columns:
        _save_overall_bar(summary["lesion_f1"], palette, "Mean lesion-wise F1 by experiment", "Lesion-wise F1",
                           out_dir / "overall_lesion_f1.png")
    else:
        print("[skip] lesion_f1 column is absent")


def save_by_size(df: pd.DataFrame, out_dir: Path) -> None:
    if "size_bin" not in df.columns:
        print("[skip] size_bin column is absent")
        return
    plot_data = df.dropna(subset=["size_bin", "experiment", "dice"]).copy()
    if plot_data.empty:
        print("[skip] no size-bin data is available")
        return
    present = [b for b in ("small", "medium", "large") if b in plot_data["size_bin"].unique()]
    _grouped_boxplot(plot_data, "size_bin", present,
                      "Dice by lesion-size bin and experiment", out_dir / "dice_by_size_bin.png")


def save_by_split(df: pd.DataFrame, out_dir: Path) -> None:
    """train/val/test_id/test_ood -- the ID-vs-OOD generalization comparison this
    project's split was built to produce (see CLAUDE.md). Mirrors save_by_size's
    grouped-boxplot layout (see _grouped_boxplot).
    """
    if "split" not in df.columns:
        print("[skip] split column is absent (use manifest.csv as --case-metadata-csv when evaluating)")
        return
    plot_data = df.dropna(subset=["split", "experiment", "dice"]).copy()
    if plot_data.empty:
        print("[skip] no split data is available")
        return
    present = [s for s in ("train", "val", "test_id", "test_ood") if s in plot_data["split"].unique()]
    _grouped_boxplot(plot_data, "split", present,
                      "Dice by split (train/val/test_id/test_ood)", out_dir / "dice_by_split.png")


def save_by_center(df: pd.DataFrame, out_dir: Path) -> None:
    if "center" not in df.columns or df["center"].dropna().empty:
        print("[skip] center metadata is absent")
        return
    summary = df.groupby(["center", "experiment"], observed=True)["dice"].agg(["mean", "std"])
    means = summary["mean"].unstack("experiment")
    errors = summary["std"].unstack("experiment").fillna(0)
    experiments = sorted(means.columns)
    means = means[experiments]
    errors = errors[experiments]
    palette = build_palette(experiments)
    colors = [palette[e] for e in experiments]

    # Wide, one-legend-outside figure: with up to 55 centers x 11 experiments,
    # the old fixed 10x5 figure squeezed everything into illegible bars with
    # the legend box sitting on top of the plot (pandas' default in-axes
    # placement) -- width now scales with center count, legend moves out,
    # and every experiment uses its project-wide color (plot_style.py).
    figure, axis = plt.subplots(figsize=(max(12, len(means) * 0.35), 6))
    means.plot(kind="bar", yerr=errors, capsize=2, ax=axis, color=colors, legend=False, width=0.82,
               error_kw={"elinewidth": 0.5, "ecolor": "#555555", "alpha": 0.6})
    axis.set_title("Dice by acquisition center")
    axis.set_xlabel("Center")
    axis.set_ylabel("Dice")
    axis.tick_params(axis="x", rotation=90, labelsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=palette[e]) for e in experiments]
    axis.legend(handles, [short_label(e) for e in experiments],
                loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    figure.tight_layout()
    figure.savefig(out_dir / "dice_by_center.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _rolling_dice_trend(
    volumes: pd.Series, dice: pd.Series, frac: float = 0.35, min_points: int = 8
) -> tuple[np.ndarray, np.ndarray] | None:
    """Smoothed Dice-vs-log(volume) trend line: rolling median over cases sorted by
    volume. A dependency-free stand-in for LOWESS (no statsmodels in requirements.txt)
    -- median rather than mean so a handful of outlier lesions can't yank the line
    around. Returns None if there are too few cases to smooth meaningfully.
    """
    order = np.argsort(volumes.to_numpy())
    sorted_volumes = volumes.to_numpy()[order]
    sorted_dice = dice.to_numpy()[order]
    n = len(sorted_volumes)
    if n < min_points:
        return None
    window = max(min_points, int(round(n * frac)))
    window = min(window, n if n % 2 == 1 else n - 1)  # odd, and no larger than the data
    if window < 3:
        return None
    trend = pd.Series(sorted_dice).rolling(window, center=True, min_periods=max(3, window // 3)).median()
    return sorted_volumes, trend.to_numpy()


def save_volume_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    if "lesion_volume_mm3" not in df.columns:
        print("[skip] lesion_volume_mm3 column is absent")
        return
    plot_data = df.copy()
    plot_data["lesion_volume_mm3"] = pd.to_numeric(plot_data["lesion_volume_mm3"], errors="coerce")
    plot_data = plot_data[plot_data["lesion_volume_mm3"] > 0]
    # Restrict to the 500-epoch trainers (project naming convention: "..._500epochs")
    # so this comparison stays controlled -- mixing in a 250-epoch run (e.g.
    # LesionAwareSamplingPow_250epochs) would confound size-vs-Dice with
    # epoch-budget-vs-Dice, not a like-for-like comparison.
    plot_data = plot_data[plot_data["experiment"].astype(str).str.contains("_500epochs")]
    if plot_data.empty:
        print("[skip] no 500-epoch experiments with positive lesion volumes are available")
        return
    palette = build_palette(plot_data["experiment"].unique())
    figure, axis = plt.subplots(figsize=(9, 6))
    for experiment, group in plot_data.groupby("experiment", observed=True):
        color = palette[experiment]
        axis.scatter(group["lesion_volume_mm3"], group["dice"], alpha=0.35, label=short_label(experiment),
                     color=color)
        trend = _rolling_dice_trend(group["lesion_volume_mm3"], group["dice"])
        if trend is not None:
            axis.plot(*trend, color=color, linewidth=2)
    axis.set_xscale("log")
    axis.set_title("Dice versus lesion volume (points + rolling-median trend)")
    axis.set_xlabel("Lesion volume (mm³, log scale)")
    axis.set_ylabel("Dice")
    # Legend outside the axes, not inline -- with up to 9+ experiments an
    # in-axes legend box was large enough to sit directly on top of the
    # low/mid-Dice scatter it's most important to actually see.
    axis.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    figure.tight_layout()
    figure.savefig(out_dir / "dice_vs_volume_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--out-dir", default="workspace/figures/results_comparison")
    args = parser.parse_args()

    results_path = Path(args.results_csv)
    if not results_path.is_file():
        parser.error(f"results CSV not found: {results_path}")
    frame = pd.read_csv(results_path)
    required = {"experiment", "dice", "hd95_mm"}
    missing = required.difference(frame.columns)
    if missing:
        parser.error(f"results CSV is missing columns: {sorted(missing)}")

    # This is the raw-trainer comparison -- ensemble combos (from
    # ensembling/ensemble_test.py, "ensemble_A+B+C+D" rows) belong only in
    # the finalist-selection/ensembling analysis
    # (analysis/finalist_selection/, workspace/results/finalist_selection/),
    # not mixed into these figures. Excluding them here also keeps legend
    # text short -- a 4-member combo's name alone is long enough to squeeze
    # a whole figure down to a sliver (see plot_style.py's short_label).
    n_before = frame["experiment"].nunique()
    frame = frame[~frame["experiment"].astype(str).str.startswith("ensemble_")]
    n_after = frame["experiment"].nunique()
    if n_after < n_before:
        print(f"[filter] excluded {n_before - n_after} ensemble experiment(s) -- "
              "raw-trainer comparison only, see analysis/finalist_selection/ for ensembles")

    # val-only: test_id/test_ood results must not feed the raw-trainer
    # comparison figures. Repeatedly scoring test_id/test_ood across 9+
    # conditions during model selection was exactly the early-project
    # mistake this guards against -- test is for a single final check, not
    # an iterative comparison axis. Once the raw-trainer test_id/test_ood
    # rows are archived (pending a backup, per project decision), this
    # filter becomes a no-op rather than something to remember to redo.
    if "split" in frame.columns:
        n_rows_before = len(frame)
        frame = frame[frame["split"] == "val"]
        print(f"[filter] restricted to split=val: {len(frame)}/{n_rows_before} rows kept "
              "(test_id/test_ood excluded from the raw-trainer comparison)")
    else:
        print("[warn] no split column -- cannot restrict to val, results may include test_id/test_ood")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_overall(frame, out_dir)
    save_by_size(frame, out_dir)
    if "split" in frame.columns and frame["split"].nunique() > 1:
        save_by_split(frame, out_dir)
    else:
        print("[skip] dice_by_split.png: only one split (val) remains after the test-set filter above")
    save_by_center(frame, out_dir)
    save_volume_scatter(frame, out_dir)
    print(f"Figures written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
