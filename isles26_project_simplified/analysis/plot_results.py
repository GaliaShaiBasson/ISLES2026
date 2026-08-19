#!/usr/bin/env python3
"""Generate report figures from the aggregated results CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_overall(df: pd.DataFrame, out_dir: Path) -> None:
    metric_cols = [c for c in ("dice", "hd95_mm", "lesion_f1") if c in df.columns]
    summary = df.groupby("experiment", observed=True)[metric_cols].mean()

    figure, axis = plt.subplots(figsize=(8, 4.5))
    summary["dice"].plot(kind="bar", ax=axis, legend=False)
    axis.set_title("Mean Dice by experiment")
    axis.set_xlabel("")
    axis.set_ylabel("Dice")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(out_dir / "overall_dice.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    summary["hd95_mm"].plot(kind="bar", ax=axis, legend=False)
    axis.set_title("Mean HD95 by experiment (lower is better)")
    axis.set_xlabel("")
    axis.set_ylabel("HD95 (mm)")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(out_dir / "overall_hd95.png", dpi=180)
    plt.close(figure)

    if "lesion_f1" in summary.columns:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        summary["lesion_f1"].plot(kind="bar", ax=axis, legend=False)
        axis.set_title("Mean lesion-wise F1 by experiment")
        axis.set_xlabel("")
        axis.set_ylabel("Lesion-wise F1")
        axis.tick_params(axis="x", rotation=30)
        figure.tight_layout()
        figure.savefig(out_dir / "overall_lesion_f1.png", dpi=180)
        plt.close(figure)
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
    plot_data["size_bin"] = pd.Categorical(
        plot_data["size_bin"], categories=["small", "medium", "large"], ordered=True
    )
    groups = []
    labels = []
    for size_bin in ["small", "medium", "large"]:
        for experiment in sorted(plot_data["experiment"].unique()):
            values = plot_data.loc[
                (plot_data["size_bin"] == size_bin) & (plot_data["experiment"] == experiment), "dice"
            ].dropna()
            if not values.empty:
                groups.append(values.to_numpy())
                labels.append(f"{size_bin}\n{experiment}")
    figure, axis = plt.subplots(figsize=(max(9, len(groups) * 1.2), 5))
    axis.boxplot(groups, tick_labels=labels, showmeans=True)
    axis.set_title("Dice by lesion-size bin and experiment")
    axis.set_ylabel("Dice")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(out_dir / "dice_by_size_bin.png", dpi=180)
    plt.close(figure)


def save_by_split(df: pd.DataFrame, out_dir: Path) -> None:
    """train/val/test_id/test_ood -- the ID-vs-OOD generalization comparison this
    project's split was built to produce (see CLAUDE.md). Mirrors save_by_size's
    boxplot-per-group-per-experiment layout.
    """
    if "split" not in df.columns:
        print("[skip] split column is absent (use manifest.csv as --case-metadata-csv when evaluating)")
        return
    plot_data = df.dropna(subset=["split", "experiment", "dice"]).copy()
    if plot_data.empty:
        print("[skip] no split data is available")
        return
    split_order = ["train", "val", "test_id", "test_ood"]
    present_splits = [s for s in split_order if s in plot_data["split"].unique()]
    plot_data["split"] = pd.Categorical(plot_data["split"], categories=present_splits, ordered=True)
    groups = []
    labels = []
    for split_name in present_splits:
        for experiment in sorted(plot_data["experiment"].unique()):
            values = plot_data.loc[
                (plot_data["split"] == split_name) & (plot_data["experiment"] == experiment), "dice"
            ].dropna()
            if not values.empty:
                groups.append(values.to_numpy())
                labels.append(f"{split_name}\n{experiment}")
    figure, axis = plt.subplots(figsize=(max(9, len(groups) * 1.2), 5))
    axis.boxplot(groups, tick_labels=labels, showmeans=True)
    axis.set_title("Dice by split (train/val/test_id/test_ood)")
    axis.set_ylabel("Dice")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(out_dir / "dice_by_split.png", dpi=180)
    plt.close(figure)


def save_by_center(df: pd.DataFrame, out_dir: Path) -> None:
    if "center" not in df.columns or df["center"].dropna().empty:
        print("[skip] center metadata is absent")
        return
    summary = df.groupby(["center", "experiment"], observed=True)["dice"].agg(["mean", "std"])
    means = summary["mean"].unstack("experiment")
    errors = summary["std"].unstack("experiment").fillna(0)
    figure, axis = plt.subplots(figsize=(10, 5))
    means.plot(kind="bar", yerr=errors, capsize=3, ax=axis)
    axis.set_title("Dice by acquisition center")
    axis.set_xlabel("Center")
    axis.set_ylabel("Dice")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(out_dir / "dice_by_center.png", dpi=180)
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
    figure, axis = plt.subplots(figsize=(8, 6))
    for experiment, group in plot_data.groupby("experiment", observed=True):
        points = axis.scatter(group["lesion_volume_mm3"], group["dice"], alpha=0.35, label=experiment)
        trend = _rolling_dice_trend(group["lesion_volume_mm3"], group["dice"])
        if trend is not None:
            axis.plot(*trend, color=points.get_facecolor()[0], linewidth=2)
    axis.set_xscale("log")
    axis.set_title("Dice versus lesion volume (points + rolling-median trend)")
    axis.set_xlabel("Lesion volume (mm³, log scale)")
    axis.set_ylabel("Dice")
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_dir / "dice_vs_volume_scatter.png", dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    results_path = Path(args.results_csv)
    if not results_path.is_file():
        parser.error(f"results CSV not found: {results_path}")
    frame = pd.read_csv(results_path)
    required = {"experiment", "dice", "hd95_mm"}
    missing = required.difference(frame.columns)
    if missing:
        parser.error(f"results CSV is missing columns: {sorted(missing)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_overall(frame, out_dir)
    save_by_size(frame, out_dir)
    save_by_split(frame, out_dir)
    save_by_center(frame, out_dir)
    save_volume_scatter(frame, out_dir)
    print(f"Figures written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
