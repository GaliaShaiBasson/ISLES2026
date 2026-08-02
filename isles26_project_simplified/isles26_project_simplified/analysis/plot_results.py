#!/usr/bin/env python3
"""Generate report figures from the aggregated results CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_overall(df: pd.DataFrame, out_dir: Path) -> None:
    summary = df.groupby("experiment", observed=True)[["dice", "hd95_mm"]].mean()

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


def save_volume_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    if "lesion_volume_mm3" not in df.columns:
        print("[skip] lesion_volume_mm3 column is absent")
        return
    plot_data = df.copy()
    plot_data["lesion_volume_mm3"] = pd.to_numeric(plot_data["lesion_volume_mm3"], errors="coerce")
    plot_data = plot_data[plot_data["lesion_volume_mm3"] > 0]
    if plot_data.empty:
        print("[skip] no positive lesion volumes are available for log-scale plotting")
        return
    figure, axis = plt.subplots(figsize=(8, 6))
    for experiment, group in plot_data.groupby("experiment", observed=True):
        axis.scatter(group["lesion_volume_mm3"], group["dice"], alpha=0.65, label=experiment)
    axis.set_xscale("log")
    axis.set_title("Dice versus lesion volume")
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
    save_by_center(frame, out_dir)
    save_volume_scatter(frame, out_dir)
    print(f"Figures written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
