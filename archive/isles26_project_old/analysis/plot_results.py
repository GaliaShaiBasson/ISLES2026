#!/usr/bin/env python
"""
Produce the figures you'll actually want for the report, from `results.csv`
(output of evaluation/aggregate_results.py).

Usage:
    python plot_results.py --results-csv results.csv --out-dir figures/

Figures produced:
    1. overall_dice_hd95.png        - bar chart, mean Dice & HD95 per experiment
    2. dice_by_size_bin.png         - the key plot: Dice per method, split by
                                       small/medium/large lesion (does the
                                       improvement actually help where it's
                                       supposed to?)
    3. dice_by_center.png           - per-center performance, if 'center'
                                       metadata is available
    4. dice_vs_volume_scatter.png   - Dice vs raw lesion volume, one point per
                                       case, colored by experiment (best for
                                       spotting the small-lesion failure mode
                                       directly, not just binned)
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_overall(df: pd.DataFrame, out_dir: Path):
    summary = df.groupby("experiment")[["dice", "hd95_mm"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(data=summary, x="experiment", y="dice", ax=axes[0])
    axes[0].set_title("Mean Dice by experiment")
    axes[0].set_ylabel("Dice")
    axes[0].tick_params(axis="x", rotation=30)

    sns.barplot(data=summary, x="experiment", y="hd95_mm", ax=axes[1])
    axes[1].set_title("Mean HD95 by experiment (lower is better)")
    axes[1].set_ylabel("HD95 (mm)")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(out_dir / "overall_dice_hd95.png", dpi=150)
    plt.close(fig)


def plot_by_size_bin(df: pd.DataFrame, out_dir: Path):
    if "size_bin" not in df.columns:
        print("[skip] no size_bin column, skipping dice_by_size_bin.png")
        return
    order = [b for b in ["small", "medium", "large"] if b in df["size_bin"].unique()]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="size_bin", y="dice", hue="experiment", order=order, ax=ax)
    ax.set_title("Dice by lesion size bin, per method\n(this is the plot that shows *why* an improvement helps)")
    ax.set_xlabel("Lesion size bin")
    ax.set_ylabel("Dice")
    fig.tight_layout()
    fig.savefig(out_dir / "dice_by_size_bin.png", dpi=150)
    plt.close(fig)


def plot_by_center(df: pd.DataFrame, out_dir: Path):
    if "center" not in df.columns:
        print("[skip] no center column, skipping dice_by_center.png")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="center", y="dice", hue="experiment", ax=ax, errorbar="sd")
    ax.set_title("Dice by center, per method (domain-generalization sanity check)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "dice_by_center.png", dpi=150)
    plt.close(fig)


def plot_dice_vs_volume(df: pd.DataFrame, out_dir: Path):
    if "lesion_volume_mm3" not in df.columns:
        print("[skip] no lesion_volume_mm3 column, skipping scatter plot")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df, x="lesion_volume_mm3", y="dice", hue="experiment", alpha=0.6, ax=ax
    )
    ax.set_xscale("log")
    ax.set_title("Dice vs. lesion volume (log scale), per method")
    ax.set_xlabel("Lesion volume (mm^3, log scale)")
    ax.set_ylabel("Dice")
    fig.tight_layout()
    fig.savefig(out_dir / "dice_vs_volume_scatter.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.results_csv)

    plot_overall(df, out_dir)
    plot_by_size_bin(df, out_dir)
    plot_by_center(df, out_dir)
    plot_dice_vs_volume(df, out_dir)

    print(f"Figures written to {out_dir}/")


if __name__ == "__main__":
    main()
