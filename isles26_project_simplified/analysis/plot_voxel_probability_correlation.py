#!/usr/bin/env python3
"""Heatmap figure for `voxel_probability_correlation.py`'s output -- the
probability-level twin of `plot_finalist_selection.py`'s
`dice_correlation_heatmap.png`. Pure plotting: reads an already-computed CSV,
writes a PNG, does no correlation computation itself.

Expected input: `voxel_probability_correlation.csv` (written by
`voxel_probability_correlation.py`), an NxN Pearson-r matrix indexed/columned
by trainer.

What it produces (under `--out-dir`, default matching `--in-csv`'s
directory): `voxel_probability_correlation.png`.

Usage:
    python analysis/plot_voxel_probability_correlation.py
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
from plot_finalist_selection import short_name, BLUE_SEQUENTIAL  # noqa: E402


def plot_voxel_correlation(corr: pd.DataFrame, out_dir: Path) -> None:
    labels = [short_name(t) for t in corr.columns]
    vmin = max(0.0, float(np.nanmin(corr.values[~np.eye(len(corr), dtype=bool)])) - 0.05)
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
            ax.text(j, i, f"{val:.2f}" if not np.isnan(val) else "n/a", ha="center", va="center",
                    fontsize=7, color=text_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r, per-voxel foreground probability")
    ax.set_title("Per-voxel probability correlation between trainers (val)\n"
                  "lower = more complementary confidence, finer than the per-case Dice view", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "voxel_probability_correlation.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-csv", default="workspace/evaluation/finalist_selection/voxel_probability_correlation.csv", type=Path)
    ap.add_argument("--out-dir", default=None, type=Path, help="Default: same directory as --in-csv.")
    args = ap.parse_args()

    if not args.in_csv.exists():
        raise SystemExit(f"{args.in_csv} not found -- run voxel_probability_correlation.py first.")
    out_dir = args.out_dir or args.in_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    corr = pd.read_csv(args.in_csv, index_col=0)
    plot_voxel_correlation(corr, out_dir)
    print(f"Wrote: {out_dir / 'voxel_probability_correlation.png'}")


if __name__ == "__main__":
    main()
