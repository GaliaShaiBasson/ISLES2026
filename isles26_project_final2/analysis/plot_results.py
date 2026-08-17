#!/usr/bin/env python3
"""Generate experiment figures with uncertainty and failure-rate reporting."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIZE_ORDER = ["small", "medium", "large"]


def hd95_column(df: pd.DataFrame) -> str:
    return "hd95_penalized_mm" if "hd95_penalized_mm" in df.columns else "hd95_mm"


def mean_ci(values: pd.Series) -> tuple[float, float, float]:
    data = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if data.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(data.mean())
    if data.size == 1:
        return mean, mean, mean
    error = 1.96 * float(data.std(ddof=1)) / np.sqrt(data.size)
    return mean, mean - error, mean + error


def save_mean_with_ci(df: pd.DataFrame, metric: str, title: str, ylabel: str, output: Path) -> None:
    labels: list[str] = []
    means: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for experiment, group in df.groupby("experiment", observed=True, sort=False):
        mean, low, high = mean_ci(group[metric])
        labels.append(str(experiment))
        means.append(mean)
        lower.append(mean - low)
        upper.append(high - mean)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(labels))
    axis.bar(x, means, yerr=np.asarray([lower, upper]), capsize=4, color="#4472C4")
    axis.set_xticks(x, labels=labels, rotation=30, ha="right")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xlabel("")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_overall(df: pd.DataFrame, out_dir: Path) -> None:
    save_mean_with_ci(df, "dice", "Mean Dice by experiment (95% normal CI)", "Dice", out_dir / "overall_dice.png")
    hd95 = hd95_column(df)
    label = "Penalized HD95 (mm)" if hd95 == "hd95_penalized_mm" else "HD95 (mm)"
    save_mean_with_ci(
        df,
        hd95,
        f"Mean {label} by experiment (lower is better)",
        label,
        out_dir / "overall_hd95.png",
    )


def save_empty_prediction_rate(df: pd.DataFrame, out_dir: Path) -> None:
    if "pred_empty" not in df.columns:
        print("[skip] pred_empty column is absent")
        return
    rates = df.groupby("experiment", observed=True, sort=False)["pred_empty"].mean()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    rates.plot(kind="bar", ax=axis, color="#C55A11", legend=False)
    axis.set_title("Empty-prediction rate by experiment")
    axis.set_ylabel("Fraction of cases")
    axis.set_xlabel("")
    axis.set_ylim(0, max(0.05, min(1.0, float(rates.max()) * 1.2)))
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(out_dir / "empty_prediction_rate.png", dpi=180)
    plt.close(figure)


def save_by_size(df: pd.DataFrame, out_dir: Path) -> None:
    if "size_bin" not in df.columns:
        print("[skip] size_bin column is absent")
        return
    plot_data = df.dropna(subset=["size_bin", "experiment", "dice"]).copy()
    if plot_data.empty:
        print("[skip] no size-bin data is available")
        return
    plot_data["size_bin"] = pd.Categorical(plot_data["size_bin"], categories=SIZE_ORDER, ordered=True)
    means = plot_data.groupby(["size_bin", "experiment"], observed=True)["dice"].mean().unstack("experiment")
    figure, axis = plt.subplots(figsize=(9, 5))
    means.plot(kind="bar", ax=axis)
    axis.set_title("Mean Dice by lesion-size bin")
    axis.set_xlabel("Lesion-size bin")
    axis.set_ylabel("Dice")
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(out_dir / "dice_by_size_bin.png", dpi=180)
    plt.close(figure)


def save_by_center(df: pd.DataFrame, out_dir: Path) -> None:
    if "center" not in df.columns or df["center"].dropna().empty:
        print("[skip] center metadata is absent")
        return
    plot_data = df.dropna(subset=["center", "experiment", "dice"])
    means = plot_data.groupby(["center", "experiment"], observed=True)["dice"].mean().unstack("experiment")
    figure, axis = plt.subplots(figsize=(max(10, len(means) * 0.8), 5))
    means.plot(kind="bar", ax=axis)
    axis.set_title("Mean Dice by acquisition center")
    axis.set_xlabel("Center")
    axis.set_ylabel("Dice")
    axis.tick_params(axis="x", rotation=45)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(out_dir / "dice_by_center.png", dpi=180)
    plt.close(figure)


def save_size_center_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    required = {"center", "size_bin", "experiment", "dice"}
    if not required.issubset(df.columns):
        print("[skip] center × size heatmaps require center and size_bin metadata")
        return
    plot_data = df.dropna(subset=list(required)).copy()
    if plot_data.empty:
        print("[skip] no center × size data is available")
        return
    plot_data["size_bin"] = pd.Categorical(plot_data["size_bin"], categories=SIZE_ORDER, ordered=True)

    for experiment, group in plot_data.groupby("experiment", observed=True):
        pivot = group.pivot_table(index="center", columns="size_bin", values="dice", aggfunc="mean", observed=True)
        counts = group.pivot_table(index="center", columns="size_bin", values="dice", aggfunc="count", observed=True)
        pivot = pivot.reindex(columns=SIZE_ORDER)
        counts = counts.reindex(index=pivot.index, columns=SIZE_ORDER).fillna(0)
        if pivot.empty:
            continue
        matrix = pivot.to_numpy(dtype=float)
        figure, axis = plt.subplots(figsize=(7, max(4, len(pivot) * 0.5)))
        image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                count = int(counts.iloc[row, column])
                label = "—" if np.isnan(value) else f"{value:.2f}\nn={count}"
                axis.text(column, row, label, ha="center", va="center", fontsize=8)
        axis.set_title(f"Dice by center and lesion size — {experiment}")
        axis.set_xlabel("Lesion-size bin")
        axis.set_ylabel("Center")
        axis.set_xticks(np.arange(len(pivot.columns)), labels=[str(x) for x in pivot.columns])
        axis.set_yticks(np.arange(len(pivot.index)), labels=[str(x) for x in pivot.index])
        figure.colorbar(image, ax=axis, label="Mean Dice")
        figure.tight_layout()
        safe_name = str(experiment).lower().replace(" ", "_").replace("/", "_")
        figure.savefig(out_dir / f"dice_center_by_size_{safe_name}.png", dpi=180)
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
        axis.scatter(group["lesion_volume_mm3"], group["dice"], alpha=0.55, label=experiment)
    axis.set_xscale("log")
    axis.set_title("Dice versus lesion volume")
    axis.set_xlabel("Lesion volume (mm³, log scale)")
    axis.set_ylabel("Dice")
    axis.legend()
    axis.grid(alpha=0.2)
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
    required = {"experiment", "dice"}
    missing = required.difference(frame.columns)
    if missing:
        parser.error(f"results CSV is missing columns: {sorted(missing)}")
    if "hd95_penalized_mm" not in frame.columns and "hd95_mm" not in frame.columns:
        parser.error("results CSV must contain hd95_penalized_mm or hd95_mm")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_overall(frame, out_dir)
    save_empty_prediction_rate(frame, out_dir)
    save_by_size(frame, out_dir)
    save_by_center(frame, out_dir)
    save_size_center_heatmaps(frame, out_dir)
    save_volume_scatter(frame, out_dir)
    print(f"Figures written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
