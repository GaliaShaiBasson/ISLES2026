#!/usr/bin/env python3
"""Combine metrics, enforce paired cohorts, and write statistical summaries."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REQUIRED_COLUMNS = {"case_id", "experiment", "dice", "hd95_mm"}
AGGREGATIONS = ["mean", "std", "median", "count"]


def metric_columns(frame: pd.DataFrame) -> list[str]:
    hd95 = "hd95_penalized_mm" if "hd95_penalized_mm" in frame.columns else "hd95_mm"
    return ["dice", hd95]


def summarize(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    summary = frame.groupby(group_columns, observed=True)[metric_columns(frame)].agg(AGGREGATIONS)
    grouped = frame.groupby(group_columns, observed=True)
    if "pred_empty" in frame.columns:
        summary[("pred_empty", "rate")] = grouped["pred_empty"].mean()
    if "hd95_defined" in frame.columns:
        summary[("hd95_defined", "rate")] = grouped["hd95_defined"].mean()
    return summary.round(4)


def validate_paired_cohorts(combined: pd.DataFrame, baseline: str) -> list[str]:
    experiments = [str(value) for value in combined["experiment"].drop_duplicates()]
    if baseline not in experiments:
        raise ValueError(f"Baseline experiment {baseline!r} is absent; found {experiments}")
    reference = set(combined.loc[combined["experiment"] == baseline, "case_id"].astype(str))
    problems: list[str] = []
    for experiment in experiments:
        cases = set(combined.loc[combined["experiment"] == experiment, "case_id"].astype(str))
        if cases != reference:
            problems.append(
                f"{experiment}: missing {len(reference - cases)}, extra {len(cases - reference)} "
                f"relative to {baseline}"
            )
    return problems


def _bootstrap_mean_ci(values: np.ndarray, iterations: int, rng: np.random.Generator) -> tuple[float, float]:
    if values.size == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, values.size, size=(iterations, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, float(p_values[index]) * (total - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def paired_comparisons(
    combined: pd.DataFrame,
    baseline: str = "baseline",
    bootstrap_iterations: int = 5000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Return paired improvements; positive values always mean better."""
    if bootstrap_iterations < 100:
        raise ValueError("bootstrap_iterations must be at least 100")
    metrics = metric_columns(combined)
    base = combined.loc[combined["experiment"] == baseline, ["case_id", *metrics]].set_index("case_id")
    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    for experiment in combined["experiment"].drop_duplicates():
        if experiment == baseline:
            continue
        candidate = combined.loc[combined["experiment"] == experiment, ["case_id", *metrics]].set_index("case_id")
        paired = base.join(candidate, how="inner", lsuffix="_baseline", rsuffix="_candidate", validate="one_to_one")
        for metric in metrics:
            paired_metric = paired[[f"{metric}_baseline", f"{metric}_candidate"]].dropna()
            if paired_metric.empty:
                continue
            if metric == "dice":
                improvement = (
                    paired_metric[f"{metric}_candidate"] - paired_metric[f"{metric}_baseline"]
                ).to_numpy(dtype=float)
                direction = "candidate_minus_baseline"
            else:
                improvement = (
                    paired_metric[f"{metric}_baseline"] - paired_metric[f"{metric}_candidate"]
                ).to_numpy(dtype=float)
                direction = "baseline_minus_candidate"
            ci_low, ci_high = _bootstrap_mean_ci(improvement, bootstrap_iterations, rng)
            p_value = 1.0 if np.allclose(improvement, 0) else float(
                wilcoxon(improvement, zero_method="wilcox", alternative="two-sided").pvalue
            )
            rows.append(
                {
                    "baseline": baseline,
                    "candidate": experiment,
                    "metric": metric,
                    "n_pairs": int(improvement.size),
                    "mean_improvement": float(improvement.mean()),
                    "median_improvement": float(np.median(improvement)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "difference_definition": direction,
                    "wilcoxon_p": p_value,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["wilcoxon_p_holm"] = np.nan
    for metric, indices in result.groupby("metric", observed=True).groups.items():
        adjusted = _holm_adjust(result.loc[indices, "wilcoxon_p"].tolist())
        result.loc[indices, "wilcoxon_p_holm"] = adjusted
    return result.round(6)


def write_optional_summary(
    combined: pd.DataFrame,
    group_columns: list[str],
    required_metadata: list[str],
    output_path: Path,
) -> None:
    missing = [column for column in required_metadata if column not in combined.columns]
    if missing:
        print(f"[skip] {output_path.name}: missing metadata columns {missing}")
        return
    usable = combined.dropna(subset=required_metadata)
    if usable.empty:
        print(f"[skip] {output_path.name}: no rows have {required_metadata}")
        return
    summarize(usable, group_columns).to_csv(output_path)
    print(f"Wrote {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_csvs", nargs="+")
    parser.add_argument("--out-combined", default="results.csv")
    parser.add_argument("--out-summary", default="summary_by_experiment.csv")
    parser.add_argument("--out-summary-by-size", default="summary_by_size_bin.csv")
    parser.add_argument("--out-summary-by-center", default="summary_by_center.csv")
    parser.add_argument("--out-summary-by-size-center", default="summary_by_size_and_center.csv")
    parser.add_argument("--out-paired", default="paired_comparisons.csv")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-unpaired", action="store_true", help="Allow unequal case sets (not recommended)")
    args = parser.parse_args()

    frames = []
    for csv_path in args.result_csvs:
        frame = pd.read_csv(csv_path)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            parser.error(f"{csv_path} is missing columns: {sorted(missing)}")
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined["case_id"] = combined["case_id"].astype(str)
    combined["experiment"] = combined["experiment"].astype(str)
    for metric in metric_columns(combined):
        combined[metric] = pd.to_numeric(combined[metric], errors="coerce")
        if combined[metric].isna().any() or not np.isfinite(combined[metric].to_numpy(dtype=float)).all():
            parser.error(f"primary metric {metric} contains missing or non-finite values")
    duplicate_keys = combined.duplicated(subset=["case_id", "experiment"])
    if duplicate_keys.any():
        examples = combined.loc[duplicate_keys, ["case_id", "experiment"]].head().to_dict("records")
        parser.error(f"duplicate case/experiment rows found, for example: {examples}")
    try:
        cohort_problems = validate_paired_cohorts(combined, args.baseline)
    except ValueError as exc:
        parser.error(str(exc))
    if cohort_problems and not args.allow_unpaired:
        parser.error("Experiments do not use the same evaluation cohort: " + "; ".join(cohort_problems))
    if cohort_problems:
        print("WARNING: unpaired cohorts allowed: " + "; ".join(cohort_problems))

    paths = {
        "combined": Path(args.out_combined),
        "overall": Path(args.out_summary),
        "size": Path(args.out_summary_by_size),
        "center": Path(args.out_summary_by_center),
        "size_center": Path(args.out_summary_by_size_center),
        "paired": Path(args.out_paired),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    combined.to_csv(paths["combined"], index=False)
    overall = summarize(combined, ["experiment"])
    overall.to_csv(paths["overall"])
    paired = paired_comparisons(combined, args.baseline, args.bootstrap_iterations, args.seed)
    paired.to_csv(paths["paired"], index=False)
    print(f"Combined {len(combined)} rows from {len(frames)} files -> {paths['combined']}")
    print(overall.to_string())
    print(f"Wrote {paths['paired']}")

    write_optional_summary(combined, ["experiment", "size_bin"], ["size_bin"], paths["size"])
    write_optional_summary(combined, ["experiment", "center"], ["center"], paths["center"])
    write_optional_summary(
        combined,
        ["experiment", "center", "size_bin"],
        ["center", "size_bin"],
        paths["size_center"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
