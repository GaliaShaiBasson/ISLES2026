#!/usr/bin/env python3
"""Combine per-experiment result CSVs and create summary tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"case_id", "experiment", "dice", "hd95_mm"}
# lesion_f1 is optional, not required: older result CSVs predate lesion-wise scoring and
# would otherwise fail the column check on re-aggregation.
OPTIONAL_METRIC_COLUMNS = ["lesion_f1"]


def site_weighted_summary(combined: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    """Macro-average by center: mean-of-per-site-means, not the pooled case-weighted mean.

    Case-weighted (plain groupby(["experiment","split"]).mean()) silently assumes every
    case is an independent, exchangeable sample -- untrue when cases cluster by center
    (shared scanner/protocol/annotator). If one center supplies most of a split's cases,
    the pooled mean mostly reflects that one center, not "generalization" broadly. This
    treats each center as one observation regardless of how many cases it contributed --
    the correct comparison against test_id when test_ood's cases are lopsided across its
    (few) centers. See CLAUDE.md "site-clustering" entry for the concrete example (a
    2026-08-18 3-centers-carry-80%-of-test_ood case that flipped test_id vs test_ood
    ranking once centers were weighted equally).

    n_sites is the real sample size for this statistic -- keep it visible; a 2-3-center
    split-weighted mean is barely more trustworthy than the pooled one it's meant to fix.
    """
    per_site = combined.groupby(["experiment", "split", "center"], observed=True)[metric_cols].mean()
    macro = per_site.groupby(["experiment", "split"], observed=True)[metric_cols].agg(["mean", "std", "median"]).round(4)
    n_sites = per_site.groupby(["experiment", "split"], observed=True).size().rename("n_sites")
    macro.columns = pd.MultiIndex.from_tuples([(c, f"site_{s}") for c, s in macro.columns])
    macro["n_sites"] = n_sites
    return macro


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_csvs", nargs="+")
    parser.add_argument("--out-combined", default="results.csv")
    parser.add_argument("--out-summary", default="summary_by_experiment.csv")
    parser.add_argument("--out-summary-by-size", default="summary_by_size_bin.csv")
    parser.add_argument("--out-summary-by-split", default="summary_by_split.csv")
    parser.add_argument(
        "--out-summary-by-split-siteweighted",
        default="summary_by_split_siteweighted.csv",
        help="Macro-averaged (mean-of-per-site-means) version of --out-summary-by-split -- "
        "the fairer test_id-vs-test_ood comparison when centers contribute very uneven "
        "case counts. See site_weighted_summary()'s docstring.",
    )
    args = parser.parse_args()

    frames = []
    for csv_path in args.result_csvs:
        frame = pd.read_csv(csv_path)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            parser.error(f"{csv_path} is missing columns: {sorted(missing)}")
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    duplicate_keys = combined.duplicated(subset=["case_id", "experiment"])
    if duplicate_keys.any():
        examples = combined.loc[duplicate_keys, ["case_id", "experiment"]].head().to_dict("records")
        parser.error(f"duplicate case/experiment rows found, for example: {examples}")

    output_paths = [
        Path(args.out_combined),
        Path(args.out_summary),
        Path(args.out_summary_by_size),
        Path(args.out_summary_by_split),
        Path(args.out_summary_by_split_siteweighted),
    ]
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    combined.to_csv(output_paths[0], index=False)
    metric_cols = ["dice", "hd95_mm"] + [c for c in OPTIONAL_METRIC_COLUMNS if c in combined.columns]
    overall = combined.groupby("experiment")[metric_cols].agg(["mean", "std", "median", "count"]).round(4)
    overall.to_csv(output_paths[1])
    print(f"Combined {len(combined)} rows from {len(frames)} experiments -> {output_paths[0]}")
    print(overall.to_string())

    if "size_bin" in combined.columns:
        by_size = combined.groupby(["experiment", "size_bin"], observed=True)[metric_cols].agg(["mean", "std", "median", "count"]).round(4)
        by_size.to_csv(output_paths[2])
    else:
        print("[note] size_bin is absent; no size-stratified summary was written")

    # "split" (train/val/test_id/test_ood) is what actually answers the generalization
    # question this project is built around -- test_id vs. test_ood Dice, side by side.
    if "split" in combined.columns:
        by_split = combined.groupby(["experiment", "split"], observed=True)[metric_cols].agg(["mean", "std", "median", "count"]).round(4)
        by_split.to_csv(output_paths[3])
        print("\nBy split (train/val/test_id/test_ood), case-weighted (pooled):")
        print(by_split.to_string())

        if "center" in combined.columns:
            by_split_site = site_weighted_summary(combined, metric_cols)
            by_split_site.to_csv(output_paths[4])
            print("\nBy split, site-weighted (mean-of-per-site-means -- see docstring; "
                  "prefer this over the pooled table above for test_id-vs-test_ood claims):")
            print(by_split_site.to_string())
        else:
            print("[note] center is absent; no site-weighted split summary was written")
    else:
        print("[note] split is absent; no split-stratified summary was written (use manifest.csv as --case-metadata-csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
