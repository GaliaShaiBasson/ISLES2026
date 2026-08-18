#!/usr/bin/env python
"""
Combine multiple `results_<experiment>.csv` files (one per trained model,
from compute_metrics.py) into a single `results.csv` plus summary tables,
ready for `analysis/plot_results.py`.

Usage:
    python aggregate_results.py results_baseline.csv results_focal.csv \
        results_tversky.csv results_focal_tversky.csv results_lesion_aware.csv \
        --out-combined results.csv --out-summary summary_by_experiment.csv \
        --out-summary-by-size summary_by_size_bin.csv
"""
import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_csvs", nargs="+", help="Per-experiment CSVs from compute_metrics.py")
    parser.add_argument("--out-combined", default="results.csv")
    parser.add_argument("--out-summary", default="summary_by_experiment.csv")
    parser.add_argument("--out-summary-by-size", default="summary_by_size_bin.csv")
    args = parser.parse_args()

    dfs = [pd.read_csv(p) for p in args.result_csvs]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(args.out_combined, index=False)
    print(f"Combined {len(combined)} rows across {len(dfs)} experiments -> {args.out_combined}")

    overall = (
        combined.groupby("experiment")[["dice", "hd95_mm"]]
        .agg(["mean", "std", "median"])
        .round(4)
    )
    overall.to_csv(args.out_summary)
    print(f"\n=== Overall summary ===\n{overall}")
    overall.to_csv(args.out_summary)

    if "size_bin" in combined.columns:
        by_size = (
            combined.groupby(["experiment", "size_bin"])[["dice", "hd95_mm"]]
            .agg(["mean", "std", "count"])
            .round(4)
        )
        by_size.to_csv(args.out_summary_by_size)
        print(f"\n=== By lesion size bin ===\n{by_size}")
    else:
        print("\n[note] no 'size_bin' column found -- skipping size-stratified summary.")


if __name__ == "__main__":
    main()
