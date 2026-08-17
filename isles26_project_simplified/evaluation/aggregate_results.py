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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_csvs", nargs="+")
    parser.add_argument("--out-combined", default="results.csv")
    parser.add_argument("--out-summary", default="summary_by_experiment.csv")
    parser.add_argument("--out-summary-by-size", default="summary_by_size_bin.csv")
    parser.add_argument("--out-summary-by-split", default="summary_by_split.csv")
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
        print("\nBy split (train/val/test_id/test_ood):")
        print(by_split.to_string())
    else:
        print("[note] split is absent; no split-stratified summary was written (use manifest.csv as --case-metadata-csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
