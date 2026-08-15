#!/usr/bin/env python3
"""Combine per-experiment metrics and create stratified summary tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"case_id", "experiment", "dice", "hd95_mm"}
METRICS = ["dice", "hd95_mm"]
AGGREGATIONS = ["mean", "std", "median", "count"]


def summarize(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return frame.groupby(group_columns, observed=True)[METRICS].agg(AGGREGATIONS).round(4)


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

    paths = {
        "combined": Path(args.out_combined),
        "overall": Path(args.out_summary),
        "size": Path(args.out_summary_by_size),
        "center": Path(args.out_summary_by_center),
        "size_center": Path(args.out_summary_by_size_center),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    combined.to_csv(paths["combined"], index=False)
    overall = summarize(combined, ["experiment"])
    overall.to_csv(paths["overall"])
    print(f"Combined {len(combined)} rows from {len(frames)} experiment files -> {paths['combined']}")
    print(overall.to_string())

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
