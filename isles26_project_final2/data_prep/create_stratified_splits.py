#!/usr/bin/env python3
"""Create deterministic, subject-grouped, center/size-aware nnU-Net folds."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def subject_group(case_id: str) -> str:
    """Keep sessions from one ATLAS subject in the same fold."""
    return case_id.rsplit("_ses", 1)[0] if "_ses" in case_id else case_id


def _mode(values) -> str:
    counts = Counter(str(value) for value in values)
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def create_splits(metadata: pd.DataFrame, n_splits: int = 5, seed: int = 2026) -> list[dict[str, list[str]]]:
    required = {"case_id", "center", "size_bin"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    frame = metadata.dropna(subset=list(required)).copy()
    frame["case_id"] = frame["case_id"].astype(str)
    if frame.empty:
        raise ValueError("No complete metadata rows are available for splitting")
    if frame["case_id"].duplicated().any():
        raise ValueError("Metadata contains duplicate case IDs")
    frame["subject_group"] = frame["case_id"].map(subject_group)

    groups: list[dict] = []
    for group_id, rows in frame.groupby("subject_group", sort=True):
        centers = rows["center"].astype(str).unique()
        if len(centers) != 1:
            raise ValueError(f"Subject group {group_id} occurs in multiple centers: {sorted(centers)}")
        groups.append(
            {
                "group": str(group_id),
                "cases": sorted(rows["case_id"].tolist()),
                "stratum": f"{centers[0]}|{_mode(rows['size_bin'])}",
            }
        )
    if len(groups) < n_splits:
        raise ValueError(f"Need at least {n_splits} subject groups, found {len(groups)}")

    rng = random.Random(seed)
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for group in groups:
        by_stratum[group["stratum"]].append(group)
    validation_groups: list[list[dict]] = [[] for _ in range(n_splits)]
    fold_case_counts = [0] * n_splits
    for stratum in sorted(by_stratum):
        items = by_stratum[stratum]
        rng.shuffle(items)
        for item in sorted(items, key=lambda value: -len(value["cases"])):
            target = min(range(n_splits), key=lambda index: (fold_case_counts[index], index))
            validation_groups[target].append(item)
            fold_case_counts[target] += len(item["cases"])

    all_cases = set(frame["case_id"])
    splits: list[dict[str, list[str]]] = []
    for fold, fold_groups in enumerate(validation_groups):
        validation = sorted(case for group in fold_groups for case in group["cases"])
        if not validation:
            raise ValueError(f"Fold {fold} has no validation cases")
        training = sorted(all_cases.difference(validation))
        splits.append({"train": training, "val": validation})

    validation_counts = Counter(case for split in splits for case in split["val"])
    if set(validation_counts) != all_cases or any(count != 1 for count in validation_counts.values()):
        raise RuntimeError("Every case must occur in exactly one validation fold")
    group_to_fold: dict[str, int] = {}
    for fold, split in enumerate(splits):
        for case in split["val"]:
            group = subject_group(case)
            previous = group_to_fold.setdefault(group, fold)
            if previous != fold:
                raise RuntimeError(f"Subject group {group} was split across folds")
    return splits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-summary-csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata_path = Path(args.metadata_csv)
    if not metadata_path.is_file():
        parser.error(f"Metadata CSV not found: {metadata_path}")
    output_path = Path(args.out_json)
    if output_path.exists() and not args.overwrite:
        parser.error(f"Split file already exists: {output_path}. Pass --overwrite to replace it.")
    metadata = pd.read_csv(metadata_path)
    try:
        splits = create_splits(metadata, args.n_splits, args.seed)
    except ValueError as exc:
        parser.error(str(exc))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    summary_path = Path(args.out_summary_csv) if args.out_summary_csv else output_path.with_name("splits_summary.csv")
    case_to_fold = {
        case_id: fold
        for fold, split in enumerate(splits)
        for case_id in split["val"]
    }
    summary = metadata[["case_id", "center", "size_bin"]].copy()
    summary["fold"] = summary["case_id"].astype(str).map(case_to_fold)
    summary = (
        summary.groupby(["fold", "center", "size_bin"], observed=True)
        .size()
        .rename("n_cases")
        .reset_index()
        .sort_values(["fold", "center", "size_bin"])
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {len(splits)} deterministic folds to {output_path}")
    print(f"Wrote fold composition to {summary_path}")
    for fold, split in enumerate(splits):
        print(f"fold {fold}: train={len(split['train'])}, val={len(split['val'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
