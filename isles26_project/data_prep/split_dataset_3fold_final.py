#!/usr/bin/env python3
"""Stratified 3-fold split over the FULL dev pool (train+val+test_id+test_ood)
-- the final-model training split, pooling everything previously held out for
dev-time comparison now that model/architecture selection is done (see
PROJECT_PLAN.md "3-fold full-data final run" entry). NOT the same thing as
split_dataset_3fold_shortlist.py, which deliberately pools only train+val for
a val-only CV upgrade -- this script's whole point is the opposite: fold in
test_id/test_ood too, because there is no more dev-time comparison left to
protect them for.

Expected input: `--train-csv`/`--val-csv`/`--test-id-csv`/`--test-ood-csv`
(defaults: workspace/splits_dataset002/{train,val,test_id,test_ood}.csv),
each with `case_id, center, lesion_volume_mm3, size_bin` columns -- exactly
what `data_prep/split_dataset.py` already produces. Deliberately takes ONLY
these four as input, with no argument for final_holdout_id/final_holdout_ood
at all -- structurally impossible to accidentally fold in the hidden
final-evaluation set, mirroring split_dataset_3fold_shortlist.py's same
safety pattern (see CLAUDE.md "Split finalized" / PROJECT_PLAN.md for why
final_holdout must stay untouched until the project's actual final check).

What it produces (under `--out-dir`, default `workspace/splits_dataset003_full/`):
- `splits_final.json` -- nnU-Net's native multi-fold format (3 folds by
  default), for direct installation into an nnUNet_raw dataset dir (see
  `scripts/run_3fold_full_experiment.sh`, which does this automatically --
  never hand-copy this over Dataset002's splits_final.json, which every
  already-trained single-fold checkpoint's fold_0 identity still depends on;
  this is meant for a brand-new dataset ID only).
- `fold_<i>_train.csv` / `fold_<i>_val.csv` per fold, for human review.
- `fold_balance_summary.csv` -- per fold, per size_bin count/fraction in
  each of train/val, plus site-diversity diagnostics (same shape as
  split_dataset_3fold_shortlist.py's).
- `pooled_manifest.csv` -- the full 1,212-ish-case pool with an
  `original_split` column (train/val/test_id/test_ood) preserved, purely
  for later provenance/debugging -- NOT used by nnU-Net itself.

Non-obvious rationale:
- Stratification is by size_bin only, NOT site-grouped -- a deliberate
  choice, not an oversight. The original test_ood existed to measure
  cross-site generalization *during model selection*; that job is done.
  Once every case here is pooled into training, the external
  final_holdout_ood carve-out (entire never-seen sites) is the only
  remaining cross-site check, and it happens completely outside this
  script/dataset. Site-grouping these 3 folds would only reduce each fold's
  training site diversity for no corresponding benefit.
- The `empty` size_bin (near-zero-volume cases) is merged into `small` for
  the stratification key ONLY when it has fewer than `--k` members --
  same precedent as split_dataset_3fold_shortlist.py and the original
  split_dataset.py.

Usage:
    python data_prep/split_dataset_3fold_final.py
    python data_prep/split_dataset_3fold_final.py --k 3 --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

STRATIFY_MERGE = {"empty": "small"}
REQUIRED_COLUMNS = ("case_id", "center", "lesion_volume_mm3", "size_bin")


def load_pool(csvs: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for split_name, path in csvs.items():
        if not path.is_file():
            raise SystemExit(f"--{split_name}-csv not found: {path}")
        frame = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise SystemExit(f"{path}: missing expected column(s) {missing}")
        frame = frame.copy()
        frame["original_split"] = split_name
        frames.append(frame)
    pool = pd.concat(frames, ignore_index=True)
    dupes = pool["case_id"][pool["case_id"].duplicated()].tolist()
    if dupes:
        raise SystemExit(
            f"case_id present in more than one input split: {dupes} -- refusing to pool "
            "an already-inconsistent set of splits."
        )
    return pool


def make_folds(pool: pd.DataFrame, k: int, seed: int) -> list[dict[str, list[str]]]:
    strat_key = pool["size_bin"].replace(STRATIFY_MERGE)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(pool, strat_key):
        train_ids = sorted(pool.iloc[train_idx]["case_id"].tolist())
        val_ids = sorted(pool.iloc[val_idx]["case_id"].tolist())
        folds.append({"train": train_ids, "val": val_ids})
    return folds


def balance_summary(pool: pd.DataFrame, folds: list[dict[str, list[str]]]) -> pd.DataFrame:
    by_case = pool.set_index("case_id")
    rows = []
    for i, fold in enumerate(folds):
        val_ids = fold["val"]
        val_df = by_case.loc[val_ids]
        train_df = by_case.loc[fold["train"]]
        for subset_name, subset_df in (("val", val_df), ("train", train_df)):
            for bin_name, count in subset_df["size_bin"].value_counts().items():
                rows.append({
                    "fold": i, "subset": subset_name, "size_bin": bin_name, "n": count,
                    "frac_of_subset": count / len(subset_df),
                })
        val_sites = set(val_df["center"])
        train_sites = set(train_df["center"])
        rows.append({"fold": i, "subset": "diagnostic", "size_bin": "n_unique_sites_val",
                      "n": len(val_sites), "frac_of_subset": float("nan")})
        rows.append({"fold": i, "subset": "diagnostic", "size_bin": "n_sites_only_in_val",
                      "n": len(val_sites - train_sites), "frac_of_subset": float("nan")})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-csv", default="workspace/splits_dataset002/train.csv", type=Path)
    ap.add_argument("--val-csv", default="workspace/splits_dataset002/val.csv", type=Path)
    ap.add_argument("--test-id-csv", default="workspace/splits_dataset002/test_id.csv", type=Path)
    ap.add_argument("--test-ood-csv", default="workspace/splits_dataset002/test_ood.csv", type=Path)
    ap.add_argument("--out-dir", default="workspace/splits_dataset003_full", type=Path)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = load_pool({
        "train": args.train_csv, "val": args.val_csv,
        "test_id": args.test_id_csv, "test_ood": args.test_ood_csv,
    })
    print(f"Pooled dev set: {len(pool)} cases from "
          f"{pool.groupby('original_split', observed=True).size().to_dict()}, "
          f"{pool['center'].nunique()} sites. size_bin: {pool['size_bin'].value_counts().to_dict()}")

    empty_n = (pool["size_bin"] == "empty").sum()
    if 0 < empty_n < args.k:
        print(f"Note: {empty_n} 'empty' case(s) merged into 'small' for stratification only "
              f"(too few for {args.k}-way stratified split) -- real size_bin label untouched in outputs.")

    folds = make_folds(pool, args.k, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "splits_final.json").write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")
    pool.to_csv(args.out_dir / "pooled_manifest.csv", index=False)

    by_case = pool.set_index("case_id")
    for i, fold in enumerate(folds):
        by_case.loc[fold["train"]].reset_index().to_csv(args.out_dir / f"fold_{i}_train.csv", index=False)
        by_case.loc[fold["val"]].reset_index().to_csv(args.out_dir / f"fold_{i}_val.csv", index=False)
        print(f"Fold {i}: {len(fold['train'])} train / {len(fold['val'])} val")

    summary = balance_summary(pool, folds)
    summary.to_csv(args.out_dir / "fold_balance_summary.csv", index=False)

    print("\n=== Size-bin balance per fold's val subset (should track the pool's overall proportions) ===")
    val_summary = summary[summary["subset"] == "val"]
    print(val_summary.pivot_table(index="fold", columns="size_bin", values="frac_of_subset").round(3).to_string())

    print(f"\nWrote: {args.out_dir / 'splits_final.json'} ({args.k} folds)")
    print(f"Wrote: {args.out_dir / 'pooled_manifest.csv'} ({len(pool)} cases, original_split preserved)")
    print(f"Wrote: {args.out_dir / 'fold_balance_summary.csv'}")
    print(
        "\nNext step (not done by this script): run "
        "data_prep/make_pooled_splits_dir.py to build a prepare-compatible splits-dir, then "
        "scripts/run_3fold_full_experiment.sh to prepare/preprocess a NEW dataset ID and install "
        f"this {args.out_dir / 'splits_final.json'} over its auto-generated single-fold placeholder. "
        "final_holdout_id/final_holdout_ood were never read by this script and must stay that way."
    )


if __name__ == "__main__":
    main()
