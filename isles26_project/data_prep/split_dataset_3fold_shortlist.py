#!/usr/bin/env python3
"""Stratified k-fold split (default k=3) over the dev pool ONLY, for the
val-shortlist CV upgrade discussed in project chat 2026-08-20 -- replaces a
single 119-case val split's bootstrap CI with real across-fold variance for
whichever 2-4 trainers survive the val-based redundancy analysis
(analysis/select_finalist_from_val.py, pca_model_redundancy.py).

Expected input: `--train-csv`/`--val-csv` (default
`workspace/splits_dataset002/{train,val}.csv`), each with `case_id, center,
lesion_volume_mm3, size_bin` columns -- exactly the files
`data_prep/split_dataset.py` already produces. Deliberately takes ONLY these
two as input, with no argument for test_id/test_ood/final_holdout at all --
this makes it structurally impossible to accidentally fold in held-out
data, rather than relying on a runtime check to catch it.

What it produces (under `--out-dir`, default
`workspace/splits_shortlist_3fold/`):
- `splits_final.json` -- nnU-Net's native multi-fold format (a list of
  `{"train": [...], "val": [...]}`, one entry per fold, same schema
  `prepare_isles26_dataset.py:write_splits_final_json` already writes for
  the single-fold case) so nnU-Net's own `-f 0/1/2` fold mechanism can be
  used directly once wired into a dataset -- see "Not done here" below.
- `fold_<i>_train.csv` / `fold_<i>_val.csv` per fold, for human review.
- `fold_balance_summary.csv` -- per fold, per size_bin count/fraction in
  each of train/val, plus site-diversity diagnostics (unique sites in val,
  and any site whose cases land ENTIRELY in one fold's val split -- not
  disqualifying by itself, since this is a within-distribution CV split,
  not a site-holdout one like test_ood, but worth a look if a fold's val
  behaves like an accidental mini site-holdout).

Method: `sklearn.model_selection.StratifiedKFold(n_splits=k, shuffle=True,
random_state=seed)` on `size_bin`. Stratifying on size_bin (not center) is
deliberate and matches `split_dataset.py`'s own precedent -- the *why* for
CV is different from why test_ood stratifies by site (test_ood exists to
measure cross-center generalization; this k-fold's whole job is a more
robust val-set model-selection signal on the same kind of in-distribution
data the original val split already represents, so site-holdout inside it
would defeat that purpose, not strengthen it).

Non-obvious rationale: the dev pool's `empty` size_bin has only 2 cases
total (1 train + 1 val) -- too few for scikit-learn's StratifiedKFold to
split into 3 groups (`n_splits` can't exceed a class's member count). Per
the exact precedent already logged in CLAUDE.md ("Split finalized 2026-08-17":
empty-lesion cases "merged into the 'small' bin for split mechanics only"),
`empty` cases are merged into `small` ONLY for the stratification key --
their real `size_bin` label is preserved unchanged in every output file.

Not done here (explicitly out of scope for this script): actually wiring
these folds into a trainable nnU-Net dataset. That needs a real
imagesTr/preprocessed-dataset decision (new dataset ID vs. swapping the
live Dataset002 splits_final.json in place, which would break resuming any
already-trained single-fold checkpoint's fold_0 identity) -- a real
architectural choice to make deliberately, not a byproduct of a split
script. See the printed "Next step" note at the end of a run.

Usage:
    python data_prep/split_dataset_3fold_shortlist.py
    python data_prep/split_dataset_3fold_shortlist.py --k 5 --seed 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

STRATIFY_MERGE = {"empty": "small"}  # stratification-key-only merge, see docstring


def load_pool(train_csv: Path, val_csv: Path) -> pd.DataFrame:
    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)
    for df, name in ((train, train_csv), (val, val_csv)):
        missing = [c for c in ("case_id", "center", "lesion_volume_mm3", "size_bin") if c not in df.columns]
        if missing:
            raise SystemExit(f"{name}: missing expected column(s) {missing}")
    pool = pd.concat([train, val], ignore_index=True)
    dupes = pool["case_id"][pool["case_id"].duplicated()].tolist()
    if dupes:
        raise SystemExit(f"case_id present in both train and val: {dupes} -- refusing to fold an already-inconsistent pool")
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
        for bin_name, count in val_df["size_bin"].value_counts().items():
            rows.append({
                "fold": i, "subset": "val", "size_bin": bin_name, "n": count,
                "frac_of_subset": count / len(val_df),
            })
        for bin_name, count in train_df["size_bin"].value_counts().items():
            rows.append({
                "fold": i, "subset": "train", "size_bin": bin_name, "n": count,
                "frac_of_subset": count / len(train_df),
            })
        val_sites = set(val_df["center"])
        train_sites = set(train_df["center"])
        val_only_sites = val_sites - train_sites
        rows.append({
            "fold": i, "subset": "diagnostic", "size_bin": "n_unique_sites_val", "n": len(val_sites),
            "frac_of_subset": float("nan"),
        })
        rows.append({
            "fold": i, "subset": "diagnostic", "size_bin": "n_sites_only_in_val", "n": len(val_only_sites),
            "frac_of_subset": float("nan"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-csv", default="workspace/splits_dataset002/train.csv", type=Path)
    ap.add_argument("--val-csv", default="workspace/splits_dataset002/val.csv", type=Path)
    ap.add_argument("--out-dir", default="workspace/splits_shortlist_3fold", type=Path)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = load_pool(args.train_csv, args.val_csv)
    print(f"Dev pool: {len(pool)} cases ({pool['case_id'].isin(pd.read_csv(args.train_csv)['case_id']).sum()} from train, "
          f"{pool['case_id'].isin(pd.read_csv(args.val_csv)['case_id']).sum()} from val), "
          f"{pool['center'].nunique()} sites. size_bin: {pool['size_bin'].value_counts().to_dict()}")

    empty_n = (pool["size_bin"] == "empty").sum()
    if 0 < empty_n < args.k:
        print(f"Note: {empty_n} 'empty' case(s) merged into 'small' for stratification only "
              f"(too few for {args.k}-way stratified split) -- real size_bin label untouched in outputs.")

    folds = make_folds(pool, args.k, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "splits_final.json").write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")

    by_case = pool.set_index("case_id")
    for i, fold in enumerate(folds):
        by_case.loc[fold["train"]].reset_index().to_csv(args.out_dir / f"fold_{i}_train.csv", index=False)
        by_case.loc[fold["val"]].reset_index().to_csv(args.out_dir / f"fold_{i}_val.csv", index=False)
        print(f"Fold {i}: {len(fold['train'])} train / {len(fold['val'])} val")

    summary = balance_summary(pool, folds)
    summary.to_csv(args.out_dir / "fold_balance_summary.csv", index=False)

    print("\n=== Size-bin balance per fold's val subset (should track the pool's overall proportions) ===")
    val_summary = summary[(summary["subset"] == "val")]
    print(val_summary.pivot_table(index="fold", columns="size_bin", values="frac_of_subset").round(3).to_string())

    print("\n=== Site diagnostics (informational -- this is a within-distribution CV split, not a site-holdout) ===")
    diag = summary[summary["subset"] == "diagnostic"]
    print(diag.pivot_table(index="fold", columns="size_bin", values="n").astype(int).to_string())

    print(f"\nWrote: {args.out_dir / 'splits_final.json'} ({args.k} folds)")
    print(f"Wrote: {args.out_dir / 'fold_balance_summary.csv'}")
    print(
        "\nNext step (not done by this script -- a deliberate dataset-wiring decision, see docstring): "
        "these folds still need to be attached to a trainable nnU-Net dataset. The live "
        "nnUNet_preprocessed/Dataset002_ATLAS/splits_final.json must NOT be overwritten in place -- "
        "it's still the single-fold split every already-trained checkpoint's fold_0 identity depends on."
    )


if __name__ == "__main__":
    main()
