#!/usr/bin/env python3
"""Build a synthetic splits-dir that makes prepare_isles26_dataset.py copy the
FULL dev pool (train+val+test_id+test_ood) into imagesTr/labelsTr, by relabeling
every pooled case as "train" -- reuses prepare's existing, already-hardened
--splits-dir mechanism unmodified instead of writing a second, parallel
raw-dataset-copying implementation.

Expected input: `--pooled-manifest` (default
workspace/splits_dataset003_full/pooled_manifest.csv, written by
split_dataset_3fold_final.py) -- needs a `case_id` column at minimum.

What it produces (under `--out-dir`, default
workspace/splits_dataset003_full/prepare_splits_dir/):
- `train.csv` -- every pooled case, `case_id` column only (prepare only reads
  case_id from these files).
- `val.csv`, `test_id.csv`, `test_ood.csv` -- header-only, zero rows.
  prepare_isles26_dataset.py's load_split_manifest() requires all four files
  to exist (it raises otherwise) but is fine with any of them being empty.
- `final_holdout_id.csv`, `final_holdout_ood.csv` -- copied through UNCHANGED
  from the real workspace/final_holdout/ CSVs (default paths), not
  regenerated. prepare reads these only to confirm those case IDs are
  "spoken for" and excludes them from imagesTr -- passing the real files
  through means a case that somehow also appears in the raw tree under a
  final_holdout ID is still correctly excluded, same safety property prepare
  already had for the dev-time run.

Non-obvious rationale: prepare_isles26_dataset.py's filter_cases_by_split()
only ever includes cases labeled "train" or "val" into imagesTr/labelsTr
(see its own docstring: test_id/test_ood/final_holdout are excluded by
design, since that's what protects them during normal dev-time runs). There
is no "pool everything" mode. Rather than adding one to a script that has an
explicit, documented safety property (test_id/test_ood/final_holdout never
enter imagesTr), this script achieves the pooling by relabeling upstream --
prepare's own logic and its safety guarantees for OTHER splits/holdouts stay
completely intact and unmodified.

Usage:
    python data_prep/make_pooled_splits_dir.py
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pooled-manifest", default="workspace/splits_dataset003_full/pooled_manifest.csv", type=Path)
    ap.add_argument("--final-holdout-id-csv", default="workspace/final_holdout/final_holdout_id.csv", type=Path)
    ap.add_argument("--final-holdout-ood-csv", default="workspace/final_holdout/final_holdout_ood.csv", type=Path)
    ap.add_argument("--out-dir", default="workspace/splits_dataset003_full/prepare_splits_dir", type=Path)
    args = ap.parse_args()

    if not args.pooled_manifest.is_file():
        raise SystemExit(f"--pooled-manifest not found: {args.pooled_manifest} "
                          "(run data_prep/split_dataset_3fold_final.py first)")
    pooled = pd.read_csv(args.pooled_manifest)
    if "case_id" not in pooled.columns:
        raise SystemExit(f"{args.pooled_manifest}: missing case_id column")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pooled[["case_id"]].to_csv(args.out_dir / "train.csv", index=False)
    for split_name in ("val", "test_id", "test_ood"):
        pd.DataFrame(columns=["case_id"]).to_csv(args.out_dir / f"{split_name}.csv", index=False)
    print(f"Wrote {args.out_dir / 'train.csv'} ({len(pooled)} cases, relabeled from train+val+test_id+test_ood)")
    print(f"Wrote empty val.csv / test_id.csv / test_ood.csv placeholders "
          "(required to exist by prepare_isles26_dataset.py's load_split_manifest)")

    for label, source in (("final_holdout_id", args.final_holdout_id_csv),
                           ("final_holdout_ood", args.final_holdout_ood_csv)):
        if not source.is_file():
            raise SystemExit(f"--{label.replace('_', '-')}-csv not found: {source} -- refusing to proceed "
                              "without the real hidden-holdout list to exclude.")
        shutil.copy2(source, args.out_dir / f"{label}.csv")
        print(f"Copied {source} -> {args.out_dir / f'{label}.csv'} unchanged (exclusion list, not regenerated)")

    print(f"\nDone. Next: python isles26.py prepare --dataset-id 3 --dataset-name ATLAS_full "
          f"--splits-dir {args.out_dir} ...")


if __name__ == "__main__":
    main()
