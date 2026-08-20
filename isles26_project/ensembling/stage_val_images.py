#!/usr/bin/env python3
"""Symlink the val-split case images into a scratch folder nnUNetv2_predict
can point at, so val-set softmax probabilities can be exported without ever
touching the existing `fold_0/validation/` output or `imagesTr/` itself.

Expected input layout: `--manifest` (default `workspace/splits_dataset002/manifest.csv`,
the same file `evaluate` already reads) with `case_id`/`split` columns, and
`--images-dir` (default `nnUNet_raw/Dataset002_ATLAS/imagesTr`, nnU-Net's flat
`<case_id>_0000.nii.gz` naming) holding the actual case files.

What it produces: `--out-dir` (default `workspace/predictions/val_images_staged/`)
populated with one symlink per val case, `<case_id>_0000.nii.gz` ->
the real file in `--images-dir`. Symlinked, not copied, so this costs no
extra disk for potentially large volumes and can never desync from the
source. Idempotent: existing correct symlinks are left alone; `--overwrite`
replaces stale ones (e.g. pointing at a moved images-dir).

Non-obvious rationale: nnU-Net's own `--val --npz` retraining flag would
also produce val-set probabilities, but it writes into the SAME
`fold_0/validation/` folder that already holds the scored prediction masks
-- overwriting files in place, even if the recomputation is deterministic
and should be byte-identical. This project's convention is to never write
over an existing result folder (see `postprocess_predictions.py`'s hard
guard, and how `predTs_prob/` was kept separate from `predTs/`) -- staging
a fresh input folder and predicting into a brand-new `predVal_prob/`
output dir (see `export_val_probabilities.sh`) avoids that risk entirely,
at the cost of one extra staging step.

Usage:
    python ensembling/stage_val_images.py
    python ensembling/stage_val_images.py --overwrite
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="workspace/splits_dataset002/manifest.csv", type=Path)
    ap.add_argument("--images-dir", default="/home/galia/ISLES2026/nnUNet_raw/Dataset002_ATLAS/imagesTr", type=Path)
    ap.add_argument("--out-dir", default="workspace/predictions/val_images_staged", type=Path)
    ap.add_argument("--channel-suffix", default="_0000", help="nnU-Net single-channel file suffix before the extension.")
    ap.add_argument("--overwrite", action="store_true", help="Replace existing symlinks instead of skipping them.")
    args = ap.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")
    manifest = pd.read_csv(args.manifest)
    if "split" not in manifest.columns or "case_id" not in manifest.columns:
        raise SystemExit(f"{args.manifest} must have case_id/split columns")
    val_cases = sorted(manifest.loc[manifest["split"] == "val", "case_id"].unique())
    if not val_cases:
        raise SystemExit(f"No split=='val' rows found in {args.manifest}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    linked, skipped, missing = 0, 0, []
    for case_id in val_cases:
        src = args.images_dir / f"{case_id}{args.channel_suffix}.nii.gz"
        dst = args.out_dir / f"{case_id}{args.channel_suffix}.nii.gz"
        if not src.exists():
            missing.append(case_id)
            continue
        if dst.exists() or dst.is_symlink():
            if args.overwrite:
                dst.unlink()
            else:
                skipped += 1
                continue
        dst.symlink_to(src.resolve())
        linked += 1

    print(f"Staged {linked} new symlink(s), skipped {skipped} already-staged, "
          f"{len(missing)} missing source file(s) into {args.out_dir}")
    if missing:
        print(f"  MISSING (not staged): {missing}")
    print(f"Total val cases in manifest: {len(val_cases)}")


if __name__ == "__main__":
    main()
