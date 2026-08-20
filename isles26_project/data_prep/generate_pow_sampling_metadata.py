#!/usr/bin/env python3
"""Generate a dedicated case_metadata.csv with power-law bin-level sampling weights.

Expected input: an existing case_metadata.csv (from `isles26.py prepare`) containing
at least `case_id`, `lesion_volume_mm3`, and `size_bin` columns.

Produces: a copy of that CSV with `sampling_weight` recomputed via
`metadata_utils.sampling_weights_from_size_bin_power` (weight per bin proportional to
`(1 / total_lesion_volume_in_bin) ** p`) instead of the fixed 4:2:1 ratio in
`sampling_weights_from_size_bin`. All other columns are passed through unchanged.

Written to a separate file (default: workspace/case_metadata/case_metadata_pow_p05.csv), never
overwriting the input -- this is specifically so
`nnUNetTrainerLesionAwareSamplingPow_250epochs` can run as an independent comparison
against the original `nnUNetTrainerLesionAwareSampling_250epochs` without either one
reading or clobbering the other's metadata. See CLAUDE.md decisions log for why p=0.5
(sqrt-dampened between uniform and full voxel-mass equalization) was chosen over the
original fixed ratio.

Usage:
    python data_prep/generate_pow_sampling_metadata.py \
        --in-csv workspace/case_metadata/case_metadata.csv \
        --out-csv workspace/case_metadata/case_metadata_pow_p05.csv \
        --p 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metadata_utils import sampling_weights_from_size_bin_power  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-csv", default="workspace/case_metadata/case_metadata.csv")
    parser.add_argument("--out-csv", default="workspace/case_metadata/case_metadata_pow_p05.csv")
    parser.add_argument("--p", type=float, default=0.5)
    args = parser.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)
    if not in_path.is_file():
        parser.error(f"Input metadata CSV not found: {in_path}")
    if in_path.resolve() == out_path.resolve():
        parser.error("--out-csv must differ from --in-csv (never overwrite the source metadata)")

    df = pd.read_csv(in_path)
    required = {"case_id", "lesion_volume_mm3", "size_bin"}
    missing = required.difference(df.columns)
    if missing:
        parser.error(f"{in_path} is missing required columns: {sorted(missing)}")

    df["sampling_weight"] = sampling_weights_from_size_bin_power(df["size_bin"], df["lesion_volume_mm3"], p=args.p)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path} (p={args.p})")
    by_bin = df.groupby("size_bin")["sampling_weight"].sum().sort_values(ascending=False)
    print("\nShare of total sampling probability by bin:")
    print(by_bin.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
