#!/usr/bin/env python3
"""Convert an ATLAS R2.1 download into nnU-Net v2 format.

Despite the historical filename, this converter targets the ATLAS R2.1 layout
used by this course project::

    <raw_root>/Training_Raw/R0XX/sub-*/ses-*/anat/
        *_metadata.csv
        *_space-orig_desc-brain_T1w.nii.gz
        *_space-orig_label-lesion_desc-T1lesion_mask.nii.gz

It writes ``DatasetXXX_<name>`` beneath ``$nnUNet_raw`` plus a case metadata
CSV used by the sampling and evaluation commands.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metadata_utils import assign_size_bin, lesion_volume_mm3, sampling_weights_from_volume  # noqa: E402

MODALITIES = ["T1"]


def find_cases(raw_root: Path) -> dict[str, dict]:
    """Discover usable subject/session pairs in the ATLAS R2.1 tree."""
    cases: dict[str, dict] = {}
    search_root = raw_root / "Training_Raw" if (raw_root / "Training_Raw").is_dir() else raw_root

    for center_dir in sorted(search_root.glob("R*")):
        if not center_dir.is_dir():
            continue
        for sub_dir in sorted(center_dir.glob("sub-*")):
            for ses_dir in sorted(sub_dir.glob("ses-*")):
                anat_dir = ses_dir / "anat"
                if not anat_dir.is_dir():
                    continue

                subject = sub_dir.name.removeprefix("sub-")
                session = ses_dir.name.removeprefix("ses-")
                case_id = f"ATLAS_{subject}_ses{session}"

                images: list[Path] = []
                for modality in MODALITIES:
                    matches = sorted(anat_dir.glob(f"*{modality}w.nii.gz"))
                    if len(matches) != 1:
                        print(f"[skip] {case_id}: expected one {modality} image, found {len(matches)}")
                        images = []
                        break
                    images.append(matches[0])
                if not images:
                    continue

                masks = sorted(anat_dir.glob("*lesion*mask.nii.gz")) or sorted(anat_dir.glob("*mask.nii.gz"))
                if len(masks) != 1:
                    print(f"[skip] {case_id}: expected one lesion mask, found {len(masks)}")
                    continue

                metadata = sorted(anat_dir.glob("*metadata.csv"))
                if case_id in cases:
                    raise RuntimeError(f"Duplicate generated case ID: {case_id}")
                cases[case_id] = {
                    "images": images,
                    "label": masks[0],
                    "center": center_dir.name,
                    "metadata_csv": metadata[0] if metadata else None,
                }
    return cases


def build_metadata_dataframe(cases: dict[str, dict], label_paths: dict[str, str]) -> pd.DataFrame:
    rows: list[dict] = []
    supplemental: list[pd.DataFrame] = []

    for case_id, info in cases.items():
        rows.append(
            {
                "case_id": case_id,
                "lesion_volume_mm3": lesion_volume_mm3(label_paths[case_id]),
                "center": info["center"],
            }
        )
        metadata_csv = info["metadata_csv"]
        if metadata_csv is not None:
            try:
                frame = pd.read_csv(metadata_csv)
                if len(frame) != 1:
                    print(f"[warn] {case_id}: metadata has {len(frame)} rows; using the first")
                    frame = frame.head(1)
                frame.insert(0, "case_id", case_id)
                supplemental.append(frame)
            except Exception as exc:
                print(f"[warn] {case_id}: could not read {metadata_csv}: {exc}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["size_bin"] = assign_size_bin(df["lesion_volume_mm3"])
    df["sampling_weight"] = sampling_weights_from_volume(df["lesion_volume_mm3"])

    if supplemental:
        meta_df = pd.concat(supplemental, ignore_index=True, sort=False)
        duplicate_columns = [column for column in meta_df.columns if column in df.columns and column != "case_id"]
        meta_df = meta_df.drop(columns=duplicate_columns)
        df = df.merge(meta_df, on="case_id", how="left", validate="one_to_one")
    return df


def write_nnunet_dataset(cases: dict[str, dict], dataset_dir: Path, dataset_name: str) -> dict[str, str]:
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    labels: dict[str, str] = {}
    for case_id, files in cases.items():
        for channel, source in enumerate(files["images"]):
            shutil.copy2(source, images_tr / f"{case_id}_{channel:04d}.nii.gz")
        target = labels_tr / f"{case_id}.nii.gz"
        shutil.copy2(files["label"], target)
        labels[case_id] = str(target)

    dataset_json = {
        "channel_names": {str(index): modality for index, modality in enumerate(MODALITIES)},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "name": dataset_name,
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8")
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--dataset-id", type=int, default=1)
    parser.add_argument("--dataset-name", default="ATLAS")
    parser.add_argument("--out-metadata-csv", default="case_metadata.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    if not raw_root.is_dir():
        parser.error(f"Raw root does not exist or is not a directory: {raw_root}")

    cases = find_cases(raw_root)
    print(f"Found {len(cases)} usable cases under {raw_root}")
    if not cases:
        raise SystemExit("No usable cases found. Check the raw-root path and expected ATLAS R2.1 layout.")

    if args.dry_run:
        for case_id, info in list(cases.items())[:10]:
            print(f"  {case_id} center={info['center']} image={info['images'][0]} label={info['label']}")
        return 0

    nnunet_raw = os.environ.get("nnUNet_raw")
    if not nnunet_raw:
        raise SystemExit("nnUNet_raw is not set. Use `python isles26.py init` and run through the project runner.")

    dataset_dir = Path(nnunet_raw) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    if dataset_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Dataset folder already exists: {dataset_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(dataset_dir)

    label_paths = write_nnunet_dataset(cases, dataset_dir, args.dataset_name)
    metadata = build_metadata_dataframe(cases, label_paths)
    metadata_path = Path(args.out_metadata_csv).expanduser().resolve()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_path, index=False)

    print(f"Wrote nnU-Net dataset: {dataset_dir}")
    print(f"Wrote metadata: {metadata_path}")
    print(metadata["size_bin"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
