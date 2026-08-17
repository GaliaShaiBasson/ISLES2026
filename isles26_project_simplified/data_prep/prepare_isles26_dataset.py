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
from metadata_utils import assign_size_bin, lesion_volume_mm3, sampling_weights_from_size_bin  # noqa: E402

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


def load_split_manifest(splits_dir: Path) -> dict[str, str]:
    """Read train/val/test_id/test_ood CSVs from split_dataset.py's --out-dir.

    Returns {case_id: split_name}. Raises if a required split file is missing
    so a stale/partial splits-dir fails loudly instead of silently training on
    the wrong set of cases.
    """
    case_to_split: dict[str, str] = {}
    for split_name in ("train", "val", "test_id", "test_ood"):
        path = splits_dir / f"{split_name}.csv"
        if not path.is_file():
            raise SystemExit(
                f"Split manifest missing: {path}. Run data_prep/split_dataset.py first "
                f"(see CLAUDE.md decisions log)."
            )
        frame = pd.read_csv(path)
        for case_id in frame["case_id"]:
            case_to_split[str(case_id)] = split_name
    return case_to_split


def filter_cases_by_split(cases: dict[str, dict], case_to_split: dict[str, str]) -> dict[str, dict]:
    """Keep only train+val cases for the nnU-Net raw dataset (imagesTr/labelsTr).

    test_id/test_ood must never enter imagesTr: nnU-Net's own fold logic treats
    everything there as fair game for training, and it has no held-out-test
    concept of its own (see CLAUDE.md).
    """
    included: dict[str, dict] = {}
    skipped_test = 0
    skipped_unknown = 0
    for case_id, info in cases.items():
        split_name = case_to_split.get(case_id)
        if split_name in ("train", "val"):
            included[case_id] = info
        elif split_name in ("test_id", "test_ood"):
            skipped_test += 1
        else:
            # Discovered on disk but absent from the split manifest -- e.g. the raw
            # tree changed since the manifest was generated. Excluding (not silently
            # including) keeps imagesTr in sync with a known, reviewed split.
            skipped_unknown += 1
            print(f"[skip] {case_id}: not present in split manifest, excluding from imagesTr")
    print(
        f"Split filter: {len(included)} train+val cases included, "
        f"{skipped_test} test_id/test_ood cases held out, {skipped_unknown} unmatched cases excluded"
    )
    return included


def write_splits_final_json(cases: dict[str, dict], case_to_split: dict[str, str], dataset_dir: Path) -> Path:
    """Write nnU-Net's splits_final.json format (single fold) from the split manifest.

    nnU-Net's do_split() uses this file verbatim if present instead of generating
    its own unstratified random 5-fold split -- see CLAUDE.md. Written next to
    dataset.json in the nnU-Net raw dataset dir; isles26.py's preprocess step
    copies it into nnUNet_preprocessed/<dataset>/ once that folder exists.
    """
    train_ids = sorted(cid for cid in cases if case_to_split.get(cid) == "train")
    val_ids = sorted(cid for cid in cases if case_to_split.get(cid) == "val")
    path = dataset_dir / "splits_final.json"
    path.write_text(json.dumps([{"train": train_ids, "val": val_ids}], indent=2) + "\n", encoding="utf-8")
    return path


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
    df["sampling_weight"] = sampling_weights_from_size_bin(df["size_bin"])

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


def write_holdout_test_set(all_cases: dict[str, dict], case_to_split: dict[str, str], dataset_dir: Path) -> int:
    """Write test_id + test_ood cases to imagesTs/labelsTs (nnU-Net's standard test-set layout).

    train/val never see these (see write_nnunet_dataset / filter_cases_by_split), but they still
    need to live *somewhere* in nnU-Net's expected flat naming for nnUNetv2_predict to run on and
    for evaluate/aggregate to score against -- otherwise there's no way to ever measure the
    ID/OOD generalization split this project was built to measure. Both test_id and test_ood are
    combined into one imagesTs/labelsTs (matching nnU-Net convention); downstream evaluate/
    aggregate distinguish them via the "split" column carried through split_dataset.py's
    manifest.csv rather than via separate folders.
    """
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"
    images_ts.mkdir(parents=True, exist_ok=True)
    labels_ts.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for case_id, info in all_cases.items():
        if case_to_split.get(case_id) not in ("test_id", "test_ood"):
            continue
        for channel, source in enumerate(info["images"]):
            shutil.copy2(source, images_ts / f"{case_id}_{channel:04d}.nii.gz")
        shutil.copy2(info["label"], labels_ts / f"{case_id}.nii.gz")
        n_written += 1
    return n_written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--dataset-id", type=int, default=1)
    parser.add_argument("--dataset-name", default="ATLAS")
    parser.add_argument("--out-metadata-csv", default="case_metadata.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--splits-dir",
        default=None,
        help=(
            "Directory containing train/val/test_id/test_ood.csv from "
            "data_prep/split_dataset.py. When set, only train+val cases are written "
            "to imagesTr/labelsTr, and splits_final.json is generated from train/val "
            "so nnU-Net's fold 0 uses this split instead of its own random 5-fold. "
            "When omitted, all discovered cases are written (legacy/no-split behavior)."
        ),
    )
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

    case_to_split: dict[str, str] | None = None
    all_cases = cases
    if args.splits_dir:
        case_to_split = load_split_manifest(Path(args.splits_dir).expanduser().resolve())
        cases = filter_cases_by_split(cases, case_to_split)
        if not cases:
            raise SystemExit("No train+val cases remain after applying the split manifest.")

    nnunet_raw = os.environ.get("nnUNet_raw")
    if not nnunet_raw:
        raise SystemExit("nnUNet_raw is not set. Use `python isles26.py init` and run through the project runner.")

    dataset_dir = Path(nnunet_raw) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    if dataset_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Dataset folder already exists: {dataset_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(dataset_dir)

    label_paths = write_nnunet_dataset(cases, dataset_dir, args.dataset_name)
    if case_to_split is not None:
        splits_path = write_splits_final_json(cases, case_to_split, dataset_dir)
        print(f"Wrote splits_final.json: {splits_path}")
        n_holdout = write_holdout_test_set(all_cases, case_to_split, dataset_dir)
        print(f"Wrote {n_holdout} held-out cases (test_id + test_ood) to imagesTs/labelsTs")
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
