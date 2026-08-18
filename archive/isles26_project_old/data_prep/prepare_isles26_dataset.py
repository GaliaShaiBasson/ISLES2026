#!/usr/bin/env python
"""
Convert a raw ISLES'26 download into nnU-Net v2's expected raw dataset
layout, and build `case_metadata.csv` (lesion volume, size bin, clinical
metadata) used later for lesion-aware sampling and stratified evaluation.

RAW_LAYOUT (assumption -- adjust the glob patterns below to match what you
actually download; ISLES challenge packaging varies year to year):

    <raw_root>/
        sub-0001/
            sub-0001_T1w.nii.gz          <- one modality assumed here (T1)
            sub-0001_lesion-mask.nii.gz
        sub-0002/
            ...
        participants.tsv (or .csv)        <- optional clinical metadata,
                                              columns e.g.:
                                              subject_id, center, chronicity,
                                              days_post_stroke

If your download instead ships multiple modalities (e.g. T1 + FLAIR + DWI),
extend `MODALITIES` and the glob patterns accordingly -- nnU-Net handles
multi-modal input natively via the `_0000`, `_0001`, ... suffix convention,
which this script already writes.

Output:

    $nnUNet_raw/Dataset0XX_ISLES26/
        imagesTr/ISLES26_XXX_0000.nii.gz
        labelsTr/ISLES26_XXX.nii.gz
        dataset.json
    <this_project>/case_metadata.csv
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from metadata_utils import build_case_metadata  # noqa: E402

# Adjust if your download includes more than one MRI sequence.
MODALITIES = ["T1"]


#!/usr/bin/env python
"""
Convert a raw ATLAS R2.1 download into nnU-Net v2's expected raw dataset
layout, and build `case_metadata.csv` (lesion volume, size bin, center,
and any per-subject metadata) used later for lesion-aware sampling and
stratified evaluation.

RAW_LAYOUT (matches ATLAS R2.1 as shipped, e.g. Training_Raw):

    <raw_root>/
        Training_Raw/
            R001/                                      <- site/cohort code -> used as "center"
                sub-r001s001/
                    ses-1/
                        anat/
                            sub-r001s001_ses-1_metadata.csv
                            sub-r001s001_ses-1_space-orig_desc-brain_T1w.nii.gz
                            sub-r001s001_ses-1_space-orig_label-lesion_desc-T1lesion_mask.nii.gz
                sub-r001s002/
                    ...
            R002/
                ...

Each session gets its own metadata.csv (not one global participants file).
This script reads every one it finds and concatenates them (outer join, so
it's fine if columns vary slightly across sites), joined to `case_id`.

Output:

    $nnUNet_raw/Dataset0XX_ATLAS/
        imagesTr/ATLAS_r001s001_ses1_0000.nii.gz
        labelsTr/ATLAS_r001s001_ses1.nii.gz
        dataset.json
    <this_project>/case_metadata.csv   (includes a `center` column = R0XX)
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent))
from metadata_utils import lesion_volume_mm3, assign_size_bin, sampling_weights_from_volume  # noqa: E402

# Adjust if a different sequence should be used (ATLAS is T1w-only).
MODALITIES = ["T1"]


def find_cases(raw_root: Path):
    """Discover subject/session folders under Training_Raw/R0XX/sub-*/ses-*/anat/.

    Returns a dict: case_id -> {
        "images": [path_per_modality], "label": path,
        "center": "R001", "metadata_csv": path or None
    }
    """
    cases = {}
    training_raw = raw_root / "Training_Raw"
    search_root = training_raw if training_raw.is_dir() else raw_root

    for center_dir in sorted(search_root.glob("R*")):
        if not center_dir.is_dir():
            continue
        center_id = center_dir.name  # e.g. "R001"

        for sub_dir in sorted(center_dir.glob("sub-*")):
            for ses_dir in sorted(sub_dir.glob("ses-*")):
                anat_dir = ses_dir / "anat"
                if not anat_dir.is_dir():
                    continue

                subj_id = sub_dir.name.replace("sub-", "")   # "r001s001"
                ses_id = ses_dir.name.replace("ses-", "")    # "1"
                case_id = f"ATLAS_{subj_id}_ses{ses_id}"

                images = []
                for mod in MODALITIES:
                    matches = list(anat_dir.glob(f"*{mod}w.nii.gz"))
                    if not matches:
                        print(f"[warn] no {mod} image found for {sub_dir.name}/{ses_dir.name}, skipping")
                        images = None
                        break
                    images.append(matches[0])
                if images is None:
                    continue

                mask_matches = list(anat_dir.glob("*mask.nii.gz"))
                if not mask_matches:
                    print(f"[warn] no lesion mask found for {sub_dir.name}/{ses_dir.name}, skipping")
                    continue

                meta_matches = list(anat_dir.glob("*metadata.csv"))

                cases[case_id] = {
                    "images": images,
                    "label": mask_matches[0],
                    "center": center_id,
                    "metadata_csv": meta_matches[0] if meta_matches else None,
                }
    return cases


def build_metadata_dataframe(cases: dict, label_paths: dict) -> pd.DataFrame:
    """Build case_metadata.csv rows: lesion volume/size bin/sampling weight
    (from the mask), `center` (from the R0XX folder), and whatever columns
    each session's own metadata.csv provides (read as-is and concatenated
    with an outer join, so differing columns across sites don't break it)."""
    rows = []
    per_case_meta = []
    for case_id, info in cases.items():
        vol = lesion_volume_mm3(label_paths[case_id])
        rows.append({"case_id": case_id, "lesion_volume_mm3": vol, "center": info["center"]})

        if info["metadata_csv"] is not None:
            try:
                m = pd.read_csv(info["metadata_csv"])
                m.insert(0, "case_id", case_id)
                per_case_meta.append(m)
            except Exception as e:
                print(f"[warn] could not read metadata csv for {case_id} ({info['metadata_csv']}): {e}")

    df = pd.DataFrame(rows)
    df["size_bin"] = assign_size_bin(df["lesion_volume_mm3"])
    df["sampling_weight"] = sampling_weights_from_volume(df["lesion_volume_mm3"])

    if per_case_meta:
        meta_df = pd.concat(per_case_meta, ignore_index=True, sort=False)
        # A session's own metadata.csv might repeat "center" under a
        # different name -- keep our folder-derived one as ground truth,
        # drop obvious duplicate columns from the per-session files.
        dupe_cols = [c for c in meta_df.columns if c in df.columns and c != "case_id"]
        meta_df = meta_df.drop(columns=dupe_cols)
        df = df.merge(meta_df, on="case_id", how="left")

    return df


def write_nnunet_dataset(cases: dict, dataset_dir: Path, dataset_name: str):
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    label_paths_out = {}
    for case_id, files in cases.items():
        for mod_idx, img_path in enumerate(files["images"]):
            out_name = f"{case_id}_{mod_idx:04d}.nii.gz"
            shutil.copy(img_path, images_tr / out_name)
        label_out = labels_tr / f"{case_id}.nii.gz"
        shutil.copy(files["label"], label_out)
        label_paths_out[case_id] = str(label_out)

    dataset_json = {
        "channel_names": {str(i): mod for i, mod in enumerate(MODALITIES)},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "name": dataset_name,
    }
    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    return label_paths_out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", required=True, help="Path to the raw ATLAS download root (the folder containing Training_Raw/)")
    parser.add_argument("--dataset-id", type=int, default=1, help="nnU-Net numeric dataset ID, e.g. 1 -> Dataset001_ATLAS")
    parser.add_argument("--dataset-name", default="ATLAS", help="Short name used in the nnU-Net dataset folder name")
    parser.add_argument("--out-metadata-csv", default="case_metadata.csv", help="Where to write the combined case metadata CSV")
    parser.add_argument("--dry-run", action="store_true", help="Only report how many cases were found, write nothing")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    nnunet_raw = os.environ.get("nnUNet_raw")
    if not args.dry_run and not nnunet_raw:
        sys.exit("nnUNet_raw environment variable is not set. Set it before running (see requirements.txt).")

    cases = find_cases(raw_root)
    print(f"Found {len(cases)} usable cases under {raw_root}")
    if args.dry_run:
        for cid in list(cases)[:10]:
            info = cases[cid]
            print(f"  {cid} [center={info['center']}]: {[str(p) for p in info['images']]} | {info['label']} | meta={info['metadata_csv']}")
        return

    dataset_dir_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    dataset_dir = Path(nnunet_raw) / dataset_dir_name
    label_paths = write_nnunet_dataset(cases, dataset_dir, args.dataset_name)
    print(f"Wrote nnU-Net raw dataset to {dataset_dir}")

    meta_df = build_metadata_dataframe(cases, label_paths)
    meta_df.to_csv(args.out_metadata_csv, index=False)
    print(f"Wrote case metadata ({len(meta_df)} rows) to {args.out_metadata_csv}")
    print(meta_df["size_bin"].value_counts())
    print(meta_df["center"].value_counts())


if __name__ == "__main__":
    main()
