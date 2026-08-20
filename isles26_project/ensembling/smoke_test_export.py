#!/usr/bin/env python3
"""Cheap end-to-end sanity check for the val-probability export pipeline,
BEFORE committing GPU/CPU time to the full 5-trainer x ~120-case run.

What it does: stages just `--n-cases` val images (default 2) for exactly
one trainer, runs `nnUNetv2_predict --save_probabilities --device cpu` on
them into a throwaway `workspace/predictions/smoketest_predVal_prob/` folder (never
touches the real predVal_prob/ path or any existing result), then scores
the resulting mask against ground truth and compares it against the
ALREADY-TRUSTED Dice for that exact case+trainer already sitting in
`workspace/results/runs/.../results_val.csv`.

Why this is the right check: if the newly-computed Dice matches the
already-scored one (should be exact or near-exact -- nnU-Net's sliding-
window + mirroring TTA inference is deterministic, no dropout/randomness
at inference time), that confirms staging, CPU prediction, and scoring are
all wired correctly end to end -- without waiting for or paying for the
full run to find out.

Usage:
    python ensembling/smoke_test_export.py
    python ensembling/smoke_test_export.py --trainer nnUNetTrainerFocalTversky_500epochs --n-cases 3
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from compute_metrics import evaluate_case  # noqa: E402

NNUNET_RAW = Path("/home/galia/ISLES2026/nnUNet_raw")
NNUNET_PREPROCESSED = Path("/home/galia/ISLES2026/nnUNet_preprocessed")
NNUNET_RESULTS = Path("/home/galia/ISLES2026/nnUNet_results")
DATASET_NAME = "Dataset002_ATLAS"
DATASET_ID = "2"
PREDICT = "/home/galia/miniconda3/envs/isles2026/bin/nnUNetv2_predict"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trainer", default="nnUNetTrainerWideAugBaseline_500epochs")
    ap.add_argument("--n-cases", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dice-tolerance", type=float, default=0.005)
    args = ap.parse_args()

    run_dirs = sorted(Path("workspace/results/runs").glob(f"{args.trainer}__3d_fullres__fold0__*"))
    if not run_dirs:
        raise SystemExit(f"No existing results_val.csv run dir found for {args.trainer} -- can't sanity-check against it.")
    known = pd.read_csv(run_dirs[0] / "results_val.csv").set_index("case_id")

    case_ids = known.index[: args.n_cases].tolist()
    print(f"Smoke-testing {args.trainer} on {len(case_ids)} case(s): {case_ids}")

    stage_dir = Path("workspace/predictions/val_images_staged_smoketest")
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    images_dir = NNUNET_RAW / DATASET_NAME / "imagesTr"
    for case_id in case_ids:
        src = images_dir / f"{case_id}_0000.nii.gz"
        if not src.exists():
            raise SystemExit(f"Missing source image for {case_id}: {src}")
        (stage_dir / f"{case_id}_0000.nii.gz").symlink_to(src.resolve())

    out_dir = Path("workspace/predictions/smoketest_predVal_prob") / args.trainer
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    env_note = f"OMP_NUM_THREADS={args.threads} MKL_NUM_THREADS={args.threads}"
    print(f"Running nnUNetv2_predict --device cpu ({env_note}) -> {out_dir} ...")
    import os
    env = dict(os.environ)
    env["nnUNet_extTrainer"] = "/home/galia/ISLES2026/isles26_project"
    env["nnUNet_raw"] = str(NNUNET_RAW)
    env["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    env["nnUNet_results"] = str(NNUNET_RESULTS)
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["MKL_NUM_THREADS"] = str(args.threads)
    cmd = [
        PREDICT, "-i", str(stage_dir), "-o", str(out_dir),
        "-d", DATASET_ID, "-c", "3d_fullres", "-tr", args.trainer, "-f", "0",
        "--save_probabilities", "-device", "cpu",
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise SystemExit(f"nnUNetv2_predict failed with exit code {result.returncode}")
    print("Prediction finished, verifying outputs...")

    gt_dir = NNUNET_RAW / DATASET_NAME / "labelsTr"
    all_pass = True
    for case_id in case_ids:
        npz_path = out_dir / f"{case_id}.npz"
        pkl_path = out_dir / f"{case_id}.pkl"
        mask_path = out_dir / f"{case_id}.nii.gz"
        if not (npz_path.exists() and pkl_path.exists() and mask_path.exists()):
            print(f"  [FAIL] {case_id}: missing output file(s)")
            all_pass = False
            continue
        probs = np.load(npz_path)["probabilities"]
        shape_ok = probs.ndim == 4 and probs.shape[0] == 2
        range_ok = 0.0 <= probs.min() and probs.max() <= 1.0
        sums_to_one = np.allclose(probs.sum(axis=0), 1.0, atol=1e-3)

        new_metrics = evaluate_case(mask_path, gt_dir / f"{case_id}.nii.gz")
        new_dice = new_metrics["dice"]
        known_dice = known.loc[case_id, "dice"]
        dice_diff = abs(new_dice - known_dice)
        dice_ok = dice_diff <= args.dice_tolerance

        status = "PASS" if (shape_ok and range_ok and sums_to_one and dice_ok) else "FAIL"
        all_pass &= (status == "PASS")
        print(f"  [{status}] {case_id}: probs shape={probs.shape} range_ok={range_ok} "
              f"sums_to_one={sums_to_one} | new_dice={new_dice:.4f} known_dice={known_dice:.4f} "
              f"diff={dice_diff:.4f} (tolerance {args.dice_tolerance})")

    print()
    if all_pass:
        print("=== SMOKE TEST PASSED: staging -> CPU predict -> scoring pipeline matches the trusted "
              "val results. Safe to run export_val_probabilities.sh for real. ===")
    else:
        print("=== SMOKE TEST FAILED: see [FAIL] lines above. Do NOT launch the full export run yet. ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
