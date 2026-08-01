#!/usr/bin/env bash
# Phase 1: baseline nnU-Net v2, default trainer/loss (Dice + CE).
#
# Usage: ./run_baseline.sh <DATASET_ID> [FOLD]
#   e.g. ./run_baseline.sh 1 0        # dataset001, fold 0 only
set -euo pipefail

DATASET_ID=${1:?Usage: run_baseline.sh <DATASET_ID> [FOLD]}
FOLD=${2:-0}
CONFIG=${CONFIG:-3d_fullres}   # override with e.g. CONFIG=3d_lowres if VRAM-limited

echo "== Preprocessing (only needs to run once per dataset) =="
nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" --verify_dataset_integrity

echo "== Training baseline (fold ${FOLD}, config ${CONFIG}) =="
nnUNetv2_train "${DATASET_ID}" "${CONFIG}" "${FOLD}"

echo "== Baseline training complete =="
echo "Next: run inference on your held-out set, then evaluation/compute_metrics.py"
