#!/usr/bin/env bash
# Phase 2: loss-function study. Trains one fold per loss variant.
# Requires custom_trainers/install_custom_trainers.py to have been run first.
#
# Usage: ./run_loss_experiments.sh <DATASET_ID> [FOLD]
set -euo pipefail

DATASET_ID=${1:?Usage: run_loss_experiments.sh <DATASET_ID> [FOLD]}
FOLD=${2:-0}
CONFIG=${CONFIG:-3d_fullres}

# nnUNetTrainer (default Dice+CE) is already covered by run_baseline.sh --
# no need to retrain it here, just reuse those results in the comparison.
TRAINERS=(
  "nnUNetTrainerDiceOnly"
  "nnUNetTrainerFocal"
  "nnUNetTrainerTversky"
  "nnUNetTrainerFocalTversky"
)

for TR in "${TRAINERS[@]}"; do
  echo "== Training ${TR} (fold ${FOLD}, config ${CONFIG}) =="
  nnUNetv2_train "${DATASET_ID}" "${CONFIG}" "${FOLD}" -tr "${TR}"
done

echo "== All loss variants trained =="
echo "Next: run inference for each trainer's checkpoint, then evaluation/compute_metrics.py"
echo "      followed by evaluation/aggregate_results.py to combine them."
