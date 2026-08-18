#!/usr/bin/env bash
# Phase 3: lesion-aware sampling. Compares uniform vs inverse-volume case
# sampling, holding the loss function fixed (default Dice+CE unless you
# pass a different -tr-compatible base via BEST_LOSS_TRAINER below).
#
# Usage:
#   export ISLES26_CASE_METADATA_CSV=/path/to/case_metadata.csv
#   ./run_sampling_experiment.sh <DATASET_ID> [FOLD]
set -euo pipefail

DATASET_ID=${1:?Usage: run_sampling_experiment.sh <DATASET_ID> [FOLD]}
FOLD=${2:-0}
CONFIG=${CONFIG:-3d_fullres}

if [[ -z "${ISLES26_CASE_METADATA_CSV:-}" ]]; then
  echo "ERROR: set ISLES26_CASE_METADATA_CSV to your case_metadata.csv path first." >&2
  exit 1
fi

echo "== Training with lesion-aware sampling (fold ${FOLD}, config ${CONFIG}) =="
nnUNetv2_train "${DATASET_ID}" "${CONFIG}" "${FOLD}" -tr nnUNetTrainerLesionAwareSampling

echo "== Done =="
echo "Compare against the baseline run (uniform sampling, same fold/config) from run_baseline.sh."
