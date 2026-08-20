#!/bin/bash
# Exports val-set softmax probabilities for nnUNetTrainerLesionAwareSampling_500epochs_full
# (plain 4:2:1 fixed-ratio sampling, default nnUNetPlans) -- not part of the
# finalist shortlist ensembling/export_val_probabilities.sh covers (that list is
# tied to workspace/evaluation/finalist_selection/redundancy_recommendation.csv
# and shouldn't silently diverge from it -- see that script's TRAINERS comment),
# so this is a standalone addition rather than an edit to the shared script.
#
# Same staging + predict pattern as export_val_probabilities.sh's run_one():
# predicts from the staged val-image symlink folder (never imagesTr directly,
# never nnU-Net's own --val --npz retraining flag -- see stage_val_images.py's
# docstring for why), writes into a brand-new predVal_prob/ (sibling to the
# existing validation/, never touched), refuses to run if predVal_prob/
# already exists unless --overwrite.
#
# No GPU-wait loop needed at launch time -- unlike the ResEnc M export, this
# was queued when the GPU was already confirmed idle. Still checked here for
# safety in case something else starts before this reaches the predict step.
#
# Usage:
#   ./workspace/queue_export_lesionawaresampling_valprob.sh                  # CPU (default)
#   ./workspace/queue_export_lesionawaresampling_valprob.sh --device cuda
#   ./workspace/queue_export_lesionawaresampling_valprob.sh --overwrite
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

export nnUNet_extTrainer=/home/galia/ISLES2026/isles26_project_simplified
export nnUNet_raw=/home/galia/ISLES2026/nnUNet_raw
export nnUNet_preprocessed=/home/galia/ISLES2026/nnUNet_preprocessed
export nnUNet_results=/home/galia/ISLES2026/nnUNet_results

PY=/home/galia/miniconda3/envs/isles2026/bin/python
PREDICT=/home/galia/miniconda3/envs/isles2026/bin/nnUNetv2_predict
DATASET_ID=2
TRAINER=nnUNetTrainerLesionAwareSampling_500epochs_full
PLANS=nnUNetPlans
STAGED_DIR="workspace/val_images_staged"
OUT_DIR="$nnUNet_results/Dataset002_ATLAS/${TRAINER}__${PLANS}__3d_fullres/fold_0/predVal_prob"

DEVICE="cpu"
OVERWRITE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== queue_export_lesionawaresampling_valprob.sh starting ==="

log "--- waiting for GPU (safety check) ---"
while pgrep -f nnUNetv2_train > /dev/null || pgrep -f nnUNetv2_predict > /dev/null; do
  sleep 30
done
log "--- clear, proceeding ---"

if [ -d "$OUT_DIR" ] && [ "$OVERWRITE" -eq 0 ]; then
  log "[skip] $OUT_DIR already exists (pass --overwrite to redo)"
  exit 0
fi

log "=== staging val images (symlinks only, source untouched) ==="
"$PY" ensembling/stage_val_images.py || { log "[FAIL] staging val images"; exit 1; }

mkdir -p "$OUT_DIR"
log "=== predicting $TRAINER / $PLANS -> $OUT_DIR (device=$DEVICE) ==="
"$PREDICT" -i "$STAGED_DIR" -o "$OUT_DIR" \
  -d "$DATASET_ID" -c 3d_fullres -tr "$TRAINER" -p "$PLANS" -f 0 \
  --save_probabilities -device "$DEVICE"
status=$?
if [ $status -eq 0 ]; then
  log "=== done (exit 0) ==="
else
  log "=== FAILED (exit $status) ==="
  exit $status
fi

log "=== queue_export_lesionawaresampling_valprob.sh finished ==="