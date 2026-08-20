#!/bin/bash
# Queues the 1000-epoch baseline run (nnUNetTrainerBaseline_1000epochs, see
# custom_trainers/nnUNetTrainer1000epochs.py) after BOTH currently-running
# scripts finish:
#   - workspace/queue_resencm_baseline500.sh (ResEnc M baseline-500 training +
#     its own predict/evaluate/aggregate/plot tail)
#   - ensembling/queue_export_after_resencm.sh (waits for that training to
#     finish, then runs export_val_probabilities.sh on GPU)
# Waits for both driver processes AND nnUNetv2_train/nnUNetv2_predict to be
# gone -- not just "no nnUNetv2_train running" -- so this never jumps into a
# gap between resencm's training and its own predict/evaluate tail, or
# between that and the export script's predict jobs.
#
# Val evaluation only, no held-out test predict/evaluate -- per explicit
# request. Unlike queue_resencm_baseline500.sh (which also predicts+evaluates
# on imagesTs/labelsTs), this only evaluates the validation/ folder nnU-Net's
# own training routine already writes at the end of training (no separate
# nnUNetv2_predict call needed for that).
#
# Default (non-ResEnc) plans -- nnUNetTrainerBaseline_1000epochs was not
# given a --plans override, so this trains on the same nnUNetPlans identifier
# as baseline-500/baseline-250, distinct output folder from the ResEnc M run
# automatically (nnU-Net namespaces by trainer+plans+config+fold).
#
# Safe to stop/restart: training resumes from checkpoint_latest.pth
# (save_every=10, this project's standing auto-resume convention -- see
# CLAUDE.md "Auto-resume on interrupted training"), and this script itself is
# idempotent to re-launch from scratch -- the wait loop just re-checks
# immediately if the GPU is already free, and `isles26.py train` resumes
# rather than restarts if a partial checkpoint already exists.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

PY=/home/galia/miniconda3/envs/isles2026/bin/python
export nnUNet_extTrainer=/home/galia/ISLES2026/isles26_project
export nnUNet_raw=/home/galia/ISLES2026/nnUNet_raw
export nnUNet_preprocessed=/home/galia/ISLES2026/nnUNet_preprocessed
export nnUNet_results=/home/galia/ISLES2026/nnUNet_results

DATASET_ID=2
CONFIG=3d_fullres
FOLD=0
TRAINER=nnUNetTrainerBaseline_1000epochs
PLANS=nnUNetPlans
MANIFEST=workspace/splits_dataset002/manifest.csv
DATASET_DIR="$nnUNet_raw/Dataset002_ATLAS"
OUT_FOLDER="$nnUNet_results/Dataset002_ATLAS/${TRAINER}__${PLANS}__${CONFIG}/fold_${FOLD}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== queue_baseline1000_after_resencm_export.sh starting ==="

log "--- waiting for queue_resencm_baseline500.sh and queue_export_after_resencm.sh (and any GPU job) to finish ---"
while pgrep -f nnUNetv2_train > /dev/null \
   || pgrep -f nnUNetv2_predict > /dev/null \
   || pgrep -f queue_resencm_baseline500.sh > /dev/null \
   || pgrep -f queue_export_after_resencm.sh > /dev/null \
   || pgrep -f export_val_probabilities.sh > /dev/null; do
  sleep 30
done
log "--- GPU free, proceeding ---"

log "--- train baseline-1000 (default plans) ---"
$PY isles26.py train baseline-1000 --dataset-id "$DATASET_ID"
train_status=$?
if [ $train_status -ne 0 ]; then
  log "[FAIL] train baseline-1000 exited $train_status"
  exit 1
fi
if [ ! -f "$OUT_FOLDER/checkpoint_final.pth" ]; then
  log "[FAIL] checkpoint_final.pth not found at $OUT_FOLDER"
  exit 1
fi

log "--- evaluate (val only, per request -- no held-out test predict/evaluate) ---"
$PY isles26.py evaluate \
  --pred-dir "$OUT_FOLDER/validation" \
  --gt-dir "$DATASET_DIR/labelsTr" \
  --case-metadata-csv "$MANIFEST" \
  --trainer "$TRAINER" --dataset-id "$DATASET_ID" --configuration "$CONFIG" --fold "$FOLD" --plans "$PLANS" \
  --split val \
  || log "[FAIL] evaluate val"

log "--- aggregate + plot ---"
$PY isles26.py aggregate || log "[FAIL] aggregate"
$PY isles26.py plot || log "[FAIL] plot"

log "=== queue_baseline1000_after_resencm_export.sh finished ==="
