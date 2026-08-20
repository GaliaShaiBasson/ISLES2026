#!/bin/bash
# Queues the ResEnc M architecture run (nnUNetTrainerBaseline_500epochs, plain
# Dice+CE, unchanged -- only the plans identifier differs) against the plain
# baseline-500 run, to isolate the architecture-only effect (see CLAUDE.md
# "Switch to nnU-Net's ResEnc planner" future-consideration entry).
#
# No overfit-sanity-gate here, unlike queue_dctopk10.sh -- ResEnc M is nnU-Net's
# own official preset (residual-encoder architecture + budget re-planning), run
# through the already-proven nnUNetTrainerBaseline_500epochs class, not a new
# bespoke loss/sampling trainer with novel logic that could hide a bug (that's
# what the overfit gates in this project actually catch -- see the topk10-500
# collapse). A gate would also need its own ResEnc-M-planned Dataset999 sample
# preprocessing (not set up), extra work not justified for a preset swap.
#
# Waits for the GPU AND for queue_dctopk10.sh's own driver process to finish --
# not just "no nnUNetv2_train running" -- so this never jumps into a brief gap
# between dctopk10's overfit-check gate and its real training start.
#
# --plans nnUNetResEncUNetMPlans gets its own nnU-Net output folder
# (nnUNetTrainerBaseline_500epochs__nnUNetResEncUNetMPlans__3d_fullres/fold_0)
# and its own run_id/results folder (see compute_run_fingerprint/
# run_id_from_fingerprint in isles26.py) -- verified before this script was
# written to be distinct from the existing default-plans baseline-500 folder,
# never collides with or overwrites it.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

PY=/home/galia/miniconda3/envs/isles2026/bin/python
PREDICT=/home/galia/miniconda3/envs/isles2026/bin/nnUNetv2_predict
export nnUNet_extTrainer=/home/galia/ISLES2026/isles26_project
export nnUNet_raw=/home/galia/ISLES2026/nnUNet_raw
export nnUNet_preprocessed=/home/galia/ISLES2026/nnUNet_preprocessed
export nnUNet_results=/home/galia/ISLES2026/nnUNet_results

DATASET_ID=2
DATASET_DIR="$nnUNet_raw/Dataset002_ATLAS"
CONFIG=3d_fullres
FOLD=0
TRAINER=nnUNetTrainerBaseline_500epochs
PLANS=nnUNetResEncUNetMPlans
MANIFEST=workspace/splits_dataset002/manifest.csv
OUT_FOLDER="$nnUNet_results/Dataset002_ATLAS/${TRAINER}__${PLANS}__${CONFIG}/fold_${FOLD}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== queue_resencm_baseline500.sh starting ==="

log "--- waiting for GPU to be free (curriculum run and/or queue_dctopk10.sh) ---"
while pgrep -f nnUNetv2_train > /dev/null || pgrep -f queue_dctopk10.sh > /dev/null; do
  sleep 30
done
log "--- GPU free, proceeding ---"

log "--- train baseline-500 on ResEnc M plans ---"
$PY isles26.py train baseline-500 --dataset-id "$DATASET_ID" --plans "$PLANS"
train_status=$?
if [ $train_status -ne 0 ]; then
  log "[FAIL] train baseline-500 (ResEnc M) exited $train_status"
  exit 1
fi
if [ ! -f "$OUT_FOLDER/checkpoint_final.pth" ]; then
  log "[FAIL] checkpoint_final.pth not found at $OUT_FOLDER"
  exit 1
fi

log "--- predict held-out test set, with probabilities (one pass) ---"
pred_dir="$OUT_FOLDER/predTs"
$PREDICT -i "$DATASET_DIR/imagesTs" -o "$pred_dir" \
  -d "$DATASET_ID" -c "$CONFIG" -tr "$TRAINER" -p "$PLANS" -f "$FOLD" \
  --save_probabilities \
  || log "[FAIL] predict"

log "--- evaluate (val) ---"
$PY isles26.py evaluate \
  --pred-dir "$OUT_FOLDER/validation" \
  --gt-dir "$DATASET_DIR/labelsTr" \
  --case-metadata-csv "$MANIFEST" \
  --trainer "$TRAINER" --dataset-id "$DATASET_ID" --configuration "$CONFIG" --fold "$FOLD" --plans "$PLANS" \
  --split val \
  || log "[FAIL] evaluate val"

log "--- evaluate (held-out test) ---"
$PY isles26.py evaluate \
  --pred-dir "$pred_dir" \
  --gt-dir "$DATASET_DIR/labelsTs" \
  --case-metadata-csv "$MANIFEST" \
  --trainer "$TRAINER" --dataset-id "$DATASET_ID" --configuration "$CONFIG" --fold "$FOLD" --plans "$PLANS" \
  --split test \
  || log "[FAIL] evaluate test"

log "--- aggregate + plot ---"
$PY isles26.py aggregate || log "[FAIL] aggregate"
$PY isles26.py plot || log "[FAIL] plot"

log "=== queue_resencm_baseline500.sh finished ==="
