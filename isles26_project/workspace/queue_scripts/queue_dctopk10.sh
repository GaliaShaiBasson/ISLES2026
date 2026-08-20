#!/bin/bash
# Queues the DC+TopK10 sanity gate, then (if it passes) the real 500-epoch/dataset002
# launch, fully self-contained -- probabilities and all evaluation metrics (dice, hd95,
# lesion_f1, avd_mm3, lesion_count_diff) saved in ONE pass each, no later re-run needed
# (see CLAUDE.md/conversation: topk10-500's predict had to be redone from scratch once
# already for missing --save_probabilities -- not repeating that mistake here).
#
# Waits for the GPU to be free first (sampling-pow-curriculum-500 was running when this
# was staged) -- queues behind it rather than contending, same reasoning as
# run_500ep_full_experiment.sh's wait loop.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY=/home/galia/miniconda3/envs/isles2026/bin/python
PREDICT=/home/galia/miniconda3/envs/isles2026/bin/nnUNetv2_predict
export nnUNet_extTrainer=/home/galia/ISLES2026/isles26_project_simplified
export nnUNet_raw=/home/galia/ISLES2026/nnUNet_raw
export nnUNet_preprocessed=/home/galia/ISLES2026/nnUNet_preprocessed
export nnUNet_results=/home/galia/ISLES2026/nnUNet_results

DATASET_ID=2
DATASET_DIR="$nnUNet_raw/Dataset002_ATLAS"
CONFIG=3d_fullres
FOLD=0
TRAINER=nnUNetTrainerDCTopk10_500epochs
MANIFEST=workspace/splits_full/manifest.csv
OUT_FOLDER="$nnUNet_results/Dataset002_ATLAS/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== queue_dctopk10.sh starting ==="

log "--- waiting for GPU to be free ---"
while pgrep -f nnUNetv2_train > /dev/null; do
  sleep 30
done
log "--- GPU free, proceeding ---"

log "--- overfit-check-dctopk10 (sanity gate) ---"
if ! ./sanity_overfit_check.sh overfit-check-dctopk10; then
  log "[FAIL] overfit-check-dctopk10 did not pass -- NOT launching the real run. Investigate before retrying."
  exit 1
fi
log "--- overfit-check-dctopk10 passed ---"

log "--- waiting for GPU to be free again (sanity gate itself may have queued behind something) ---"
while pgrep -f nnUNetv2_train > /dev/null; do
  sleep 30
done

log "--- train dctopk10-500 ---"
$PY isles26.py train dctopk10-500 --dataset-id "$DATASET_ID"
train_status=$?
if [ $train_status -ne 0 ]; then
  log "[FAIL] train dctopk10-500 exited $train_status"
  exit 1
fi
if [ ! -f "$OUT_FOLDER/checkpoint_final.pth" ]; then
  log "[FAIL] checkpoint_final.pth not found at $OUT_FOLDER"
  exit 1
fi

log "--- predict held-out test set, with probabilities (one pass) ---"
pred_dir="$OUT_FOLDER/predTs"
$PREDICT -i "$DATASET_DIR/imagesTs" -o "$pred_dir" \
  -d "$DATASET_ID" -c "$CONFIG" -tr "$TRAINER" -f "$FOLD" \
  --save_probabilities \
  || log "[FAIL] predict"

log "--- evaluate (val) ---"
$PY isles26.py evaluate \
  --pred-dir "$OUT_FOLDER/validation" \
  --gt-dir "$DATASET_DIR/labelsTr" \
  --case-metadata-csv "$MANIFEST" \
  --trainer "$TRAINER" --dataset-id "$DATASET_ID" --configuration "$CONFIG" --fold "$FOLD" \
  --split val \
  || log "[FAIL] evaluate val"

log "--- evaluate (held-out test) ---"
$PY isles26.py evaluate \
  --pred-dir "$pred_dir" \
  --gt-dir "$DATASET_DIR/labelsTs" \
  --case-metadata-csv "$MANIFEST" \
  --trainer "$TRAINER" --dataset-id "$DATASET_ID" --configuration "$CONFIG" --fold "$FOLD" \
  --split test \
  || log "[FAIL] evaluate test"

log "--- aggregate + plot ---"
$PY isles26.py aggregate || log "[FAIL] aggregate"
$PY isles26.py plot || log "[FAIL] plot"

log "=== queue_dctopk10.sh finished ==="
