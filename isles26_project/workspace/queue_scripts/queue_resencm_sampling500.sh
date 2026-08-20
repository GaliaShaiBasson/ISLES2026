#!/bin/bash
# Queues the ResEnc M + lesion-aware sampling (4:2:1) combo
# (nnUNetTrainerLesionAwareSampling_500epochs_full on nnUNetResEncUNetMPlans) -- the
# second-priority architecture combo, chosen because sampling's own value proposition
# (small/medium-lesion recall, at a real large-lesion Dice/F1 cost -- see CLAUDE.md) is
# mechanistically orthogonal to architecture capacity, unlike wideaug's; worth checking
# whether ResEnc M's extra capacity narrows that large-lesion cost rather than just
# amplifying the existing strengths.
#
# Waits for queue_resencm_wideaug500.sh (by process name) as well as the GPU, so this
# never launches concurrently with it -- runs strictly after, same
# one-condition-at-a-time discipline as every other queue script tonight.
#
# Same template as queue_resencm_baseline500.sh / queue_resencm_wideaug500.sh (both
# verified working). No overfit-sanity-gate, same reasoning: lesion-aware sampling is
# an already-proven trainer (real completed run tonight) and ResEnc M is nnU-Net's own
# official preset, not new bespoke logic.
#
# --plans nnUNetResEncUNetMPlans gets its own output folder
# (nnUNetTrainerLesionAwareSampling_500epochs_full__nnUNetResEncUNetMPlans__3d_fullres/
# fold_0) -- distinct from the existing default-plans sampling-500 run.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

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
GROUP=sampling-500
TRAINER=nnUNetTrainerLesionAwareSampling_500epochs_full
PLANS=nnUNetResEncUNetMPlans
MANIFEST=workspace/splits_dataset002/manifest.csv
OUT_FOLDER="$nnUNet_results/Dataset002_ATLAS/${TRAINER}__${PLANS}__${CONFIG}/fold_${FOLD}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== queue_resencm_sampling500.sh starting ==="

log "--- waiting for GPU AND queue_resencm_wideaug500.sh to finish ---"
while pgrep -f nnUNetv2_train > /dev/null || pgrep -f queue_resencm_wideaug500.sh > /dev/null; do
  sleep 30
done
log "--- clear, proceeding ---"

log "--- train sampling-500 on ResEnc M plans ---"
$PY isles26.py train "$GROUP" --dataset-id "$DATASET_ID" --plans "$PLANS"
train_status=$?
if [ $train_status -ne 0 ]; then
  log "[FAIL] train $GROUP (ResEnc M) exited $train_status"
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

log "=== queue_resencm_sampling500.sh finished ==="
