#!/bin/bash
# Queues the ResEnc M + widened-augmentation combo (nnUNetTrainerWideAugBaseline_500epochs
# on nnUNetResEncUNetMPlans) -- the two cleanest, no-clear-weakness wins from tonight's
# study, combined. See conversation: deliberately NOT also expanding the augmentation
# ranges further in this run -- that would stack a third, unvalidated variable onto an
# already-compound (architecture x augmentation) test and confound the result if it
# under/over-performs. wideaug's ranges are used exactly as already validated in
# baseline-wideaug-500 (custom_trainers/nnUNetTrainerWideAug.py), unchanged.
#
# Same template as queue_resencm_baseline500.sh (verified working, ran baseline-500 on
# ResEnc M successfully) -- only the trainer group differs.
#
# No overfit-sanity-gate, same reasoning as queue_resencm_baseline500.sh: WideAug is an
# already-proven trainer (real completed run tonight), and ResEnc M is nnU-Net's own
# official preset, not new bespoke loss/sampling logic -- the overfit gates in this
# project exist to catch novel-logic bugs (see topk10-500's collapse), not preset swaps.
#
# --plans nnUNetResEncUNetMPlans gets its own output folder
# (nnUNetTrainerWideAugBaseline_500epochs__nnUNetResEncUNetMPlans__3d_fullres/fold_0) --
# verified below (empty-dir check) before this script was launched, never collides with
# the existing default-plans wideaug-500 run.
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
GROUP=baseline-wideaug-500
TRAINER=nnUNetTrainerWideAugBaseline_500epochs
PLANS=nnUNetResEncUNetMPlans
MANIFEST=workspace/splits_dataset002/manifest.csv
OUT_FOLDER="$nnUNet_results/Dataset002_ATLAS/${TRAINER}__${PLANS}__${CONFIG}/fold_${FOLD}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== queue_resencm_wideaug500.sh starting ==="

log "--- waiting for GPU to be free ---"
while pgrep -f nnUNetv2_train > /dev/null; do
  sleep 30
done
log "--- GPU free, proceeding ---"

log "--- train baseline-wideaug-500 on ResEnc M plans ---"
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

log "=== queue_resencm_wideaug500.sh finished ==="
