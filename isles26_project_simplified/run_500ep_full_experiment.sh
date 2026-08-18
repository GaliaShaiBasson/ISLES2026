#!/bin/bash
# Unattended full-pipeline runner for the 500-epoch / dataset002 (corrected full-split)
# study: baseline, focal-tversky, tversky-mild, sampling (original 4:2:1, not pow --
# see CLAUDE.md), sequentially on dataset-id 2 (the corrected ~1,453-case ATLAS split).
#
# Mirrors run_full_experiment.sh's structure and unattended-safety properties exactly
# (retry-with-auto-resume per condition, incremental aggregate/plot, no `set -e`) --
# see that script's header comment for the full rationale. Differences specific to
# this run:
#   - Does NOT call `preprocess` -- Dataset002_ATLAS was already prepared+preprocessed
#     and verified this session (see CLAUDE.md). Errors out loudly instead of
#     silently reprocessing if that's missing, rather than risk overwriting verified
#     preprocessed data unattended.
#   - Waits for any currently-running nnUNetv2_train process to finish before starting
#     (tversky-mild_250epochs was already running on the GPU, dataset 1, when this was
#     staged) -- queues behind it instead of contending for GPU compute. Separate
#     dataset id, so zero output-path overlap regardless -- this wait is purely about
#     not splitting GPU compute between two simultaneous training jobs.
#   - tversky-mild-500 is included ALONGSIDE focal-tversky-500, not instead of it --
#     tversky-mild's own 250-epoch/dataset001 run had no results yet when this was
#     assembled, so it's a fourth condition here, not a substitution.
#   - No --overwrite passed anywhere in this script -- isles26.py's own checkpoint /
#     fingerprint guards stay fully active as a second line of defense against any
#     accidental collision, on top of the fresh (never-before-used) Dataset002_ATLAS
#     results tree and the run-fingerprint-namespaced evaluate output.
#   - Stock augmentation only (DA5 was investigated, not verified in time -- deferred,
#     see CLAUDE.md future considerations).
#
# Usage:
#   nohup ./run_500ep_full_experiment.sh > workspace/full_run_500ep_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# To check progress while it runs: tail -f workspace/full_run_500ep_*.log
# To stop it: pkill -f run_500ep_full_experiment.sh ; pkill -f nnUNetv2_train

PY=/home/galia/miniconda3/envs/isles2026/bin/python
PREDICT=/home/galia/miniconda3/envs/isles2026/bin/nnUNetv2_predict
export nnUNet_extTrainer=/home/galia/ISLES2026/isles26_project_simplified
export nnUNet_raw=/home/galia/ISLES2026/nnUNet_raw
export nnUNet_preprocessed=/home/galia/ISLES2026/nnUNet_preprocessed
export nnUNet_results=/home/galia/ISLES2026/nnUNet_results

cd /home/galia/ISLES2026/isles26_project_simplified || exit 1

DATASET_ID=2
DATASET_DIR="$nnUNet_raw/Dataset002_ATLAS"
CONFIG=3d_fullres
FOLD=0
MANIFEST=workspace/splits_full/manifest.csv

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== 500-epoch dataset002 experiment run starting ==="

if [ ! -f "$nnUNet_preprocessed/Dataset002_ATLAS/nnUNetPlans.json" ]; then
  log "[FATAL] Dataset002_ATLAS is not preprocessed (nnUNetPlans.json missing). Refusing to auto-preprocess in this script -- run 'python isles26.py preprocess --dataset-id 2' manually first."
  exit 1
fi

log "--- waiting for GPU to be free (queuing behind any already-running nnUNetv2_train) ---"
while pgrep -f nnUNetv2_train > /dev/null; do
  sleep 30
done
log "--- GPU free, proceeding ---"

# group -> trainer class name (must match TRAINER_GROUPS in isles26.py)
declare -A TRAINERS=(
  [baseline-500]="nnUNetTrainerBaseline_500epochs"
  [focal-tversky-500]="nnUNetTrainerFocalTversky_500epochs"
  [tversky-mild-500]="nnUNetTrainerTverskyMild_500epochs"
  [sampling-500]="nnUNetTrainerLesionAwareSampling_500epochs_full"
)
ORDER=(baseline-500 focal-tversky-500 tversky-mild-500 sampling-500)

reaggregate() {
  log "--- aggregate + plot (incremental) ---"
  $PY isles26.py aggregate || log "[FAIL] aggregate"
  $PY isles26.py plot || log "[FAIL] plot"
}

MAX_TRAIN_ATTEMPTS=3

for group in "${ORDER[@]}"; do
  trainer="${TRAINERS[$group]}"
  out_folder="$nnUNet_results/Dataset002_ATLAS/${trainer}__nnUNetPlans__${CONFIG}/fold_${FOLD}"

  log "=== Condition: $group ($trainer) ==="

  attempt=1
  train_status=1
  while [ "$attempt" -le "$MAX_TRAIN_ATTEMPTS" ] && [ "$train_status" -ne 0 ]; do
    log "--- train $group (attempt $attempt/$MAX_TRAIN_ATTEMPTS) ---"
    $PY isles26.py train "$group" --dataset-id "$DATASET_ID"
    train_status=$?
    if [ "$train_status" -ne 0 ]; then
      log "[WARN] train $group attempt $attempt failed (exit $train_status)"
      attempt=$((attempt + 1))
      sleep 30
    fi
  done
  if [ "$train_status" -ne 0 ]; then
    log "[FAIL] train $group exhausted $MAX_TRAIN_ATTEMPTS attempts -- skipping predict/evaluate for this condition"
    reaggregate
    continue
  fi

  if [ ! -f "$out_folder/checkpoint_final.pth" ]; then
    log "[FAIL] $group: checkpoint_final.pth not found at $out_folder -- skipping predict/evaluate"
    reaggregate
    continue
  fi

  log "--- predict held-out test set ($group) ---"
  pred_dir="$out_folder/predTs"
  $PREDICT -i "$DATASET_DIR/imagesTs" -o "$pred_dir" \
    -d "$DATASET_ID" -c "$CONFIG" -tr "$trainer" -f "$FOLD" \
    || log "[FAIL] predict $group"

  log "--- evaluate $group (val) ---"
  $PY isles26.py evaluate \
    --pred-dir "$out_folder/validation" \
    --gt-dir "$DATASET_DIR/labelsTr" \
    --case-metadata-csv "$MANIFEST" \
    --trainer "$trainer" --dataset-id "$DATASET_ID" --configuration "$CONFIG" --fold "$FOLD" \
    --split val \
    || log "[FAIL] evaluate ${group}_val"

  log "--- evaluate $group (held-out test) ---"
  $PY isles26.py evaluate \
    --pred-dir "$pred_dir" \
    --gt-dir "$DATASET_DIR/labelsTs" \
    --case-metadata-csv "$MANIFEST" \
    --trainer "$trainer" --dataset-id "$DATASET_ID" --configuration "$CONFIG" --fold "$FOLD" \
    --split test \
    || log "[FAIL] evaluate ${group}_test"

  reaggregate
  log "=== Condition $group done ==="
done

log "=== 500-epoch dataset002 experiment run finished (all conditions attempted) ==="
