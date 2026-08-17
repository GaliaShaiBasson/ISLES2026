#!/bin/bash
# Unattended full-pipeline runner: preprocess the real dataset, then train + predict +
# evaluate every condition in the study (baseline, 4 loss variants, lesion-aware
# sampling) sequentially on dataset-id 1 (the real ~1,284-case ATLAS R3.0 split).
#
# Designed to run for many hours with nobody watching it:
#   - No `set -e`: one condition failing (bad case, OOM, etc.) is logged and the
#     script moves on to the next condition rather than dying silently. Each
#     condition is independent by the project's own design (PROJECT_PLAN.md /
#     CLAUDE.md), so a partial run is still a valid partial result.
#   - Aggregate + plot are re-run after every condition, not just at the end, so
#     whatever has finished is always reflected in workspace/evaluation/results.csv
#     and workspace/figures/*.png even if the script is still mid-way through when
#     someone checks on it.
#   - Absolute binary paths (not relying on PATH/conda activation in a
#     non-interactive shell).
#
# Usage:
#   nohup ./run_full_experiment.sh > workspace/full_run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# To check progress while it runs: tail -f workspace/full_run_*.log
# To stop it: pkill -f run_full_experiment.sh ; pkill -f nnUNetv2_train

PY=/home/galia/miniconda3/envs/isles2026/bin/python
PREDICT=/home/galia/miniconda3/envs/isles2026/bin/nnUNetv2_predict
export nnUNet_extTrainer=/home/galia/ISLES2026/isles26_project_simplified
export nnUNet_raw=/home/galia/ISLES2026/nnUNet_raw
export nnUNet_preprocessed=/home/galia/ISLES2026/nnUNet_preprocessed
export nnUNet_results=/home/galia/ISLES2026/nnUNet_results

cd /home/galia/ISLES2026/isles26_project_simplified || exit 1

DATASET_ID=1
DATASET_DIR="$nnUNet_raw/Dataset001_ATLAS"
CONFIG=3d_fullres
FOLD=0
MANIFEST=workspace/splits/manifest.csv

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== Full experiment run starting ==="

log "--- preprocess (dataset $DATASET_ID) ---"
$PY isles26.py preprocess --dataset-id "$DATASET_ID" \
  || log "[FAIL] preprocess -- subsequent training will likely fail too"

# group -> trainer class name (must match TRAINER_GROUPS in isles26.py)
declare -A TRAINERS=(
  [baseline]="nnUNetTrainerBaseline_250epochs"
  [dice]="nnUNetTrainerDiceOnly_250epochs"
  [focal]="nnUNetTrainerFocal_250epochs"
  [tversky]="nnUNetTrainerTversky_250epochs"
  [focal-tversky]="nnUNetTrainerFocalTversky_250epochs"
  [sampling]="nnUNetTrainerLesionAwareSampling_250epochs"
)
# Baseline first (also doubles as a smoke test of the full-scale 3d_fullres run
# before committing to the rest), then losses, then sampling -- matches
# PROJECT_PLAN.md's phase order.
ORDER=(baseline dice focal tversky focal-tversky sampling)

reaggregate() {
  log "--- aggregate + plot (incremental) ---"
  $PY isles26.py aggregate || log "[FAIL] aggregate"
  $PY isles26.py plot || log "[FAIL] plot"
}

MAX_TRAIN_ATTEMPTS=3

for group in "${ORDER[@]}"; do
  trainer="${TRAINERS[$group]}"
  out_folder="$nnUNet_results/Dataset001_ATLAS/${trainer}__nnUNetPlans__${CONFIG}/fold_${FOLD}"

  log "=== Condition: $group ($trainer) ==="

  # Retry loop, not a single attempt: isles26.py train auto-resumes from
  # checkpoint_latest.pth when re-invoked (see isles26.py's [resume] guard) --
  # a single failed attempt here (transient CUDA hiccup, brief OOM from
  # something else on the GPU, etc.) would otherwise abandon a partially
  # trained condition instead of giving it a chance to pick back up.
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
  $PREDICT -i "$nnUNet_raw/Dataset001_ATLAS/imagesTs" -o "$pred_dir" \
    -d "$DATASET_ID" -c "$CONFIG" -tr "$trainer" -f "$FOLD" \
    || log "[FAIL] predict $group"

  log "--- evaluate $group (val) ---"
  $PY isles26.py evaluate \
    --pred-dir "$out_folder/validation" \
    --gt-dir "$nnUNet_raw/Dataset001_ATLAS/labelsTr" \
    --case-metadata-csv "$MANIFEST" \
    --experiment "${group}_val" \
    --out-csv "workspace/evaluation/results_${group}_val.csv" \
    || log "[FAIL] evaluate ${group}_val"

  log "--- evaluate $group (held-out test) ---"
  $PY isles26.py evaluate \
    --pred-dir "$pred_dir" \
    --gt-dir "$nnUNet_raw/Dataset001_ATLAS/labelsTs" \
    --case-metadata-csv "$MANIFEST" \
    --experiment "${group}_test" \
    --out-csv "workspace/evaluation/results_${group}_test.csv" \
    || log "[FAIL] evaluate ${group}_test"

  reaggregate
  log "=== Condition $group done ==="
done

log "=== Full experiment run finished (all conditions attempted) ==="
