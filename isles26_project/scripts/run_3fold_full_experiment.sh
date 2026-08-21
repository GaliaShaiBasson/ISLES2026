#!/bin/bash
# Unattended driver for the final 3-fold, full-dev-pool training run --
# WideAugBaseline on nnUNetResEncUNetMPlans (the confirmed winner, see
# workspace/results/finalist_selection/summary_val.csv and PROJECT_PLAN.md),
# trained on ALL of train+val+test_id+test_ood (1,212 cases) pooled into 3
# stratified-by-size_bin folds. workspace/final_holdout/{final_holdout_id,
# final_holdout_ood} is the ONLY evaluation left for this model and is never
# read, copied, or referenced anywhere in this script.
#
# New dataset ID (3), not a mutation of Dataset002 -- keeps every existing
# single-fold checkpoint/result completely untouched (see CLAUDE.md
# "Gotcha for later" / the fingerprinting entries for why in-place splits_final.json
# swaps on a live dataset are unsafe). Costs one full preprocessing pass on
# 1,212 cases before training can start.
#
# Stages (each idempotent / skip-if-already-done -- a re-run after a crash,
# reboot, or lost SSH session just picks back up, no manual bookkeeping):
#   1. Pool train+val+test_id+test_ood into a stratified 3-fold splits_final.json
#      (data_prep/split_dataset_3fold_final.py) -- skipped if it already exists.
#   2. Build a prepare-compatible synthetic splits-dir that relabels the pool
#      as "train" (data_prep/make_pooled_splits_dir.py) -- prepare's own
#      test_id/test_ood/final_holdout exclusion logic stays completely
#      unmodified and does the real safety work; see that script's docstring.
#   3. prepare --dataset-id 3 (writes imagesTr/labelsTr for all 1,212 cases;
#      prepare's own --overwrite guard makes this a no-op on re-run).
#   4. Install our real 3-fold splits_final.json over prepare's auto-written
#      single-fold placeholder, in nnUNet_raw ONLY -- before preprocessing,
#      so preprocess's own copy-into-nnUNet_preprocessed step (see
#      data_prep_cli.py:_copy_splits_final_json) carries the 3-fold version
#      through automatically. Guarded: refuses if the installed file doesn't
#      already have exactly 3 folds (catches a stale/wrong source file).
#   5. preprocess --dataset-id 3 (default nnUNetPlans; skip-if-planned guard).
#   6. nnUNetv2_plan_experiment + nnUNetv2_preprocess for nnUNetResEncUNetMPlans
#      specifically -- isles26.py's preprocess stage only ever runs the
#      default planner (see CLAUDE.md "Optimizing GPU usage" background);
#      the ResEncM plans variant needs this same explicit extra step every
#      other ResEncM run in this project also needed (queue_resencm_*.sh),
#      just never automated until now. Uses -pl ResEncUNetPlanner with NO
#      gpu_memory_target override (class default 8GB) and
#      -overwrite_plans_name nnUNetResEncUNetMPlans -- verified to reproduce
#      Dataset002's own nnUNetResEncUNetMPlans.json byte-for-byte on the
#      architecture fields that don't depend on per-dataset fingerprinting
#      (patch_size [128,128,128], batch_size 2, features_per_stage
#      [32,64,128,256,320,320]) before this script was written.
#   7. Train fold 0, then 1, then 2 of nnUNetTrainerWideAugBaseline_500epochs
#      on nnUNetResEncUNetMPlans -- sequential (same GPU-contention reasoning
#      documented in CLAUDE.md "Optimizing GPU usage": two concurrent 3d_fullres
#      jobs are ~2x slower each, a net loss). Each fold gets its own
#      retry-with-auto-resume loop (train_cli.py's checkpoint_latest.pth
#      resume, same mechanism as run_full_experiment.sh) -- a crash mid-fold
#      picks back up from its last saved epoch, not from epoch 0.
#
# No predict/evaluate step at the end: there is no more internal held-out
# test for this model (everything went into the pooled training set) -- the
# only remaining evaluation is workspace/final_holdout/, deliberately a
# separate, explicit, manual step never triggered by this script.
#
# Usage:
#   nohup ./scripts/run_3fold_full_experiment.sh > workspace/logs/run_3fold_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# To check progress: tail -f workspace/logs/run_3fold_full_*.log
# To stop it: pkill -f run_3fold_full_experiment.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY=/home/galia/miniconda3/envs/isles2026/bin/python
PLAN_EXPERIMENT=/home/galia/miniconda3/envs/isles2026/bin/nnUNetv2_plan_experiment
PREPROCESS_BIN=/home/galia/miniconda3/envs/isles2026/bin/nnUNetv2_preprocess
export nnUNet_extTrainer=/home/galia/ISLES2026/isles26_project
export nnUNet_raw=/home/galia/ISLES2026/nnUNet_raw
export nnUNet_preprocessed=/home/galia/ISLES2026/nnUNet_preprocessed
export nnUNet_results=/home/galia/ISLES2026/nnUNet_results

DATASET_ID=3
DATASET_ID_PADDED=$(printf '%03d' "$DATASET_ID")
DATASET_NAME=ATLAS_full
DATASET_DIR="$nnUNet_raw/Dataset${DATASET_ID_PADDED}_${DATASET_NAME}"
PREPROCESSED_DIR="$nnUNet_preprocessed/Dataset${DATASET_ID_PADDED}_${DATASET_NAME}"
GROUP=baseline-wideaug-500
TRAINER=nnUNetTrainerWideAugBaseline_500epochs
PLANS=nnUNetResEncUNetMPlans
CONFIG=3d_fullres
SPLIT_OUT_DIR=workspace/splits_dataset003_full
PREPARE_SPLITS_DIR="$SPLIT_OUT_DIR/prepare_splits_dir"
MAX_ATTEMPTS=3

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_with_retries() {
  local desc="$1"; shift
  local attempt=1 status=1
  while [ "$attempt" -le "$MAX_ATTEMPTS" ] && [ "$status" -ne 0 ]; do
    "$@"; status=$?
    if [ "$status" -ne 0 ]; then
      log "[warn] $desc failed (attempt $attempt/$MAX_ATTEMPTS, exit $status)"
      attempt=$((attempt + 1))
      [ "$attempt" -le "$MAX_ATTEMPTS" ] && sleep 15
    fi
  done
  if [ "$status" -ne 0 ]; then
    log "[error] $desc exhausted $MAX_ATTEMPTS attempts -- aborting (this stage's output is a prerequisite for everything after it, unlike the postprocess grid's independent combos)"
    exit 1
  fi
  return 0
}

log "=== run_3fold_full_experiment.sh starting ==="

# --- Stage 1: pool + 3-fold split ---
if [ -f "$SPLIT_OUT_DIR/splits_final.json" ]; then
  log "[skip] $SPLIT_OUT_DIR/splits_final.json already exists"
else
  run_with_retries "3-fold pool split" "$PY" data_prep/split_dataset_3fold_final.py --out-dir "$SPLIT_OUT_DIR"
fi

# --- Stage 2: synthetic prepare splits-dir ---
if [ -f "$PREPARE_SPLITS_DIR/train.csv" ]; then
  log "[skip] $PREPARE_SPLITS_DIR/train.csv already exists"
else
  run_with_retries "pooled splits-dir build" "$PY" data_prep/make_pooled_splits_dir.py \
    --pooled-manifest "$SPLIT_OUT_DIR/pooled_manifest.csv" --out-dir "$PREPARE_SPLITS_DIR"
fi

# --- Stage 3: prepare (imagesTr/labelsTr for all 1,212 pooled cases) ---
if [ -f "$DATASET_DIR/dataset.json" ]; then
  log "[skip] $DATASET_DIR/dataset.json already exists"
else
  run_with_retries "prepare" "$PY" isles26.py prepare \
    --dataset-id "$DATASET_ID" --dataset-name "$DATASET_NAME" \
    --splits-dir "$PREPARE_SPLITS_DIR" \
    --out-metadata-csv workspace/case_metadata/case_metadata_dataset003_full.csv
fi

# --- Stage 4: install the real 3-fold splits_final.json over prepare's single-fold placeholder ---
n_folds=$("$PY" -c "import json; print(len(json.load(open('$SPLIT_OUT_DIR/splits_final.json'))))")
if [ "$n_folds" != "3" ]; then
  log "[FATAL] $SPLIT_OUT_DIR/splits_final.json has $n_folds folds, expected 3 -- refusing to install"
  exit 1
fi
installed_folds="0"
if [ -f "$DATASET_DIR/splits_final.json" ]; then
  installed_folds=$("$PY" -c "import json; print(len(json.load(open('$DATASET_DIR/splits_final.json'))))" 2>/dev/null || echo 0)
fi
if [ "$installed_folds" = "3" ]; then
  log "[skip] $DATASET_DIR/splits_final.json already has 3 folds installed"
else
  cp "$SPLIT_OUT_DIR/splits_final.json" "$DATASET_DIR/splits_final.json"
  log "Installed 3-fold splits_final.json into $DATASET_DIR (was $installed_folds fold(s))"
fi

# --- Stage 5: default-plans preprocessing (also re-copies splits_final.json into nnUNet_preprocessed) ---
if [ -f "$PREPROCESSED_DIR/nnUNetPlans.json" ]; then
  log "[skip] $PREPROCESSED_DIR/nnUNetPlans.json already exists"
else
  run_with_retries "preprocess (default plans)" "$PY" isles26.py preprocess --dataset-id "$DATASET_ID"
fi
installed_preprocessed_folds="0"
if [ -f "$PREPROCESSED_DIR/splits_final.json" ]; then
  installed_preprocessed_folds=$("$PY" -c "import json; print(len(json.load(open('$PREPROCESSED_DIR/splits_final.json'))))" 2>/dev/null || echo 0)
fi
if [ "$installed_preprocessed_folds" != "3" ]; then
  log "[FATAL] $PREPROCESSED_DIR/splits_final.json has $installed_preprocessed_folds folds after preprocess -- expected 3. "
  log "        preprocess's own copy step (data_prep_cli.py:_copy_splits_final_json) should have carried the 3-fold "
  log "        version from $DATASET_DIR/splits_final.json -- investigate before training on a possibly-wrong split."
  exit 1
fi
log "Confirmed: $PREPROCESSED_DIR/splits_final.json has 3 folds"

# --- Stage 6: ResEncM plans + preprocessing ---
if [ -f "$PREPROCESSED_DIR/${PLANS}.json" ]; then
  log "[skip] $PREPROCESSED_DIR/${PLANS}.json already exists"
else
  run_with_retries "plan_experiment (ResEncM)" "$PLAN_EXPERIMENT" -d "$DATASET_ID" \
    -pl ResEncUNetPlanner -overwrite_plans_name "$PLANS"
fi
if [ -d "$PREPROCESSED_DIR/${PLANS}_${CONFIG}" ]; then
  log "[skip] $PREPROCESSED_DIR/${PLANS}_${CONFIG} already exists"
else
  run_with_retries "preprocess (ResEncM)" "$PREPROCESS_BIN" -d "$DATASET_ID" \
    -plans_name "$PLANS" -c "$CONFIG"
fi

# --- Stage 7: train folds 0, 1, 2 sequentially, each with auto-resume ---
for FOLD in 0 1 2; do
  CKPT="$nnUNet_results/Dataset${DATASET_ID_PADDED}_${DATASET_NAME}/${TRAINER}__${PLANS}__${CONFIG}/fold_${FOLD}/checkpoint_final.pth"
  if [ -f "$CKPT" ]; then
    log "[skip] fold $FOLD already has checkpoint_final.pth"
    continue
  fi
  log "=== training fold $FOLD ($TRAINER on $PLANS) ==="
  run_with_retries "train fold $FOLD" "$PY" isles26.py train "$GROUP" \
    --dataset-id "$DATASET_ID" --dataset-name "$DATASET_NAME" \
    --fold "$FOLD" --plans "$PLANS" --device cuda
done

log "=== run_3fold_full_experiment.sh finished: all 3 folds trained ==="
log "Next (manual, deliberately not automated here): evaluate against workspace/final_holdout/ "
log "(final_holdout_id.csv + final_holdout_ood.csv) -- the project's real, one-time final check."
