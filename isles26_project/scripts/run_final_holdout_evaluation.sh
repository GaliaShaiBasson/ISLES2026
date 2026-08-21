#!/bin/bash
# THE real final check -- run this exactly once, only after every model/
# architecture/hyperparameter decision is already locked in. Predicts the
# 3-fold-ensembled WideAugBaseline (ResEncM) model (fold 0+1+2 softmax-
# averaged via nnU-Net's own multi-fold -f mechanism) on
# workspace/final_holdout/{id,ood} -- the project's stand-in for the real,
# never-received ISLES'26 challenge test set -- and scores against
# ground_truth_DO_NOT_TOUCH/. See CLAUDE.md's "Split finalized" entry and
# data_prep/split_dataset.py's own docstring for why this data has never
# been looked at until now.
#
# NOT wired into scripts/run_3fold_full_experiment.sh or any other
# automation -- deliberately requires a separate, manual, deliberate
# invocation. Idempotent (skip-if-output-exists) so a re-run only fills in
# what's missing, but there is no "--overwrite" convenience flag here on
# purpose: if you genuinely need to redo this, that's a decision worth
# making explicitly (rm the output dir yourself), not one line away by
# habit.
#
# Prerequisite: all 3 folds of scripts/run_3fold_full_experiment.sh finished
# (checkpoint_final.pth under fold_0/1/2) -- this script checks and refuses
# to run otherwise.
#
# Usage:
#   nohup ./scripts/run_final_holdout_evaluation.sh > workspace/logs/final_holdout_eval_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY=/home/galia/miniconda3/envs/isles2026/bin/python
PREDICT=/home/galia/miniconda3/envs/isles2026/bin/nnUNetv2_predict
export nnUNet_extTrainer=/home/galia/ISLES2026/isles26_project
export nnUNet_raw=/home/galia/ISLES2026/nnUNet_raw
export nnUNet_preprocessed=/home/galia/ISLES2026/nnUNet_preprocessed
export nnUNet_results=/home/galia/ISLES2026/nnUNet_results

DATASET_ID=3
TRAINER=nnUNetTrainerWideAugBaseline_500epochs
PLANS=nnUNetResEncUNetMPlans
CONFIG=3d_fullres
MODEL_DIR="$nnUNet_results/Dataset003_ATLAS_full/${TRAINER}__${PLANS}__${CONFIG}"
HOLDOUT_DIR=workspace/final_holdout
OUT_ROOT=workspace/predictions/final_holdout
RESULTS_ROOT=workspace/results/final_holdout
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
    log "[error] $desc exhausted $MAX_ATTEMPTS attempts -- aborting"
    exit 1
  fi
  return 0
}

log "=== run_final_holdout_evaluation.sh starting ==="

for fold in 0 1 2; do
  if [ ! -f "$MODEL_DIR/fold_${fold}/checkpoint_final.pth" ]; then
    log "[FATAL] fold $fold has no checkpoint_final.pth under $MODEL_DIR -- 3-fold training isn't finished yet. Refusing to touch final_holdout/."
    exit 1
  fi
done
log "Confirmed: all 3 fold checkpoints exist under $MODEL_DIR"

for tag in id ood; do
  images_dir="$HOLDOUT_DIR/images/$tag"
  gt_dir="$HOLDOUT_DIR/ground_truth_DO_NOT_TOUCH/$tag"
  manifest_csv="$HOLDOUT_DIR/final_holdout_${tag}.csv"
  pred_dir="$OUT_ROOT/$tag"
  results_csv="$RESULTS_ROOT/results_${tag}.csv"

  for path in "$images_dir" "$gt_dir" "$manifest_csv"; do
    if [ ! -e "$path" ]; then
      log "[FATAL] missing: $path"
      exit 1
    fi
  done

  n_expected=$(($(wc -l < "$manifest_csv") - 1))
  if [ -d "$pred_dir" ] && [ "$(find "$pred_dir" -maxdepth 1 -name '*.nii.gz' | wc -l)" -ge "$n_expected" ]; then
    log "[skip] $pred_dir already has $n_expected+ predictions"
  else
    log "=== predicting final_holdout_$tag ($n_expected cases, fold 0+1+2 ensembled) ==="
    run_with_retries "predict $tag" "$PREDICT" \
      -i "$images_dir" -o "$pred_dir" \
      -d "$DATASET_ID" -c "$CONFIG" -p "$PLANS" -tr "$TRAINER" \
      -f 0 1 2 -device cuda
  fi

  mkdir -p "$RESULTS_ROOT"
  if [ -f "$results_csv" ]; then
    log "[skip] $results_csv already exists"
  else
    log "=== scoring final_holdout_$tag ==="
    run_with_retries "compute_metrics $tag" "$PY" evaluation/compute_metrics.py \
      --pred-dir "$pred_dir" --gt-dir "$gt_dir" \
      --case-metadata-csv "$manifest_csv" \
      --experiment-name "wideaug_resencm_3fold_final_holdout_${tag}" \
      --out-csv "$results_csv"
  fi
done

log "=== run_final_holdout_evaluation.sh finished ==="
"$PY" - <<'PYEOF'
import pandas as pd
for tag in ("id", "ood"):
    path = f"workspace/results/final_holdout/results_{tag}.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"{tag}: {path} not found")
        continue
    print(f"final_holdout_{tag}: n={len(df)} dice={df['dice'].mean():.4f} "
          f"hd95={df['hd95_mm'].mean():.3f} lesion_f1={df['lesion_f1'].mean():.4f}")
PYEOF
