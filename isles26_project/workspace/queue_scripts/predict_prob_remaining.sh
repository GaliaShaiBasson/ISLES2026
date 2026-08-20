#!/bin/bash
# Runs nnUNetv2_predict --save_probabilities sequentially for the 5 remaining
# finished 500-epoch/Dataset002 trainers (baseline already done separately).
# Writes each into its own predTs_prob/ dir, alongside the untouched predTs/.
# Sequential on purpose -- concurrent nnUNetv2_predict jobs contend for GPU
# compute (see CLAUDE.md "Optimizing GPU usage" finding for the train case;
# same GPU-bound reasoning applies here).
set -uo pipefail

export nnUNet_extTrainer=/home/galia/ISLES2026/isles26_project
export nnUNet_raw=/home/galia/ISLES2026/nnUNet_raw
export nnUNet_preprocessed=/home/galia/ISLES2026/nnUNet_preprocessed
export nnUNet_results=/home/galia/ISLES2026/nnUNet_results

PREDICT=/home/galia/miniconda3/envs/isles2026/bin/nnUNetv2_predict
DATASET_DIR="$nnUNet_raw/Dataset002_ATLAS"

TRAINERS=(
  nnUNetTrainerFocalTversky_500epochs
  nnUNetTrainerTverskyMild_500epochs
  nnUNetTrainerWideAugBaseline_500epochs
  nnUNetTrainerLesionAwareSampling_500epochs_full
  nnUNetTrainerLesionAwareSamplingPow_500epochs_full
)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

for trainer in "${TRAINERS[@]}"; do
  OUT_DIR="$nnUNet_results/Dataset002_ATLAS/${trainer}__nnUNetPlans__3d_fullres/fold_0/predTs_prob"
  mkdir -p "$OUT_DIR"
  log "=== starting $trainer -> $OUT_DIR ==="
  "$PREDICT" -i "$DATASET_DIR/imagesTs" -o "$OUT_DIR" \
    -d 2 -c 3d_fullres -tr "$trainer" -f 0 \
    --save_probabilities
  status=$?
  if [ $status -eq 0 ]; then
    log "=== done $trainer (exit 0) ==="
  else
    log "=== FAILED $trainer (exit $status) ==="
  fi
done

log "=== all remaining trainers processed ==="
