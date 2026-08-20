#!/bin/bash
# Exports val-set softmax probabilities for the finalist shortlist, so
# ensemble_val.py can score real (not proxied) ensemble candidates on val
# before anything touches test_id/test_ood.
#
# NOT auto-run by any pipeline script -- launch this by hand once GPU (or
# CPU, see --device below) time is actually free. Mirrors
# workspace/queue_scripts/predict_prob_remaining.sh's sequential-only pattern (see
# CLAUDE.md "Optimizing GPU usage": concurrent nnUNetv2_predict jobs
# contend for GPU compute even when VRAM looks idle).
#
# Never touches existing results: writes into a brand-new predVal_prob/
# per trainer (sibling to the existing validation/ and predTs_prob/,
# neither of which this script reads or writes), refuses to run if that
# directory already exists unless --overwrite is passed. Predicts from a
# staged symlink folder (ensembling/stage_val_images.py), not imagesTr
# directly, and not via nnU-Net's own `--val --npz` retraining flag --
# see stage_val_images.py's docstring for why (that flag overwrites files
# inside the existing scored validation/ folder; this path never does).
#
# Usage:
#   ./ensembling/export_val_probabilities.sh                     # CPU, sequential, default shortlist
#   ./ensembling/export_val_probabilities.sh --parallel          # CPU, all 5 trainers concurrently
#   ./ensembling/export_val_probabilities.sh --parallel --total-cores 100
#   ./ensembling/export_val_probabilities.sh --device cuda       # once GPU is free (forces sequential --
#                                                                 # see the GPU-contention finding in CLAUDE.md)
#   ./ensembling/export_val_probabilities.sh --overwrite
#
# --parallel launches all shortlisted trainers as background CPU jobs at once
# instead of looping. Safe ONLY on --device cpu: on cuda this would recreate
# exactly the GPU-contention problem CLAUDE.md's "Optimizing GPU usage" entry
# already measured (~2x slowdown from two GPU-bound jobs sharing one GPU) --
# --parallel is refused if --device isn't cpu, not silently downgraded.
# Each job's OMP_NUM_THREADS/MKL_NUM_THREADS is capped to
# floor(--total-cores / n_trainers) so 5 concurrent jobs sum to about
# --total-cores threads, not 5x full-machine oversubscription (the default
# behavior if these were left unset).
set -uo pipefail

export nnUNet_extTrainer=/home/galia/ISLES2026/isles26_project
export nnUNet_raw=/home/galia/ISLES2026/nnUNet_raw
export nnUNet_preprocessed=/home/galia/ISLES2026/nnUNet_preprocessed
export nnUNet_results=/home/galia/ISLES2026/nnUNet_results

PY=/home/galia/miniconda3/envs/isles2026/bin/python
PREDICT=/home/galia/miniconda3/envs/isles2026/bin/nnUNetv2_predict
DATASET_ID=2
DATASET_DIR="$nnUNet_raw/Dataset002_ATLAS"
STAGED_DIR="workspace/predictions/val_images_staged"

# Default device is CPU deliberately -- pick this over GPU whenever a real
# training job might be using the GPU (check `nvidia-smi`/`ps aux | grep
# nnUNetv2_train` before overriding). Slower, but zero contention risk.
DEVICE="cpu"
OVERWRITE=0
PARALLEL=0
TOTAL_CORES=100

# Finalist shortlist from the val-based redundancy analysis
# (workspace/results/finalist_selection/redundancy_recommendation.csv) --
# update this list if the shortlist changes, don't silently diverge from it.
TRAINERS=(
  nnUNetTrainerWideAugBaseline_500epochs
  nnUNetTrainerFocalTversky_500epochs
  nnUNetTrainerTverskyMild_500epochs
  nnUNetTrainerLesionAwareSamplingPow_500epochs_full
  nnUNetTrainerLesionAwareSamplingPowCurriculum_500epochs_full
)

while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --parallel) PARALLEL=1; shift ;;
    --total-cores) TOTAL_CORES="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [ "$PARALLEL" -eq 1 ] && [ "$DEVICE" != "cpu" ]; then
  echo "--parallel is only supported with --device cpu (would recreate real GPU contention" >&2
  echo "on cuda -- see CLAUDE.md 'Optimizing GPU usage'). Refusing rather than silently ignoring." >&2
  exit 1
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== staging val images (symlinks only, source untouched) ==="
"$PY" ensembling/stage_val_images.py || { log "[FAIL] staging val images"; exit 1; }

run_one() {
  local trainer="$1"
  local out_dir="$nnUNet_results/Dataset002_ATLAS/${trainer}__nnUNetPlans__3d_fullres/fold_0/predVal_prob"
  if [ -d "$out_dir" ] && [ "$OVERWRITE" -eq 0 ]; then
    log "[skip] $out_dir already exists (pass --overwrite to redo)"
    return 0
  fi
  mkdir -p "$out_dir"
  log "=== starting $trainer -> $out_dir (device=$DEVICE) ==="
  "$PREDICT" -i "$STAGED_DIR" -o "$out_dir" \
    -d "$DATASET_ID" -c 3d_fullres -tr "$trainer" -f 0 \
    --save_probabilities -device "$DEVICE"
  local status=$?
  if [ $status -eq 0 ]; then
    log "=== done $trainer (exit 0) ==="
  else
    log "=== FAILED $trainer (exit $status) ==="
  fi
  return $status
}

if [ "$PARALLEL" -eq 1 ]; then
  n_trainers=${#TRAINERS[@]}
  threads_per_job=$(( TOTAL_CORES / n_trainers ))
  if [ "$threads_per_job" -lt 1 ]; then threads_per_job=1; fi
  log "=== parallel mode: $n_trainers trainers concurrently, $threads_per_job threads each (~${TOTAL_CORES} cores total) ==="
  export OMP_NUM_THREADS="$threads_per_job"
  export MKL_NUM_THREADS="$threads_per_job"
  pids=()
  for trainer in "${TRAINERS[@]}"; do
    run_one "$trainer" > "workspace/logs/export_val_prob_${trainer}.log" 2>&1 &
    pids+=($!)
    log "launched $trainer (pid $!), log -> workspace/logs/export_val_prob_${trainer}.log"
  done
  fail=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { log "[FAIL] ${TRAINERS[$i]} (pid ${pids[$i]})"; fail=1; }
  done
  [ "$fail" -eq 0 ] && log "=== all $n_trainers trainers finished successfully ===" \
                     || log "=== finished with at least one failure -- check the per-trainer logs above ==="
else
  for trainer in "${TRAINERS[@]}"; do
    run_one "$trainer"
  done
  log "=== all shortlisted trainers processed ==="
fi
