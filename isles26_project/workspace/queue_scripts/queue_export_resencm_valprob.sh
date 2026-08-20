#!/bin/bash
# Exports val-set softmax probabilities for both real ResEnc M runs done so far
# -- nnUNetTrainerBaseline_500epochs and nnUNetTrainerWideAugBaseline_500epochs,
# both on nnUNetResEncUNetMPlans -- so they can join the finalist shortlist that
# already has val probabilities exported (see ensembling/export_val_probabilities.sh,
# which only covers the 5 default-plans shortlist trainers and has no --plans
# flag, so these ResEnc M combos can't just be added to its TRAINERS list).
#
# Same staging + predict pattern as export_val_probabilities.sh's run_one():
# predicts from the staged val-image symlink folder (never imagesTr directly,
# never nnU-Net's own --val --npz retraining flag -- see stage_val_images.py's
# docstring for why), writes into a brand-new predVal_prob/ per trainer
# (sibling to the existing validation/ and predTs/predTs_prob, neither
# touched), refuses to run for a given trainer if its predVal_prob/ already
# exists unless --overwrite. Sequential, not parallel -- see CLAUDE.md
# "Optimizing GPU usage" (concurrent jobs contend for compute even on CPU).
#
# Deliberately a standalone one-off script, not a generalization of
# export_val_probabilities.sh -- matches this project's existing convention of
# one dedicated queue_*.sh per run/group (queue_resencm_wideaug500.sh,
# queue_resencm_baseline500.sh, queue_resencm_sampling500.sh are all separate
# scripts too), zero risk to the shared script other trainers already depend on.
#
# Waits only for real GPU training (nnUNetv2_train) and queue_resencm_wideaug500.sh's
# own driver process -- NOT for any nnUNetv2_predict process in general. An earlier
# version of this wait condition also matched nnUNetv2_predict, which meant it never
# started while ANY other CPU-only predict export (e.g.
# queue_export_lesionawaresampling_valprob.sh) was still running, even though this
# script defaults to --device cpu itself and the two don't actually contend for GPU
# at all -- plenty of free CPU cores on this box (144 total) for both concurrently.
# Only re-add a predict check here if --device cuda is actually used.
#
# Usage:
#   ./workspace/queue_export_resencm_valprob.sh                  # CPU (default, zero GPU-contention risk)
#   ./workspace/queue_export_resencm_valprob.sh --device cuda    # once GPU is confirmed free
#   ./workspace/queue_export_resencm_valprob.sh --overwrite
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

export nnUNet_extTrainer=/home/galia/ISLES2026/isles26_project
export nnUNet_raw=/home/galia/ISLES2026/nnUNet_raw
export nnUNet_preprocessed=/home/galia/ISLES2026/nnUNet_preprocessed
export nnUNet_results=/home/galia/ISLES2026/nnUNet_results

PY=/home/galia/miniconda3/envs/isles2026/bin/python
PREDICT=/home/galia/miniconda3/envs/isles2026/bin/nnUNetv2_predict
DATASET_ID=2
PLANS=nnUNetResEncUNetMPlans
STAGED_DIR="workspace/predictions/val_images_staged"

TRAINERS=(
  nnUNetTrainerBaseline_500epochs
  nnUNetTrainerWideAugBaseline_500epochs
)

# Default device is CPU deliberately -- pick this over GPU whenever a real
# training job might be using the GPU (check `nvidia-smi`/`ps aux | grep
# nnUNetv2_train` before overriding). Slower, but zero contention risk.
DEVICE="cpu"
OVERWRITE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== queue_export_resencm_valprob.sh starting ==="

log "--- waiting for GPU training AND queue_resencm_wideaug500.sh to finish ---"
# Deliberately does NOT wait on nnUNetv2_predict in general -- that matched ANY
# predict job including unrelated CPU-only exports (e.g.
# queue_export_lesionawaresampling_valprob.sh), which never contend for GPU and
# blocked this script for hours for no reason. Only real GPU users matter here.
while pgrep -f nnUNetv2_train > /dev/null \
   || pgrep -f queue_resencm_wideaug500.sh > /dev/null; do
  sleep 30
done
log "--- clear, proceeding ---"

log "=== staging val images (symlinks only, source untouched) ==="
"$PY" ensembling/stage_val_images.py || { log "[FAIL] staging val images"; exit 1; }

fail=0
for trainer in "${TRAINERS[@]}"; do
  out_dir="$nnUNet_results/Dataset002_ATLAS/${trainer}__${PLANS}__3d_fullres/fold_0/predVal_prob"
  if [ -d "$out_dir" ] && [ "$OVERWRITE" -eq 0 ]; then
    log "[skip] $trainer -- $out_dir already exists (pass --overwrite to redo)"
    continue
  fi
  mkdir -p "$out_dir"
  log "=== predicting $trainer / $PLANS -> $out_dir (device=$DEVICE) ==="
  "$PREDICT" -i "$STAGED_DIR" -o "$out_dir" \
    -d "$DATASET_ID" -c 3d_fullres -tr "$trainer" -p "$PLANS" -f 0 \
    --save_probabilities -device "$DEVICE"
  status=$?
  if [ $status -eq 0 ]; then
    log "=== done $trainer (exit 0) ==="
  else
    log "=== FAILED $trainer (exit $status) ==="
    fail=1
  fi
done

if [ "$fail" -eq 0 ]; then
  log "=== queue_export_resencm_valprob.sh finished, both trainers done ==="
else
  log "=== queue_export_resencm_valprob.sh finished with at least one failure -- check log above ==="
  exit 1
fi
