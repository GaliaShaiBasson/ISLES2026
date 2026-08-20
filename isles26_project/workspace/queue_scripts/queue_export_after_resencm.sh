#!/bin/bash
# Waits for the currently-running resencm-baseline training (nnUNetv2_train,
# launched via `isles26.py train baseline-500 --dataset-id 2 --plans
# nnUNetResEncUNetMPlans`, PID 2225298 as of 2026-08-20) to fully exit, then
# launches export_val_probabilities.sh on GPU. Same waiting pattern as
# queue_resencm_baseline500.sh/queue_postprocess_after_samplingpow.sh:
# poll for the process, not a fixed sleep -- so this starts the instant the
# GPU is actually free, not on a guessed timer.
#
# GPU, not CPU: per project chat 2026-08-20, CPU inference measured at
# ~131s/case (smoke test) -> ~4-6h for the full 5-trainer x ~120-case
# export. GPU is much faster per case and the earlier "avoid GPU
# contention" concern (CLAUDE.md "Optimizing GPU usage": ~2.1x slowdown)
# was specifically two full TRAINING jobs sharing the GPU -- prediction has
# no backward pass/optimizer step, so it's lighter, and this script waits
# for the training job to finish anyway rather than running concurrently
# with it, so there's no contention question here at all.
#
# Runs export_val_probabilities.sh WITHOUT --parallel: --parallel is
# refused on -device cuda by that script's own guard (parallel GPU jobs
# would recreate the real contention problem the guard exists to prevent)
# -- sequential is correct and already fast enough on GPU.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== queue_export_after_resencm.sh starting ==="
log "--- waiting for nnUNetv2_train (resencm-baseline) to finish ---"
while pgrep -f nnUNetv2_train > /dev/null; do
  sleep 60
done
log "--- GPU free, launching val-probability export ---"

./ensembling/export_val_probabilities.sh --device cuda
status=$?
if [ $status -eq 0 ]; then
  log "=== export_val_probabilities.sh finished successfully ==="
else
  log "=== export_val_probabilities.sh FAILED (exit $status) ==="
fi
exit $status
