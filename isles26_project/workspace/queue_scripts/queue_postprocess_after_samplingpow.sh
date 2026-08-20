#!/bin/bash
# Waits for the already-running /tmp/queue_samplingpow.sh (train sampling-pow-500 ->
# predict -> evaluate -> aggregate -> plot, queued behind baseline-wideaug-500) to
# fully exit, then launches run_postprocess_grid.sh. Requested explicitly so the
# CPU-only post-processing grid doesn't run until the whole night's GPU pipeline --
# including sampling-pow-500's own dataloader-worker CPU usage -- is done, not just
# until the GPU itself frees up.
set -u
cd /home/galia/ISLES2026/isles26_project || exit 1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== waiting for queue_samplingpow.sh to finish ==="
while pgrep -f "queue_samplingpow.sh" > /dev/null; do
  sleep 60
done
log "=== queue_samplingpow.sh process gone -- launching post-processing grid ==="

bash run_postprocess_grid.sh
