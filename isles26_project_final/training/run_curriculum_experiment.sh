#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ID=${1:?Usage: run_curriculum_experiment.sh DATASET_ID [FOLD]}
FOLD=${2:-0}
exec python "$ROOT/isles26.py" train curriculum --dataset-id "$DATASET_ID" --fold "$FOLD" --configuration "${CONFIG:-3d_fullres}"
