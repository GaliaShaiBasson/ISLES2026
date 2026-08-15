#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ID=${1:?Usage: run_loss_experiments.sh DATASET_ID [FOLD]}
FOLD=${2:-0}
python "$ROOT/isles26.py" train focal --dataset-id "$DATASET_ID" --fold "$FOLD" --configuration "${CONFIG:-3d_fullres}"
python "$ROOT/isles26.py" train tversky --dataset-id "$DATASET_ID" --fold "$FOLD" --configuration "${CONFIG:-3d_fullres}"
