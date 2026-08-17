#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ID=${1:?Usage: run_full_cv.sh DATASET_ID}
python "$ROOT/isles26.py" train all --dataset-id "$DATASET_ID" --folds 0 1 2 3 4 --configuration "${CONFIG:-3d_fullres}"

