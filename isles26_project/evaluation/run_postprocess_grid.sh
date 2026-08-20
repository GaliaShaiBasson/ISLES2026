#!/bin/bash
# Generalized connected-component post-processing grid search -- works
# against ANY trainer's val predictions (or an ensemble's evaluated masks),
# not hardcoded to baseline-500. See scripts/run_postprocess_grid.sh (project root)
# for the original baseline-500-specific run this was generalized from --
# left untouched as the historical record of that already-completed search,
# not superseded in place.
#
# Two-phase, test-set-leakage-safe by construction, same discipline as the
# original:
#   Phase 1 (search, ALWAYS runs): every (min_voxels, connectivity) combo is
#     filtered + scored against --val-pred-dir vs --val-gt-dir. Never touches
#     test.
#   Phase 2 (frozen check, OPT-IN via --apply-to-test): the single best combo
#     by val hd95_mean is applied ONCE to --test-pred-dir vs --test-gt-dir.
#     Deliberately NOT automatic here (unlike the original baseline-500
#     script) -- per the val-first finalist-selection protocol worked out in
#     project chat 2026-08-20, post-processing search for a NEW candidate
#     model/ensemble should stay val-only while it's still being compared
#     against other shortlist candidates. Only pass --apply-to-test once this
#     IS the locked final recipe.
#
# Usage:
#   ./evaluation/run_postprocess_grid.sh \
#     --tag wideaug500 \
#     --val-pred-dir /path/to/fold_0/validation \
#     --val-gt-dir /home/galia/ISLES2026/nnUNet_raw/Dataset002_ATLAS/labelsTr \
#     --out-root workspace/predictions/postprocess_grid/wideaug500
#
#   # once this candidate IS the frozen final pick:
#   ./evaluation/run_postprocess_grid.sh --tag wideaug500 --val-pred-dir ... \
#     --apply-to-test --test-pred-dir .../predTs --test-gt-dir .../labelsTs \
#     --out-root workspace/predictions/postprocess_grid/wideaug500
set -uo pipefail

PY=/home/galia/miniconda3/envs/isles2026/bin/python
cd "$(dirname "$0")/.." || exit 1

TAG=""
VAL_PRED_DIR=""
VAL_GT_DIR="/home/galia/ISLES2026/nnUNet_raw/Dataset002_ATLAS/labelsTr"
TEST_PRED_DIR=""
TEST_GT_DIR="/home/galia/ISLES2026/nnUNet_raw/Dataset002_ATLAS/labelsTs"
OUT_ROOT=""
META="workspace/splits_dataset002/manifest.csv"
APPLY_TO_TEST=0
MIN_VOXELS_GRID=(1 2 3 4 5 6 8 10 15 20)
CONNECTIVITY_GRID=(1 2 3)
MAX_ATTEMPTS=3

while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --val-pred-dir) VAL_PRED_DIR="$2"; shift 2 ;;
    --val-gt-dir) VAL_GT_DIR="$2"; shift 2 ;;
    --test-pred-dir) TEST_PRED_DIR="$2"; shift 2 ;;
    --test-gt-dir) TEST_GT_DIR="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --meta) META="$2"; shift 2 ;;
    --apply-to-test) APPLY_TO_TEST=1; shift ;;
    --min-voxels-grid) IFS=' ' read -r -a MIN_VOXELS_GRID <<< "$2"; shift 2 ;;
    --connectivity-grid) IFS=' ' read -r -a CONNECTIVITY_GRID <<< "$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [ -z "$TAG" ] || [ -z "$VAL_PRED_DIR" ] || [ -z "$OUT_ROOT" ]; then
  echo "Usage: $0 --tag TAG --val-pred-dir DIR --out-root DIR [--val-gt-dir DIR] [--apply-to-test --test-pred-dir DIR --test-gt-dir DIR]" >&2
  exit 1
fi
if [ "$APPLY_TO_TEST" -eq 1 ] && { [ -z "$TEST_PRED_DIR" ] || [ -z "$TEST_GT_DIR" ]; }; then
  echo "--apply-to-test requires --test-pred-dir and --test-gt-dir" >&2
  exit 1
fi
if [ ! -d "$VAL_PRED_DIR" ] || [ ! -d "$VAL_GT_DIR" ]; then
  log "[FATAL] val pred/gt dir not found: $VAL_PRED_DIR / $VAL_GT_DIR"
  exit 1
fi

VAL_ROOT="$OUT_ROOT/val_search"
TEST_ROOT="$OUT_ROOT/test_final"
mkdir -p "$VAL_ROOT"

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
  [ "$status" -ne 0 ] && log "[error] $desc exhausted $MAX_ATTEMPTS attempts -- giving up on this step, continuing"
  return "$status"
}

log "=== post-processing grid starting: $TAG ==="
log "search grid: min_voxels in (${MIN_VOXELS_GRID[*]}) x connectivity in (${CONNECTIVITY_GRID[*]}) "
log "= $((${#MIN_VOXELS_GRID[@]} * ${#CONNECTIVITY_GRID[@]})) combos, on val ($VAL_PRED_DIR)"

manifest_csv="$OUT_ROOT/grid_manifest.csv"
echo "min_voxels,connectivity,tag,filtered_dir,results_csv,config_json" > "$manifest_csv"
for c in "${CONNECTIVITY_GRID[@]}"; do
  for mv in "${MIN_VOXELS_GRID[@]}"; do
    combo_tag="cc${mv}_conn${c}"
    echo "${mv},${c},${combo_tag},${VAL_ROOT}/${combo_tag}_pred,${VAL_ROOT}/results_${combo_tag}.csv,${VAL_ROOT}/${combo_tag}_config.json" >> "$manifest_csv"
  done
done
log "wrote $manifest_csv"

for c in "${CONNECTIVITY_GRID[@]}"; do
  for mv in "${MIN_VOXELS_GRID[@]}"; do
    combo_tag="cc${mv}_conn${c}"
    filtered_dir="$VAL_ROOT/${combo_tag}_pred"
    results_csv="$VAL_ROOT/results_${combo_tag}.csv"
    config_json="$VAL_ROOT/${combo_tag}_config.json"

    log "--- val: min_voxels=$mv connectivity=$c (tag=$combo_tag) ---"
    cat > "$config_json" <<EOF
{
  "tag": "$combo_tag",
  "min_voxels": $mv,
  "connectivity": $c,
  "split": "val",
  "source_pred_dir": "$VAL_PRED_DIR",
  "gt_dir": "$VAL_GT_DIR",
  "generated_at": "$(date -Iseconds)"
}
EOF

    if [ -f "$filtered_dir/postprocess_summary.csv" ]; then
      log "[skip] $filtered_dir already filtered"
    else
      run_with_retries "postprocess $combo_tag (val)" \
        "$PY" evaluation/postprocess_predictions.py \
        --pred-dir "$VAL_PRED_DIR" --out-dir "$filtered_dir" \
        --min-voxels "$mv" --connectivity "$c"
    fi

    if [ -f "$results_csv" ]; then
      log "[skip] $results_csv already exists"
    else
      run_with_retries "compute_metrics $combo_tag (val)" \
        "$PY" evaluation/compute_metrics.py \
        --pred-dir "$filtered_dir" --gt-dir "$VAL_GT_DIR" --case-metadata-csv "$META" \
        --experiment-name "${TAG}_${combo_tag}_val" --out-csv "$results_csv"
    fi
  done
done

log "--- building val grid summary + selecting winner ---"
"$PY" evaluation/summarize_postprocess_grid.py --val-root "$VAL_ROOT" --out-root "$OUT_ROOT"

if [ ! -f "$OUT_ROOT/selected_combo.json" ]; then
  log "[FATAL] no combo was selected (val search produced no usable results)"
  exit 1
fi

if [ "$APPLY_TO_TEST" -eq 0 ]; then
  log "=== $TAG post-processing search finished (val only, --apply-to-test not passed) ==="
  log "  $OUT_ROOT/grid_summary_val.csv"
  log "  $OUT_ROOT/selected_combo.json"
  log "Test set NOT touched. Re-run with --apply-to-test once this is the locked final recipe."
  exit 0
fi

if [ ! -d "$TEST_PRED_DIR" ] || [ ! -d "$TEST_GT_DIR" ]; then
  log "[FATAL] test pred/gt dir not found: $TEST_PRED_DIR / $TEST_GT_DIR"
  exit 1
fi
mkdir -p "$TEST_ROOT"

sel_mv=$("$PY" -c "import json; print(json.load(open('$OUT_ROOT/selected_combo.json'))['min_voxels'])")
sel_conn=$("$PY" -c "import json; print(json.load(open('$OUT_ROOT/selected_combo.json'))['connectivity'])")
sel_tag=$("$PY" -c "import json; print(json.load(open('$OUT_ROOT/selected_combo.json'))['tag'])")

log "=== Phase 2: applying selected combo ($sel_tag) ONCE to test ($TEST_PRED_DIR) ==="

test_filtered_dir="$TEST_ROOT/${sel_tag}_pred"
test_results_csv="$TEST_ROOT/results_${sel_tag}_test.csv"
test_config_json="$TEST_ROOT/${sel_tag}_config.json"

cat > "$test_config_json" <<EOF
{
  "tag": "$sel_tag",
  "min_voxels": $sel_mv,
  "connectivity": $sel_conn,
  "split": "test",
  "source_pred_dir": "$TEST_PRED_DIR",
  "gt_dir": "$TEST_GT_DIR",
  "selected_from": "$OUT_ROOT/selected_combo.json",
  "generated_at": "$(date -Iseconds)"
}
EOF

if [ -f "$test_filtered_dir/postprocess_summary.csv" ]; then
  log "[skip] $test_filtered_dir already filtered"
else
  run_with_retries "postprocess $sel_tag (test)" \
    "$PY" evaluation/postprocess_predictions.py \
    --pred-dir "$TEST_PRED_DIR" --out-dir "$test_filtered_dir" \
    --min-voxels "$sel_mv" --connectivity "$sel_conn"
fi

if [ -f "$test_results_csv" ]; then
  log "[skip] $test_results_csv already exists"
else
  run_with_retries "compute_metrics $sel_tag (test)" \
    "$PY" evaluation/compute_metrics.py \
    --pred-dir "$test_filtered_dir" --gt-dir "$TEST_GT_DIR" --case-metadata-csv "$META" \
    --experiment-name "${TAG}_${sel_tag}_test" --out-csv "$test_results_csv"
fi

raw_test_results_csv="$TEST_ROOT/results_raw_test.csv"
if [ -f "$raw_test_results_csv" ]; then
  log "[skip] $raw_test_results_csv already exists"
else
  run_with_retries "compute_metrics raw (test)" \
    "$PY" evaluation/compute_metrics.py \
    --pred-dir "$TEST_PRED_DIR" --gt-dir "$TEST_GT_DIR" --case-metadata-csv "$META" \
    --experiment-name "${TAG}_raw_test" --out-csv "$raw_test_results_csv"
fi

log "=== $TAG post-processing grid finished (val search + one frozen test check) ==="
log "  $OUT_ROOT/grid_summary_val.csv"
log "  $OUT_ROOT/selected_combo.json"
log "  $test_results_csv"
log "  $raw_test_results_csv"
