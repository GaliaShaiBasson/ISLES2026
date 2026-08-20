#!/bin/bash
# Unattended connected-component post-processing grid search for baseline-500
# (Dataset002_ATLAS, nnUNetTrainerBaseline_500epochs).
#
# Two-phase, test-set-leakage-safe by construction:
#   Phase 1 (search): every (min_voxels, connectivity) combo is filtered + scored
#     against nnU-Net's own internal VALIDATION predictions (fold_0/validation/,
#     120 cases) vs labelsTr -- never against predTs. This is what a grid search is
#     for: comparing many settings and picking a winner is exactly what "tuning"
#     means, so it must happen on val, not on the held-out test_id/test_ood set
#     (see evaluation/postprocess_predictions.py's own docstring on this).
#   Phase 2 (frozen check): the single best combo by val hd95_mean (see selection
#     rule below) is applied ONCE to predTs (test_id+test_ood, 250 cases) vs
#     labelsTs -- one evaluation, no comparison shopping on the test set.
#
# Safe to run fully in parallel with the currently-training baseline-wideaug-500 job:
# this script (and both scripts it calls) is pure NumPy/SciPy/SimpleITK/pandas -- no
# torch, no CUDA, never touches the GPU. The documented ~2x GPU-contention slowdown in
# CLAUDE.md ("Optimizing GPU usage") is specific to two concurrent GPU training jobs
# and does not apply here.
#
# Non-destructive by construction: every --pred-dir this script passes (validation/,
# predTs) is only ever read. All filtered-mask and result-CSV output lands under
# workspace/predictions/postprocess_grid/baseline500/, a brand-new tree -- nothing under
# nnUNet_results/ or nnUNet_raw/ is written to. postprocess_predictions.py
# additionally hard-refuses if any --out-dir were ever accidentally pointed at or
# inside a --pred-dir (see its own guard).
#
# Traceable by combination: every combo gets its own tag "cc{min_voxels}_conn{c}"
# used consistently for its filtered-mask folder, its results CSV, and a per-combo
# config.json (min_voxels/connectivity/split/source pred_dir/timestamp) written
# alongside -- so any single folder found later is self-describing even in
# isolation. grid_manifest.csv (written up front) and grid_summary_val.csv (written
# at the end) are the two flat indexes over the whole grid.
#
# Idempotent/resumable: each combo's postprocess/compute_metrics call is skipped if
# its output already exists -- a re-run after a partial failure only redoes what
# didn't finish, mirroring run_full_experiment.sh's retry-without-duplication
# convention. No `set -e` for the same reason: one failed combo must not abort the
# rest of the grid.
#
# Usage:
#   nohup ./run_postprocess_grid.sh > workspace/logs/postprocess_grid_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# To check progress while it runs: tail -f workspace/logs/postprocess_grid_*.log
# To stop it: pkill -f run_postprocess_grid.sh

PY=/home/galia/miniconda3/envs/isles2026/bin/python

cd /home/galia/ISLES2026/isles26_project || exit 1

CHECKPOINT_DIR=/home/galia/ISLES2026/nnUNet_results/Dataset002_ATLAS/nnUNetTrainerBaseline_500epochs__nnUNetPlans__3d_fullres/fold_0
VAL_PRED_DIR="$CHECKPOINT_DIR/validation"
VAL_GT_DIR=/home/galia/ISLES2026/nnUNet_raw/Dataset002_ATLAS/labelsTr
TEST_PRED_DIR="$CHECKPOINT_DIR/predTs"
TEST_GT_DIR=/home/galia/ISLES2026/nnUNet_raw/Dataset002_ATLAS/labelsTs
META=workspace/splits_dataset002/manifest.csv
OUT_ROOT=workspace/predictions/postprocess_grid/baseline500
VAL_ROOT="$OUT_ROOT/val_search"
TEST_ROOT="$OUT_ROOT/test_final"

MIN_VOXELS_GRID=(1 2 3 4 5 6 8 10 15 20)
CONNECTIVITY_GRID=(1 2 3)
MAX_ATTEMPTS=3

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Retries a command up to MAX_ATTEMPTS times, matching run_500ep_full_experiment.sh /
# queue_samplingpow.sh's convention (3 attempts, never abort the whole run on one
# failure). Safe to retry blindly here because every command this wraps is already
# idempotent via its own skip-if-output-exists guard (postprocess_predictions.py's
# postprocess_summary.csv check, compute_metrics.py's out-csv check) -- a retry after
# a partial failure only redoes the piece that didn't finish, never duplicates work.
run_with_retries() {
  local desc="$1"; shift
  local attempt=1
  local status=1
  while [ "$attempt" -le "$MAX_ATTEMPTS" ] && [ "$status" -ne 0 ]; do
    "$@"
    status=$?
    if [ "$status" -ne 0 ]; then
      log "[warn] $desc failed (attempt $attempt/$MAX_ATTEMPTS, exit $status)"
      attempt=$((attempt + 1))
      if [ "$attempt" -le "$MAX_ATTEMPTS" ]; then
        sleep 15
      fi
    fi
  done
  if [ "$status" -ne 0 ]; then
    log "[error] $desc exhausted $MAX_ATTEMPTS attempts -- giving up on this step, continuing"
  fi
  return "$status"
}

log "=== post-processing grid starting: baseline-500 ==="
log "search grid: min_voxels in (${MIN_VOXELS_GRID[*]}) x connectivity in (${CONNECTIVITY_GRID[*]}) = $((${#MIN_VOXELS_GRID[@]} * ${#CONNECTIVITY_GRID[@]})) combos, on val"

if [ ! -f "$CHECKPOINT_DIR/checkpoint_final.pth" ]; then
  log "[FATAL] $CHECKPOINT_DIR/checkpoint_final.pth missing -- baseline-500 training did not finish."
  exit 1
fi
for d in "$VAL_PRED_DIR" "$VAL_GT_DIR" "$TEST_PRED_DIR" "$TEST_GT_DIR"; do
  if [ ! -d "$d" ]; then
    log "[FATAL] directory not found: $d"
    exit 1
  fi
done

val_pred_n=$(find "$VAL_PRED_DIR" -maxdepth 1 -name '*.nii.gz' | wc -l)
test_pred_n=$(find "$TEST_PRED_DIR" -maxdepth 1 -name '*.nii.gz' | wc -l)
log "val predictions: $val_pred_n cases | test predictions: $test_pred_n cases"

mkdir -p "$VAL_ROOT" "$TEST_ROOT"

# --- grid_manifest.csv: flat index of every combo's paths, written up front so the
# --- layout is traceable even if the run is interrupted partway through. ---
manifest_csv="$OUT_ROOT/grid_manifest.csv"
echo "min_voxels,connectivity,tag,filtered_dir,results_csv,config_json" > "$manifest_csv"
for c in "${CONNECTIVITY_GRID[@]}"; do
  for mv in "${MIN_VOXELS_GRID[@]}"; do
    tag="cc${mv}_conn${c}"
    echo "${mv},${c},${tag},${VAL_ROOT}/${tag}_predTs,${VAL_ROOT}/results_${tag}.csv,${VAL_ROOT}/${tag}_config.json" >> "$manifest_csv"
  done
done
log "wrote $manifest_csv"

# --- Phase 1: val search ---
for c in "${CONNECTIVITY_GRID[@]}"; do
  for mv in "${MIN_VOXELS_GRID[@]}"; do
    tag="cc${mv}_conn${c}"
    filtered_dir="$VAL_ROOT/${tag}_predTs"
    results_csv="$VAL_ROOT/results_${tag}.csv"
    config_json="$VAL_ROOT/${tag}_config.json"

    log "--- val: min_voxels=$mv connectivity=$c (tag=$tag) ---"

    cat > "$config_json" <<EOF
{
  "tag": "$tag",
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
      run_with_retries "postprocess $tag (val)" \
        "$PY" evaluation/postprocess_predictions.py \
        --pred-dir "$VAL_PRED_DIR" \
        --out-dir "$filtered_dir" \
        --min-voxels "$mv" \
        --connectivity "$c"
    fi

    if [ -f "$results_csv" ]; then
      log "[skip] $results_csv already exists"
    else
      run_with_retries "compute_metrics $tag (val)" \
        "$PY" evaluation/compute_metrics.py \
        --pred-dir "$filtered_dir" \
        --gt-dir "$VAL_GT_DIR" \
        --case-metadata-csv "$META" \
        --experiment-name "baseline500_${tag}_val" \
        --out-csv "$results_csv"
    fi
  done
done

log "--- building val grid summary + selecting winner ---"
"$PY" - "$VAL_ROOT" "$OUT_ROOT" <<'PYEOF'
import json
import sys
from pathlib import Path
import pandas as pd

val_root = Path(sys.argv[1])
out_root = Path(sys.argv[2])

rows = []
for config_json in sorted(val_root.glob("*_config.json")):
    cfg = json.loads(config_json.read_text())
    tag = cfg["tag"]
    results_csv = val_root / f"results_{tag}.csv"
    pp_summary = val_root / f"{tag}_predTs" / "postprocess_summary.csv"
    if not results_csv.is_file():
        print(f"[missing] {results_csv} -- {tag} did not complete, excluded from summary")
        continue
    df = pd.read_csv(results_csv)
    row = {
        "tag": tag,
        "min_voxels": cfg["min_voxels"],
        "connectivity": cfg["connectivity"],
        "n_cases": len(df),
        "dice_mean": df["dice"].mean(),
        "dice_median": df["dice"].median(),
        "hd95_mean": df["hd95_mm"].mean(),
        "hd95_median": df["hd95_mm"].median(),
    }
    if "lesion_f1" in df.columns:
        row["lesion_f1_mean"] = df["lesion_f1"].mean()
    if pp_summary.is_file():
        pp = pd.read_csv(pp_summary)
        row["cases_with_component_removed"] = int((pp["components_removed"] > 0).sum())
        row["total_voxels_removed"] = int((pp["voxels_before"] - pp["voxels_after"]).sum())
    rows.append(row)

if not rows:
    print("[FATAL] no combo produced results -- nothing to summarize")
    sys.exit(1)

summary = pd.DataFrame(rows).round(4).sort_values(["connectivity", "min_voxels"])
summary_csv = out_root / "grid_summary_val.csv"
summary.to_csv(summary_csv, index=False)
print(f"\nWrote {summary_csv}\n")
print(summary.to_string(index=False))

raw = summary[summary.min_voxels == 1].iloc[0]  # min_voxels=1 is the no-op/raw control
# Selection rule: lowest mean HD95 on val -- HD95 is the metric small spurious
# components distort most (see CLAUDE.md / the DiceOnly_250epochs test finding),
# so it's the most informative single criterion for this specific post-processing
# step. Dice is a tiebreaker only since CC filtering barely moves it either way.
best = summary.sort_values(["hd95_mean", "dice_mean"], ascending=[True, False]).iloc[0]
print(f"\nRaw (min_voxels=1) control: dice={raw.dice_mean:.4f} hd95={raw.hd95_mean:.4f}")
print(f"Selected winner (lowest val hd95_mean, dice as tiebreaker): "
      f"tag={best.tag} min_voxels={best.min_voxels:.0f} connectivity={best.connectivity:.0f} "
      f"dice={best.dice_mean:.4f} hd95={best.hd95_mean:.4f}")

(out_root / "selected_combo.json").write_text(json.dumps({
    "tag": best.tag,
    "min_voxels": int(best.min_voxels),
    "connectivity": int(best.connectivity),
    "selection_rule": "lowest val hd95_mean, dice_mean as tiebreaker",
    "val_dice_mean": float(best.dice_mean),
    "val_hd95_mean": float(best.hd95_mean),
    "val_dice_mean_raw_control": float(raw.dice_mean),
    "val_hd95_mean_raw_control": float(raw.hd95_mean),
}, indent=2) + "\n")
print(f"Wrote {out_root / 'selected_combo.json'} -- auto-selected candidate, REVIEW before using in the report.")
PYEOF

if [ ! -f "$OUT_ROOT/selected_combo.json" ]; then
  log "[FATAL] no combo was selected (val search produced no usable results) -- skipping phase 2"
  exit 1
fi

sel_mv=$("$PY" -c "import json; print(json.load(open('$OUT_ROOT/selected_combo.json'))['min_voxels'])")
sel_conn=$("$PY" -c "import json; print(json.load(open('$OUT_ROOT/selected_combo.json'))['connectivity'])")
sel_tag=$("$PY" -c "import json; print(json.load(open('$OUT_ROOT/selected_combo.json'))['tag'])")

log "=== Phase 2: applying selected combo ($sel_tag) ONCE to test_id+test_ood (predTs) ==="

test_filtered_dir="$TEST_ROOT/${sel_tag}_predTs"
test_results_csv="$TEST_ROOT/results_${sel_tag}_test.csv"
test_config_json="$TEST_ROOT/${sel_tag}_config.json"

cat > "$test_config_json" <<EOF
{
  "tag": "$sel_tag",
  "min_voxels": $sel_mv,
  "connectivity": $sel_conn,
  "split": "test_id+test_ood",
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
    --pred-dir "$TEST_PRED_DIR" \
    --out-dir "$test_filtered_dir" \
    --min-voxels "$sel_mv" \
    --connectivity "$sel_conn"
fi

if [ -f "$test_results_csv" ]; then
  log "[skip] $test_results_csv already exists"
else
  run_with_retries "compute_metrics $sel_tag (test)" \
    "$PY" evaluation/compute_metrics.py \
    --pred-dir "$test_filtered_dir" \
    --gt-dir "$TEST_GT_DIR" \
    --case-metadata-csv "$META" \
    --experiment-name "baseline500_${sel_tag}_test" \
    --out-csv "$test_results_csv"
fi

# Also score the untouched raw test predictions once, for a direct before/after row
# in the same place (not a search -- this is the one fixed "raw" comparison point).
raw_test_results_csv="$TEST_ROOT/results_raw_test.csv"
if [ -f "$raw_test_results_csv" ]; then
  log "[skip] $raw_test_results_csv already exists"
else
  run_with_retries "compute_metrics raw (test)" \
    "$PY" evaluation/compute_metrics.py \
    --pred-dir "$TEST_PRED_DIR" \
    --gt-dir "$TEST_GT_DIR" \
    --case-metadata-csv "$META" \
    --experiment-name "baseline500_raw_test" \
    --out-csv "$raw_test_results_csv"
fi

log "=== post-processing grid finished. See: ==="
log "  $OUT_ROOT/grid_summary_val.csv   (all 20 val combos)"
log "  $OUT_ROOT/selected_combo.json    (auto-selected winner, min hd95 on val)"
log "  $test_results_csv                (winner scored once on held-out test)"
log "  $raw_test_results_csv            (raw/unfiltered, same held-out test, for before/after)"
