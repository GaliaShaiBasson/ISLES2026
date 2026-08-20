#!/bin/bash
# Overfit-a-tiny-subset sanity gate (training-tips checklist item 1 -- see
# PROJECT_REVIEW.md "Checklist review" / "two runs remain" entries).
#
# Runs one or more *_OverfitCheck trainers (custom_trainers/nnUNetTrainerOverfitCheck.py)
# against Dataset999_ATLASsample (the project's small sample dataset) for a few
# minutes each, then reports whether Pseudo Dice/train_loss show a clear
# learning trend -- the same check that would have caught the 2026-08-17
# sampling-collapse bug in minutes instead of after 96 wasted epochs. Run this
# against any newly-touched trainer BEFORE launching it for real.
#
# Usage:
#   ./sanity_overfit_check.sh                        # runs the default group below
#   ./sanity_overfit_check.sh overfit-check-wideaug   # one TRAINER_GROUPS key
#   ./sanity_overfit_check.sh overfit-check-wideaug overfit-check-samplingpow
#
# Each argument must be a key in isles26.py's TRAINER_GROUPS (add one there,
# and a matching *_OverfitCheck class in nnUNetTrainerOverfitCheck.py, for
# every new trainer that needs gating).
#
# Not a real experiment result -- disposable checkpoints, same spirit as
# `train debug`.

set -uo pipefail
cd "$(dirname "$0")"

DATASET_ID=999
DATASET_NAME=ATLASsample
DEFAULT_GROUPS=(overfit-check-wideaug)

SELECTED_GROUPS=("$@")
if [[ ${#SELECTED_GROUPS[@]} -eq 0 ]]; then
    SELECTED_GROUPS=("${DEFAULT_GROUPS[@]}")
fi

# Pseudo Dice values below this are read as "flat" (no real learning signal)
# rather than a genuine trend -- deliberately loose, since this run is short
# by design (see nnUNetTrainerOverfitCheck.py) and isn't meant to reach a high
# absolute Dice, only to show it's clearly moving.
DICE_TREND_MIN=0.15

overall_status=0

for group in "${SELECTED_GROUPS[@]}"; do
    echo
    echo "=== ${group} (Dataset${DATASET_ID}_${DATASET_NAME}) ==="

    if ! python isles26.py train "$group" \
        --dataset-id "$DATASET_ID" --dataset-name "$DATASET_NAME" \
        --configuration 3d_fullres --device cuda --overwrite; then
        echo "[FAIL] ${group}: training command itself failed -- see output above."
        overall_status=1
        continue
    fi

    trainer_class=$(python - "$group" <<'PY'
import sys
sys.path.insert(0, ".")
from isles26 import TRAINER_GROUPS
group = sys.argv[1]
print(TRAINER_GROUPS[group][0])
PY
)

    log_dir=$(python - "$trainer_class" "$DATASET_ID" "$DATASET_NAME" <<'PY'
import sys
sys.path.insert(0, ".")
from isles26 import build_environment, _output_folder
trainer, dataset_id, dataset_name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
env = build_environment()
print(_output_folder(trainer, dataset_id, dataset_name, "3d_fullres", "0", env))
PY
)

    log_file=$(ls -t "${log_dir}"/training_log_*.txt 2>/dev/null | head -1)
    if [[ -z "$log_file" ]]; then
        echo "[FAIL] ${group}: no training log found under ${log_dir}"
        overall_status=1
        continue
    fi

    # Pseudo dice line looks like: "... Pseudo dice [np.float32(0.4667)] "
    dice_values=$(grep -oP 'Pseudo dice \[np\.float32\(\K[0-9.]+' "$log_file")
    loss_values=$(grep -oP 'train_loss \K-?[0-9.]+' "$log_file")

    first_dice=$(echo "$dice_values" | head -1)
    last_dice=$(echo "$dice_values" | tail -1)
    first_loss=$(echo "$loss_values" | head -1)
    last_loss=$(echo "$loss_values" | tail -1)

    echo "  log: ${log_file}"
    echo "  Pseudo dice: ${first_dice:-n/a} -> ${last_dice:-n/a}"
    echo "  train_loss:  ${first_loss:-n/a} -> ${last_loss:-n/a}"

    if [[ -z "$first_dice" || -z "$last_dice" ]]; then
        echo "[FAIL] ${group}: could not parse Pseudo dice from the log -- inspect manually."
        overall_status=1
        continue
    fi

    dice_delta=$(python3 -c "print(${last_dice} - ${first_dice})")
    trend_ok=$(python3 -c "print(1 if ${dice_delta} >= ${DICE_TREND_MIN} else 0)")

    if [[ "$trend_ok" == "1" ]]; then
        echo "[PASS] ${group}: Pseudo dice rose by ${dice_delta} (>= ${DICE_TREND_MIN}) -- clear learning signal."
    else
        echo "[WARN] ${group}: Pseudo dice only rose by ${dice_delta} (< ${DICE_TREND_MIN})."
        echo "       Flat/stuck Pseudo dice on a tiny dataset usually means a real bug -- see"
        echo "       training-tips checklist item 1 (bad labels/loss/tensor-shape/normalization/"
        echo "       augmentation/LR, or output not aligned with target) before trusting this"
        echo "       trainer with real GPU time."
        overall_status=1
    fi
done

echo
if [[ $overall_status -eq 0 ]]; then
    echo "All sanity checks passed -- safe to launch the real run(s)."
else
    echo "At least one sanity check did not clearly pass -- investigate before launching tonight's run."
fi
exit $overall_status
