#!/bin/bash
# Quick status check for an unattended full-pipeline run (run_full_experiment.sh /
# run_500ep_full_experiment.sh style: sequential conditions, each doing
# train -> predict -> evaluate, logged to workspace/logs/full_run*.log).
#
# Usage (run from anywhere -- resolves the project root itself):
#   scripts/check_status.sh              # auto-picks the most recently modified full_run*.log
#   scripts/check_status.sh <path-to-log>  # check a specific log file explicitly
#
# What it reports:
#   - which run log it's reading, and how long it's been running
#   - each condition's status: done / in-progress / not-started-yet
#   - for the in-progress condition: current epoch, latest train/val loss,
#     pseudo dice, best EMA pseudo dice so far, epoch pace, and an ETA for
#     that condition to finish training
#   - whether the driving shell script and an nnUNetv2_train process are
#     actually alive (a stalled/dead run looks very different from this)
#   - any Error/Traceback lines near the end of the log

set -uo pipefail
# Project root, not this script's own directory (scripts/) -- so
# workspace/logs/full_run*.log resolves correctly regardless of where this
# is invoked from.
cd "$(dirname "$0")/.."

LOG="${1:-}"
if [[ -z "$LOG" ]]; then
    LOG=$(ls -t workspace/logs/full_run*.log 2>/dev/null | head -1)
fi
if [[ -z "$LOG" || ! -f "$LOG" ]]; then
    echo "No workspace/logs/full_run*.log found. Pass a log path explicitly."
    exit 1
fi

echo "=== Log: $LOG ==="
START_LINE=$(head -1 "$LOG")
echo "Started: $START_LINE"
echo "Now:     $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo

echo "=== Conditions ==="
# "=== Condition: X (...) ===" marks start, "=== Condition X done ===" marks finish.
grep -nE '=== Condition' "$LOG" | sed -E 's/^([0-9]+):\[([^]]+)\] (.*)$/\2  \3/'
echo

# Figure out the current (last-started, not-yet-done) condition, if any.
LAST_START=$(grep -nE '=== Condition: ' "$LOG" | tail -1)
LAST_DONE=$(grep -nE '=== Condition .* done ===' "$LOG" | tail -1)
LAST_START_LINE=$(echo "$LAST_START" | cut -d: -f1)
LAST_DONE_LINE=$(echo "$LAST_DONE" | cut -d: -f1)

if [[ -n "$LAST_START_LINE" && ( -z "$LAST_DONE_LINE" || "$LAST_START_LINE" -gt "$LAST_DONE_LINE" ) ]]; then
    COND_NAME=$(echo "$LAST_START" | sed -E 's/.*Condition: ([a-zA-Z0-9_-]+).*/\1/')
    TRAINER_NAME=$(echo "$LAST_START" | sed -E 's/.*\(([A-Za-z0-9_]+)\).*/\1/')
    TOTAL_EPOCHS=500
    [[ "$TRAINER_NAME" == *250epochs* ]] && TOTAL_EPOCHS=250
    echo "=== In progress: $COND_NAME ($TRAINER_NAME, $TOTAL_EPOCHS-epoch budget) ==="

    # Latest epoch block (nnU-Net's own per-epoch log lines).
    EPOCH_LINE=$(grep -E '^20.*: Epoch [0-9]+$' "$LOG" | tail -1)
    EPOCH_NUM=$(echo "$EPOCH_LINE" | grep -oE 'Epoch [0-9]+' | grep -oE '[0-9]+')
    TRAIN_LOSS=$(grep -E 'train_loss' "$LOG" | tail -1)
    VAL_LOSS=$(grep -E 'val_loss' "$LOG" | tail -1)
    PSEUDO_DICE=$(grep -E 'Pseudo dice' "$LOG" | tail -1)
    EPOCH_TIME=$(grep -E 'Epoch time' "$LOG" | tail -1)
    BEST_EMA=$(grep -E 'New best EMA pseudo Dice' "$LOG" | tail -1)

    [[ -n "$EPOCH_NUM" ]] && echo "Epoch: $EPOCH_NUM / $TOTAL_EPOCHS"
    [[ -n "$TRAIN_LOSS" ]] && echo "  $TRAIN_LOSS" | sed -E 's/^[0-9-]+ [0-9:.]+: //'
    [[ -n "$VAL_LOSS" ]] && echo "  $VAL_LOSS" | sed -E 's/^[0-9-]+ [0-9:.]+: //'
    [[ -n "$PSEUDO_DICE" ]] && echo "  $PSEUDO_DICE" | sed -E 's/^[0-9-]+ [0-9:.]+: //'
    [[ -n "$EPOCH_TIME" ]] && echo "  $EPOCH_TIME" | sed -E 's/^[0-9-]+ [0-9:.]+: //'
    [[ -n "$BEST_EMA" ]] && echo "  Best so far: $BEST_EMA" | sed -E 's/^[0-9-]+ [0-9:.]+: //'

    # Rough ETA: average the last 20 "Epoch time: X s" values, multiply by
    # epochs remaining to 500 (safe overestimate if actually a 250-epoch trainer).
    AVG_SEC=$(grep -oE 'Epoch time: [0-9.]+ s' "$LOG" | tail -20 | grep -oE '[0-9.]+' | awk '{s+=$1; n++} END {if (n>0) printf "%.1f", s/n}')
    if [[ -n "$EPOCH_NUM" && -n "$AVG_SEC" && "$EPOCH_NUM" -lt "$TOTAL_EPOCHS" ]]; then
        REMAIN=$((TOTAL_EPOCHS - EPOCH_NUM))
        ETA_SEC=$(awk -v a="$AVG_SEC" -v r="$REMAIN" 'BEGIN{printf "%.0f", a*r}')
        ETA_H=$(awk -v s="$ETA_SEC" 'BEGIN{printf "%.1f", s/3600}')
        echo "  ~${AVG_SEC}s/epoch avg -> ${REMAIN} epochs left, ~${ETA_H}h to finish training"
    fi
    echo "(after training finishes: predict + evaluate typically adds ~15-25 min based on prior conditions)"
else
    echo "=== No condition currently in progress (either finished, or waiting on GPU) ==="
fi
echo

echo "=== Process check ==="
DRIVER_PROC=$(ps aux | grep -E "run_(500ep_)?full_experiment.sh" | grep -v grep)
TRAIN_PROC=$(ps aux | grep "nnUNetv2_train" | grep -v grep | head -1)
if [[ -n "$DRIVER_PROC" ]]; then
    echo "Driver script: RUNNING"
    echo "  $DRIVER_PROC" | awk '{print "  pid", $2, "started", $9}'
else
    echo "Driver script: NOT running (finished, or died -- check log tail below)"
fi
if [[ -n "$TRAIN_PROC" ]]; then
    TRAINER=$(echo "$TRAIN_PROC" | grep -oE '\-tr [A-Za-z0-9_]+' | awk '{print $2}')
    echo "Training process: RUNNING ($TRAINER)"
else
    echo "Training process: not running right now"
fi
echo

echo "=== Errors near end of log (if any) ==="
ERR=$(tail -300 "$LOG" | grep -iE 'error|traceback|exception' | grep -viE 'nnUNet_n_proc|error_correction')
if [[ -n "$ERR" ]]; then
    echo "$ERR"
else
    echo "(none found in last 300 lines)"
fi
echo

echo "=== Last 10 log lines ==="
tail -10 "$LOG"
