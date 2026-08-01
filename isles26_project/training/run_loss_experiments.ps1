# Phase 2: loss-function study. Trains one fold per loss variant.
# Requires custom_trainers\install_custom_trainers.py to have been run first.
#
# Usage: .\run_loss_experiments.ps1 -DatasetId 1 -Fold 0 -Config 3d_fullres
param(
    [Parameter(Mandatory=$true)][int]$DatasetId,
    [int]$Fold = 0,
    [string]$Config = "3d_fullres"
)

$ErrorActionPreference = "Stop"

# nnUNetTrainer (default Dice+CE) is already covered by run_baseline.ps1 --
# no need to retrain it here, just reuse those results in the comparison.
$Trainers = @(
    "nnUNetTrainerDiceOnly",
    "nnUNetTrainerFocal",
    "nnUNetTrainerTversky",
    "nnUNetTrainerFocalTversky"
)

foreach ($TR in $Trainers) {
    Write-Host "== Training $TR (fold $Fold, config $Config) =="
    nnUNetv2_train $DatasetId $Config $Fold -tr $TR
}

Write-Host "== All loss variants trained =="
Write-Host "Next: run inference for each trainer's checkpoint, then evaluation\compute_metrics.py"
Write-Host "      followed by evaluation\aggregate_results.py to combine them."
