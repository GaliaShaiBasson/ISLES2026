# Phase 1: baseline nnU-Net v2, default trainer/loss (Dice + CE).
#
# Usage: .\run_baseline.ps1 -DatasetId 1 -Fold 0 -Config 3d_fullres
param(
    [Parameter(Mandatory=$true)][int]$DatasetId,
    [int]$Fold = 0,
    [string]$Config = "3d_fullres"   # use "2d" if CPU-only or VRAM-limited
)

$ErrorActionPreference = "Stop"

Write-Host "== Preprocessing (only needs to run once per dataset) =="
nnUNetv2_plan_and_preprocess -d $DatasetId --verify_dataset_integrity

Write-Host "== Training baseline (fold $Fold, config $Config) =="
nnUNetv2_train $DatasetId $Config $Fold

Write-Host "== Baseline training complete =="
Write-Host "Next: run inference on your held-out set, then evaluation\compute_metrics.py"
