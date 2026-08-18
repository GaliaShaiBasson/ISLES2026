# Phase 3: lesion-aware sampling. Compares uniform vs inverse-volume case
# sampling, holding the loss function fixed (default Dice+CE).
#
# Usage:
#   $env:ISLES26_CASE_METADATA_CSV = "C:\path\to\case_metadata.csv"
#   .\run_sampling_experiment.ps1 -DatasetId 1 -Fold 0 -Config 3d_fullres
param(
    [Parameter(Mandatory=$true)][int]$DatasetId,
    [int]$Fold = 0,
    [string]$Config = "3d_fullres"
)

$ErrorActionPreference = "Stop"

if (-not $env:ISLES26_CASE_METADATA_CSV) {
    Write-Error "Set `$env:ISLES26_CASE_METADATA_CSV to your case_metadata.csv path first."
    exit 1
}

Write-Host "== Training with lesion-aware sampling (fold $Fold, config $Config) =="
nnUNetv2_train $DatasetId $Config $Fold -tr nnUNetTrainerLesionAwareSampling

Write-Host "== Done =="
Write-Host "Compare against the baseline run (uniform sampling, same fold/config) from run_baseline.ps1."
