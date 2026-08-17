param([Parameter(Mandatory=$true)][int]$DatasetId, [string]$Config = "3d_fullres")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
python (Join-Path $Root "isles26.py") train all --dataset-id $DatasetId --folds 0 1 2 3 4 --configuration $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
