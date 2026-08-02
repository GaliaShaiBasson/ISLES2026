param([Parameter(Mandatory=$true)][int]$DatasetId, [int]$Fold = 0, [string]$Config = "3d_fullres")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
python (Join-Path $Root "isles26.py") train sampling --dataset-id $DatasetId --fold $Fold --configuration $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
