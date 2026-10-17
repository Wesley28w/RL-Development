[CmdletBinding()]
param(
    [string] $Methods = "Baseline,ERC,TSCL",
    [int[]] $Seeds = (42..53),
    [switch] $DryRun,
    [string] $PythonExe
)
$ErrorActionPreference = "Stop"
$methodList = @($Methods -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
& (Join-Path $PSScriptRoot "train_campaign.ps1") -Tasks Lift -Methods $methodList -Seeds $Seeds -DryRun:$DryRun -PythonExe $PythonExe
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
