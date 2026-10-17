[CmdletBinding()]
param(
    [string] $Methods = "Baseline,ERC,TSCL",
    [int[]] $Seeds = (42..53),
    [switch] $AllowMissing,
    [switch] $Overwrite,
    [switch] $DryRun,
    [string] $PythonExe
)
$ErrorActionPreference = "Stop"
$methodList = @($Methods -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
& (Join-Path $PSScriptRoot "evaluate_campaign.ps1") -Tasks Cabinet -Methods $methodList -Seeds $Seeds -AllowMissing:$AllowMissing -Overwrite:$Overwrite -DryRun:$DryRun -PythonExe $PythonExe
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
