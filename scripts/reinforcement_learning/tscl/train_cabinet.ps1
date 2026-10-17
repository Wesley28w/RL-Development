[CmdletBinding()]
param(
    [string] $Methods = "Baseline,ERC,TSCL",
    [int[]] $Seeds = (43..52),
    [int] $NumEnvs = 16384,
    [int] $MaxIterations = 1500,
    [string] $MethodDirectorySuffix = "_16384env_1500iter",
    [switch] $DryRun,
    [string] $PythonExe
)
$ErrorActionPreference = "Stop"
$methodList = @($Methods -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
& (Join-Path $PSScriptRoot "train_campaign.ps1") -Tasks Cabinet -Methods $methodList -Seeds $Seeds `
    -CabinetNumEnvs $NumEnvs -CabinetIterations $MaxIterations `
    -MethodDirectorySuffix $MethodDirectorySuffix -DryRun:$DryRun -PythonExe $PythonExe
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
