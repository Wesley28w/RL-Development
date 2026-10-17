[CmdletBinding()]
param(
    [string] $Methods = "Baseline,ERC,TSCL",
    [int[]] $Seeds = (42..57),
    [int] $CabinetNumEnvs = 16384,
    [int] $CabinetIterations = 1500,
    [int] $LiftNumEnvs = 4096,
    [int] $LiftIterations = 1500,
    [int] $FactoryNumEnvs = 512,
    [int] $FactoryIterations = 75,
    [switch] $CoreMethodsOnly,
    [switch] $DryRun,
    [string] $PythonExe
)
$ErrorActionPreference = "Stop"
$methodList = @($Methods -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
& (Join-Path $PSScriptRoot "train_campaign.ps1") -Tasks Cabinet,Lift,Factory -Methods $methodList -Seeds $Seeds `
    -CabinetNumEnvs $CabinetNumEnvs -CabinetIterations $CabinetIterations `
    -LiftNumEnvs $LiftNumEnvs -LiftIterations $LiftIterations `
    -FactoryNumEnvs $FactoryNumEnvs -FactoryIterations $FactoryIterations `
    -AppendBudgetToMethodDirectory -IncludeCabinetAblations:(-not $CoreMethodsOnly) `
    -DryRun:$DryRun -PythonExe $PythonExe
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
