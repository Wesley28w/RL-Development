[CmdletBinding()]
param(
    [int[]] $Seeds = (42..57),
    [int] $CabinetNumEnvs = 16384,
    [int] $CabinetIterations = 1500,
    [int] $LiftNumEnvs = 4096,
    [int] $LiftIterations = 1500,
    [int] $FactoryNumEnvs = 512,
    [int] $FactoryIterations = 75,
    [switch] $DryRun,
    [string] $PythonExe
)

$ErrorActionPreference = "Stop"

& (Join-Path $PSScriptRoot "train_campaign.ps1") `
    -Tasks Cabinet,Lift,Factory `
    -Methods ERC `
    -Seeds $Seeds `
    -CabinetNumEnvs $CabinetNumEnvs `
    -CabinetIterations $CabinetIterations `
    -LiftNumEnvs $LiftNumEnvs `
    -LiftIterations $LiftIterations `
    -FactoryNumEnvs $FactoryNumEnvs `
    -FactoryIterations $FactoryIterations `
    -AppendBudgetToMethodDirectory `
    -DryRun:$DryRun `
    -PythonExe $PythonExe

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
