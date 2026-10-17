[CmdletBinding()]
param(
    [ValidateSet("Baseline", "ERC", "TSCL")]
    [string[]] $Methods = @("Baseline", "ERC", "TSCL"),
    [ValidateSet("Lift", "Cabinet", "Factory")]
    [string[]] $Tasks = @("Lift", "Cabinet", "Factory"),
    [int[]] $Seeds = (42..53),
    [switch] $AllowMissing,
    [switch] $Overwrite,
    [switch] $DryRun,
    [string] $JournalRoot,
    [string] $PythonExe
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$baselineErcLauncher = Join-Path $PSScriptRoot "evaluate_baseline_erc.ps1"
$tsclLauncher = Join-Path $PSScriptRoot "evaluate_tscl.ps1"

# This benchmark protocol is deliberately fixed so all methods are directly comparable.
$episodes = 10000
$numEnvs = 512
$evalSeed = 12345
$fullEvalRoot = Join-Path $repoRoot "logs\journal\full_eval"

Write-Host "[PROTOCOL] 10,000 completed episodes per checkpoint"
Write-Host "[PROTOCOL] 512 parallel environments; evaluation seed 12345"
Write-Host "[PROTOCOL] deterministic RSL-RL; saved stochastic RL-Games Factory player"
Write-Host "[PROTOCOL] original task reset distribution; curriculum disabled during evaluation"
Write-Host "[OUTPUT]   $fullEvalRoot"

$common = @{
    Tasks = $Tasks
    Seeds = $Seeds
    Episodes = $episodes
    NumEnvs = $numEnvs
    EvalSeed = $evalSeed
    AllowMissing = $AllowMissing
    Overwrite = $Overwrite
    DryRun = $DryRun
}
if (-not [string]::IsNullOrWhiteSpace($JournalRoot)) { $common.JournalRoot = $JournalRoot }
if (-not [string]::IsNullOrWhiteSpace($PythonExe)) { $common.PythonExe = $PythonExe }

$comparisonMethods = @($Methods | Where-Object { $_ -in @("Baseline", "ERC") })
if ($comparisonMethods.Count -gt 0) {
    $comparisonArgs = @{}
    foreach ($entry in $common.GetEnumerator()) { $comparisonArgs[$entry.Key] = $entry.Value }
    $comparisonArgs.Methods = $comparisonMethods
    $comparisonArgs.ResultsRoot = $fullEvalRoot
    # Explicitly acknowledge the fixed 10,000-episode campaign protocol.
    $comparisonArgs.AllowProtocolOverride = $true
    & $baselineErcLauncher @comparisonArgs
}

if ($Methods -contains "TSCL") {
    $tsclArgs = @{}
    foreach ($entry in $common.GetEnumerator()) { $tsclArgs[$entry.Key] = $entry.Value }
    $tsclArgs.ResultsRoot = Join-Path $fullEvalRoot "tscl"
    & $tsclLauncher @tsclArgs
}

if ($DryRun) {
    Write-Host "[DONE] Full benchmark dry-run preflight completed; no evaluation was launched."
} else {
    Write-Host "[DONE] Requested full benchmark evaluations completed under $fullEvalRoot"
}
