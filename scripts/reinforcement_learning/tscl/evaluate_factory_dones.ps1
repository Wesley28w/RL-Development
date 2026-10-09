[CmdletBinding()]
param(
    [ValidateSet("Baseline", "ERC", "TSCL")]
    [string[]] $Methods = @("Baseline", "ERC", "TSCL"),
    [int[]] $Seeds = (42..57),
    [ValidateRange(1, 1000000)]
    [int] $Episodes = 2000,
    [switch] $AllowMissing,
    [switch] $DryRun,
    [string] $JournalRoot,
    [string] $PythonExe
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$baselineErcLauncher = Join-Path $PSScriptRoot "evaluate_baseline_erc.ps1"
$tsclLauncher = Join-Path $PSScriptRoot "evaluate_tscl.ps1"
$resultsRoot = Join-Path $repoRoot "logs\journal\full_eval"

# Keep the evaluation protocol identical across all three methods. Factory's saved
# RL-Games player setting remains stochastic, matching the prior evaluation protocol.
$common = @{
    Tasks = @("Factory")
    Seeds = $Seeds
    Episodes = $Episodes
    NumEnvs = 512
    EvalSeed = 12345
    AllowMissing = $AllowMissing
    # This launcher exists specifically to replace the earlier mislabeled Factory
    # evaluation artifacts. Its task filter prevents Lift or Cabinet files from
    # being touched.
    Overwrite = $true
    DryRun = $DryRun
}
if (-not [string]::IsNullOrWhiteSpace($JournalRoot)) { $common.JournalRoot = $JournalRoot }
if (-not [string]::IsNullOrWhiteSpace($PythonExe)) { $common.PythonExe = $PythonExe }

Write-Host "[PROTOCOL] Factory Nut-Thread only"
Write-Host "[PROTOCOL] $Episodes completed evaluation episodes per checkpoint; 512 environments"
Write-Host "[PROTOCOL] evaluation seed 12345; original reset distribution; curriculum disabled"
Write-Host "[METRICS]  per-step dones/success_rate, 128-step RL-Games windows, and terminal success"
Write-Host "[OUTPUT]   $resultsRoot"
Write-Host "[REPLACE]  Existing Factory results only; Lift and Cabinet are untouched"

$comparisonMethods = @($Methods | Where-Object { $_ -in @("Baseline", "ERC") })
if ($comparisonMethods.Count -gt 0) {
    $arguments = @{}
    foreach ($entry in $common.GetEnumerator()) { $arguments[$entry.Key] = $entry.Value }
    $arguments.Methods = $comparisonMethods
    $arguments.ResultsRoot = $resultsRoot
    $arguments.AllowProtocolOverride = $true
    & $baselineErcLauncher @arguments
}

if ($Methods -contains "TSCL") {
    $arguments = @{}
    foreach ($entry in $common.GetEnumerator()) { $arguments[$entry.Key] = $entry.Value }
    $arguments.ResultsRoot = Join-Path $resultsRoot "tscl"
    & $tsclLauncher @arguments
}

if ($DryRun) {
    Write-Host "[DONE] Factory corrected-metric dry-run completed; no evaluation was launched."
} else {
    Write-Host "[DONE] Corrected Factory results replaced under the existing full_eval method folders"
}
