# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Benchmark sweep: Reverse Curriculum Generation on Franka Lift.
#
# Runs one arm per invocation, over a list of seeds, so that the baseline and the RCG arm are
# matched seed-for-seed: the same seed means the same initial policy weights and the same
# environment randomisation, which is what makes the paired permutation test in
# scripts/rcg/permutation_test.py applicable.
#
# Runs sequentially: each run needs the whole GPU. Safe to leave overnight; a failed seed is logged
# and skipped rather than aborting the sweep.
#
#   .\scripts\rcg\run_lift_seeds.ps1 -Arm baseline
#   .\scripts\rcg\run_lift_seeds.ps1 -Arm rcg
#   .\scripts\rcg\run_lift_seeds.ps1 -Arm rcg -Seeds 42,43 -MaxIterations 50    # quick shakedown

param(
    [ValidateSet("baseline", "rcg")]
    [string] $Arm            = "rcg",
    [int[]]  $Seeds          = @(42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57),
    [int]    $NumEnvs        = 4096,
    [int]    $MaxIterations  = 1500,
    [string] $RunPrefix      = ""
)

$ErrorActionPreference = "Continue"

# resolve the repo root from this script's location, so the sweep can be started from anywhere
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

if ($Arm -eq "rcg") {
    $Task = "Isaac-Lift-Cube-Franka-RCG-v0"
    $Experiment = "franka_lift_rcg"
} else {
    $Task = "Isaac-Lift-Cube-Franka-Baseline-v0"
    $Experiment = "franka_lift_baseline"
}
if ($RunPrefix -eq "") { $RunPrefix = "journal_lift_$Arm" }

$GoalStates = Join-Path $RepoRoot "source\isaaclab_tasks\isaaclab_tasks\manager_based\manipulation\lift\data\goal_states_franka_lift.pt"
if ($Arm -eq "rcg" -and -not (Test-Path $GoalStates)) {
    Write-Host "[FATAL] Goal states not found: $GoalStates" -ForegroundColor Red
    Write-Host "        Record them first:" -ForegroundColor Red
    Write-Host "          isaaclab.bat -p scripts/rcg/record_goal_states.py --task Isaac-Lift-Cube-Franka-RCG-v0 ``" -ForegroundColor Gray
    Write-Host "              --checkpoint logs/rsl_rl/franka_lift_baseline/<run>/model_1499.pt --num_states 1000 --headless" -ForegroundColor Gray
    exit 1
}

$LogRoot = Join-Path $RepoRoot "logs\rcg_sweep"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host " RCG Franka Lift sweep -- arm: $Arm" -ForegroundColor Cyan
Write-Host "   task             : $Task"
Write-Host "   seeds            : $($Seeds -join ', ')"
Write-Host "   envs / iters     : $NumEnvs / $MaxIterations"
Write-Host "   start state      : arm joints + cube pose + the commanded goal (23 values)"
if ($Arm -eq "rcg") { Write-Host "   goal states      : $GoalStates" }
Write-Host "   tensorboard      : logs\rsl_rl\$Experiment"
Write-Host "   console logs     : $LogRoot"
Write-Host "=========================================================================" -ForegroundColor Cyan

$started  = Get-Date
$failures = @()

foreach ($seed in $Seeds) {
    $label   = "$RunPrefix" + "_s$seed"
    $logFile = Join-Path $LogRoot "$label.log"
    Write-Host ""
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] seed $seed -> $logFile" -ForegroundColor Yellow

    $seedStart = Get-Date
    & isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py `
        --task $Task `
        --headless `
        --num_envs $NumEnvs `
        --max_iterations $MaxIterations `
        --seed $seed `
        agent.run_name=$label `
        2>&1 | Tee-Object -FilePath $logFile

    $code    = $LASTEXITCODE
    $elapsed = (Get-Date) - $seedStart
    if ($code -ne 0) {
        Write-Host "[FAIL] seed $seed exited $code after $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Red
        $failures += $seed
    } else {
        Write-Host "[ OK ] seed $seed finished in $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
    }
}

$total = (Get-Date) - $started
Write-Host ""
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host " done in $($total.ToString('hh\:mm\:ss'))   $($Seeds.Count - $failures.Count)/$($Seeds.Count) succeeded"
if ($failures.Count -gt 0) { Write-Host " failed seeds: $($failures -join ', ')" -ForegroundColor Red }
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next -- evaluate every seed of both arms from rho_0 (curriculum OFF), which is the only" -ForegroundColor Cyan
Write-Host "number comparable across arms:" -ForegroundColor Cyan
Write-Host "  isaaclab.bat -p scripts/rcg/quick_lift_eval.py ``" -ForegroundColor Gray
Write-Host "      --run_glob `"logs/rsl_rl/franka_lift_baseline/*`" `"logs/rsl_rl/franka_lift_rcg/*`" ``" -ForegroundColor Gray
Write-Host "      --checkpoint_name model_$($MaxIterations - 1).pt --output lift_eval.csv --headless" -ForegroundColor Gray
Write-Host "  python scripts/rcg/permutation_test.py --csv lift_eval.csv --metrics success min_dist" -ForegroundColor Gray
