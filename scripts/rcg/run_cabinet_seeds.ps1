# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Benchmark sweep: Reverse Curriculum Generation on Franka Cabinet.
#
# 16 seeds (42-57), 2500 iterations, 16384 environments -- matched to the journal_baseline and
# journal_curriculum sweeps so the three arms are directly comparable.
#
# Runs sequentially: each run needs the whole GPU at 16384 envs. Budget ~50 min per seed,
# so roughly 13-14 hours total. Safe to leave overnight; a failed seed is logged and skipped
# rather than aborting the sweep.
#
#   .\scripts\rcg\run_cabinet_seeds.ps1
#   .\scripts\rcg\run_cabinet_seeds.ps1 -Seeds 42,43 -MaxIterations 50    # quick shakedown

param(
    [int[]]  $Seeds          = @(42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57),
    [int]    $NumEnvs        = 16384,
    [int]    $MaxIterations  = 2500,
    [string] $RunPrefix      = "journal_rcg"
)

$ErrorActionPreference = "Continue"

# resolve the repo root from this script's location, so the sweep can be started from anywhere
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

$GoalStates = Join-Path $RepoRoot "source\isaaclab_tasks\isaaclab_tasks\direct\franka_cabinet\data\goal_states_franka_cabinet.pt"
if (-not (Test-Path $GoalStates)) {
    Write-Host "[FATAL] Goal states not found: $GoalStates" -ForegroundColor Red
    Write-Host "        Record them first with scripts/rcg/record_goal_states.py" -ForegroundColor Red
    exit 1
}

$LogRoot = Join-Path $RepoRoot "logs\rcg_sweep"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host " RCG Franka Cabinet sweep" -ForegroundColor Cyan
Write-Host "   seeds            : $($Seeds -join ', ')"
Write-Host "   envs / iters     : $NumEnvs / $MaxIterations"
Write-Host "   start state      : joint positions only (13 values, matching the reset-pose curriculum)"
Write-Host "   goal states      : $GoalStates"
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
    # terminate_on_success is deliberately left at the task default (upstream behaviour): the
    # benchmark measures the task as published, and all existing journal runs used it.
    & isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py `
        --task Isaac-Franka-Cabinet-RCG-Direct-v0 `
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
Write-Host "Next -- evaluate from rho_0 (curriculum OFF), which is the only number comparable to" -ForegroundColor Cyan
Write-Host "the baseline and reset-pose-curriculum arms:" -ForegroundColor Cyan
Write-Host "  isaaclab.bat -p scripts/rcg/evaluate.py --task Isaac-Franka-Cabinet-RCG-Direct-v0 ``" -ForegroundColor Gray
Write-Host "      --run_dir logs/rsl_rl/franka_cabinet_rcg/<run> --last --episodes 512 --headless" -ForegroundColor Gray
