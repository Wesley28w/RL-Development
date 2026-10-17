[CmdletBinding()]
param(
    [int[]] $Seeds = (42..57),
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"

# This launcher intentionally fixes the benchmark protocol. Change the source-controlled
# constants here only when defining a new experiment family.
$task = "Isaac-Factory-NutThread-Direct-v0"
$numEnvs = 512
$maxIterations = 200
$softmaxTemperature = 0.25
$runPrefix = "aces_nut_thread_512env_200iter_temp_0p25"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$isaacLabBat = Join-Path $repoRoot "isaaclab.bat"
$trainScript = Join-Path $repoRoot "scripts\reinforcement_learning\rl_games\train.py"
$factoryCfg = Join-Path $repoRoot "source\isaaclab_tasks\isaaclab_tasks\direct\factory\factory_env_cfg.py"
$factoryEnv = Join-Path $repoRoot "source\isaaclab_tasks\isaaclab_tasks\direct\factory\factory_env.py"
$outputRoot = Join-Path $repoRoot "logs\rl_games\Factory"

foreach ($requiredPath in @($isaacLabBat, $trainScript, $factoryCfg, $factoryEnv)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file is missing: $requiredPath"
    }
}

$expectedSeeds = @(42..57)
if ($Seeds.Count -ne $expectedSeeds.Count -or @(Compare-Object $Seeds $expectedSeeds).Count -ne 0) {
    throw "This benchmark launcher requires exactly seeds 42 through 57."
}

# Fail before launching expensive jobs if the checked-out implementation is not the revised ACES soft branch.
$cfgText = Get-Content -LiteralPath $factoryCfg -Raw
$envText = Get-Content -LiteralPath $factoryEnv -Raw
if ($cfgText -notmatch '(?m)^\s*softmax_temperature\s*=\s*0\.25\s*$') {
    throw "FactoryCfg no longer declares softmax_temperature = 0.25. Review the experiment before running."
}
if ($envText -notmatch 'soft\s*=\s*\(confidence\s*/\s*self\.cfg\.softmax_temperature\)\.softmax\(dim=0\)') {
    throw "The revised normalized-gap ACES softmax implementation was not found in factory_env.py."
}

Write-Host "[PROTOCOL] Factory Nut-Thread ACES"
Write-Host "[PROTOCOL] Seeds: 42-57 (16 runs)"
Write-Host "[PROTOCOL] Environments per run: $numEnvs"
Write-Host "[PROTOCOL] RL-Games epochs per run: $maxIterations"
Write-Host "[PROTOCOL] Revised normalized-gap softmax temperature: $softmaxTemperature"
Write-Host "[OUTPUT]   $outputRoot\$runPrefix`_seed_<seed>"

Push-Location $repoRoot
try {
    foreach ($seed in $Seeds) {
        $runName = "${runPrefix}_seed_${seed}"
        $runDir = Join-Path $outputRoot $runName

        if (Test-Path -LiteralPath $runDir) {
            throw "Refusing to overwrite existing run directory: $runDir"
        }

        $arguments = @(
            "-p",
            $trainScript,
            "--task", $task,
            "--headless",
            "--device", "cuda:0",
            "--num_envs", "$numEnvs",
            "--max_iterations", "$maxIterations",
            "--seed", "$seed",
            "env.reset_state_curriculum_enabled=true",
            "env.reset_curriculum_mode=aces",
            "env.softmax_temperature=$softmaxTemperature",
            "env.curriculum_total_iterations=$maxIterations",
            "agent.params.config.full_experiment_name=$runName"
        )

        Write-Host ""
        Write-Host "[RUN] Seed $seed -> $runName"
        if ($DryRun) {
            Write-Host ("& `"{0}`" {1}" -f $isaacLabBat, ($arguments -join " "))
            continue
        }

        & $isaacLabBat @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Factory ACES seed $seed failed with exit code $LASTEXITCODE."
        }

        $savedEnvCfg = Join-Path $runDir "params\env.yaml"
        $savedAgentCfg = Join-Path $runDir "params\agent.yaml"
        if (-not (Test-Path -LiteralPath $savedEnvCfg) -or -not (Test-Path -LiteralPath $savedAgentCfg)) {
            throw "Seed $seed finished without saved configuration files under $runDir."
        }

        $savedEnv = Get-Content -LiteralPath $savedEnvCfg -Raw
        $savedAgent = Get-Content -LiteralPath $savedAgentCfg -Raw
        $checks = @(
            @{ Name = "num_envs=512"; Passed = $savedEnv -match '(?m)^\s*num_envs:\s*512\s*$' },
            @{ Name = "ACES enabled"; Passed = $savedEnv -match '(?m)^reset_state_curriculum_enabled:\s*true\s*$' },
            @{ Name = "ACES scheduler"; Passed = $savedEnv -match '(?m)^reset_curriculum_mode:\s*aces\s*$' },
            @{ Name = "temperature=0.25"; Passed = $savedEnv -match '(?m)^softmax_temperature:\s*0\.25\s*$' },
            @{ Name = "seed=$seed"; Passed = $savedAgent -match "(?m)^\s*seed:\s*$seed\s*`$" },
            @{ Name = "max_epochs=200"; Passed = $savedAgent -match '(?m)^\s*max_epochs:\s*200\s*$' }
        )
        $failedChecks = @($checks | Where-Object { -not $_.Passed } | ForEach-Object { $_.Name })
        if ($failedChecks.Count -gt 0) {
            throw "Seed $seed saved unexpected settings: $($failedChecks -join ', ')"
        }
        Write-Host "[VERIFIED] Seed $seed saved the requested ACES protocol."
    }
}
finally {
    Pop-Location
}

if ($DryRun) {
    Write-Host ""
    Write-Host "[DONE] Dry run only; no training was launched."
}
else {
    Write-Host ""
    Write-Host "[DONE] All 16 Factory ACES runs completed and their saved configs were verified."
}
