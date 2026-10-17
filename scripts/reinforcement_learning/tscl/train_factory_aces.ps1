[CmdletBinding()]
param(
    [int[]] $Seeds = (42..57),
    [string] $PythonExe = "",
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
$trainScript = Join-Path $repoRoot "scripts\reinforcement_learning\rl_games\train.py"
$factoryCfg = Join-Path $repoRoot "source\isaaclab_tasks\isaaclab_tasks\direct\factory\factory_env_cfg.py"
$factoryEnv = Join-Path $repoRoot "source\isaaclab_tasks\isaaclab_tasks\direct\factory\factory_env.py"
$outputRoot = Join-Path $repoRoot "logs\rl_games\Factory"

foreach ($requiredPath in @($trainScript, $factoryCfg, $factoryEnv)) {
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

# Calling this script through `powershell -File` can run the user's PowerShell profile in the child
# process and reactivate Conda base. Resolve an interpreter that can actually import IsaacLab instead
# of relying on whichever environment isaaclab.bat happens to observe.
$pythonCandidates = @()
if ($PythonExe) {
    $pythonCandidates += $PythonExe
}
else {
    if ($env:CONDA_PREFIX) {
        $pythonCandidates += (Join-Path $env:CONDA_PREFIX "python.exe")
    }
    if ($env:CONDA_EXE) {
        $condaRoot = Split-Path (Split-Path $env:CONDA_EXE -Parent) -Parent
        $pythonCandidates += (Join-Path $condaRoot "envs\env_isaaclab\python.exe")
    }
    if ($env:USERPROFILE) {
        $pythonCandidates += (Join-Path $env:USERPROFILE "miniconda3\envs\env_isaaclab\python.exe")
        $pythonCandidates += (Join-Path $env:USERPROFILE "anaconda3\envs\env_isaaclab\python.exe")
    }
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $pythonCandidates += $pythonCommand.Source
    }
}

$pythonPath = $null
foreach ($candidate in @($pythonCandidates | Select-Object -Unique)) {
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        continue
    }
    & $candidate -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('isaaclab') and importlib.util.find_spec('rl_games') else 1)"
    if ($LASTEXITCODE -eq 0) {
        $pythonPath = (Resolve-Path -LiteralPath $candidate).Path
        break
    }
}
if (-not $pythonPath) {
    throw "Could not find a Python interpreter with both isaaclab and rl_games. Pass -PythonExe with the env_isaaclab python.exe path."
}

Write-Host "[PROTOCOL] Factory Nut-Thread ACES"
Write-Host "[PROTOCOL] Seeds: 42-57 (16 runs)"
Write-Host "[PROTOCOL] Environments per run: $numEnvs"
Write-Host "[PROTOCOL] RL-Games epochs per run: $maxIterations"
Write-Host "[PROTOCOL] Revised normalized-gap softmax temperature: $softmaxTemperature"
Write-Host "[PYTHON]   $pythonPath"
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
            Write-Host ("& `"{0}`" {1}" -f $pythonPath, ($arguments -join " "))
            continue
        }

        & $pythonPath @arguments
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
