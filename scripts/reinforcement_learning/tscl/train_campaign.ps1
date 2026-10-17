[CmdletBinding()]
param(
    [ValidateSet("Cabinet", "Lift", "Factory")]
    [string[]] $Tasks = @("Cabinet", "Lift", "Factory"),
    [ValidateSet(
        "Baseline", "ERC", "TSCL", "Uniform", "AbsoluteDifficulty", "Fixed5050Blend",
        "ArgmaxOnly", "SoftmaxOnly", "NoEMA", "SamplingRatio05", "Delay15",
        "BufferSize1", "VelocityWorld", "SimplifiedSubtasks"
    )]
    [string[]] $Methods = @("Baseline", "ERC", "TSCL"),
    [int[]] $Seeds = (42..53),
    [int] $CabinetNumEnvs = 4096,
    [int] $CabinetIterations = 1500,
    [int] $LiftNumEnvs = 4096,
    [int] $LiftIterations = 1500,
    [int] $FactoryNumEnvs = 128,
    [int] $FactoryIterations = 200,
    [string] $MethodDirectorySuffix = "",
    [switch] $AppendBudgetToMethodDirectory,
    [switch] $IncludeCabinetAblations,
    [switch] $DryRun,
    [string] $JournalRoot,
    [string] $PythonExe
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$rslTrainer = Join-Path $repoRoot "scripts\reinforcement_learning\rsl_rl\train.py"
$factoryTrainer = Join-Path $repoRoot "scripts\reinforcement_learning\rl_games\train.py"
$ablationNames = @(
    "Uniform", "AbsoluteDifficulty", "Fixed5050Blend", "ArgmaxOnly", "SoftmaxOnly", "NoEMA",
    "SamplingRatio05", "Delay15", "BufferSize1", "VelocityWorld", "SimplifiedSubtasks"
)

if ($CabinetNumEnvs -le 0) { throw "CabinetNumEnvs must be positive." }
if ($CabinetIterations -le 0) { throw "CabinetIterations must be positive." }
if ($LiftNumEnvs -le 0) { throw "LiftNumEnvs must be positive." }
if ($LiftIterations -le 0) { throw "LiftIterations must be positive." }
if ($FactoryNumEnvs -le 0) { throw "FactoryNumEnvs must be positive." }
if ($FactoryIterations -le 0) { throw "FactoryIterations must be positive." }
if ($MethodDirectorySuffix -match '[\\/]') {
    throw "MethodDirectorySuffix must not contain path separators."
}

function Test-IsaacLabPython {
    param([string] $Candidate)
    if ([string]::IsNullOrWhiteSpace($Candidate) -or -not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
        return $false
    }
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Candidate -c "import importlib.util,sys; mods=['isaaclab','isaaclab_tasks','isaaclab_rl','isaacsim','rsl_rl','rl_games']; sys.exit(0 if all(importlib.util.find_spec(m) is not None for m in mods) else 1)" 2>$null
    $importExitCode = $LASTEXITCODE
    $ErrorActionPreference = $oldErrorActionPreference
    return $importExitCode -eq 0
}

function Resolve-IsaacLabPython {
    if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
        $candidate = if ([IO.Path]::IsPathRooted($PythonExe)) {
            [IO.Path]::GetFullPath($PythonExe)
        } else {
            [IO.Path]::GetFullPath((Join-Path $repoRoot $PythonExe))
        }
        if (-not (Test-IsaacLabPython $candidate)) {
            throw "The requested Python cannot import Isaac Lab, RSL-RL, and RL-Games: $candidate"
        }
        return $candidate
    }
    $candidates = [System.Collections.Generic.List[string]]::new()
    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
        $candidates.Add((Join-Path $env:CONDA_PREFIX "python.exe"))
    }
    $candidates.Add("C:\Users\wesle\miniconda3\envs\env_isaaclab\python.exe")
    $python = Get-Command python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $python) { $candidates.Add($python.Source) }
    foreach ($candidate in @($candidates | Select-Object -Unique)) {
        if (Test-IsaacLabPython $candidate) { return [IO.Path]::GetFullPath($candidate) }
    }
    throw "No compatible Isaac Lab Python was found. Activate env_isaaclab or pass -PythonExe."
}

function Get-MethodSpec {
    param([string] $Task, [string] $Method)
    if ($Task -ne "Cabinet" -and $Method -in $ablationNames) {
        throw "$Method is a Cabinet-only ablation."
    }

    $methodSlug = switch ($Method) {
        "Baseline" { "baseline" }
        "ERC" { "erc" }
        "TSCL" { "tscl" }
        "Uniform" { "uniform" }
        "AbsoluteDifficulty" { "absolute_difficulty" }
        "Fixed5050Blend" { "fixed_50_50_blend" }
        "ArgmaxOnly" { "argmax_only" }
        "SoftmaxOnly" { "softmax_only" }
        "NoEMA" { "no_ema" }
        "SamplingRatio05" { "sampling_ratio_0_5" }
        "Delay15" { "delay_15" }
        "BufferSize1" { "buffer_size_1" }
        "VelocityWorld" { "velocity_world" }
        "SimplifiedSubtasks" { "simplified_subtasks" }
    }
    $mode = switch ($Method) {
        "Baseline" { "aces" }
        "ERC" { "aces" }
        "TSCL" { "tscl_window" }
        "Uniform" { "uniform_random" }
        "AbsoluteDifficulty" { "absolute_difficulty" }
        "Fixed5050Blend" { "fixed_50_50_blend" }
        "ArgmaxOnly" { "argmax_only" }
        "SoftmaxOnly" { "softmax_only" }
        "NoEMA" { "no_ema" }
        default { "aces" }
    }
    $enabled = $Method -ne "Baseline"
    $taskId = switch ($Task) {
        "Cabinet" {
            switch ($Method) {
                "Uniform" { "Isaac-Franka-Cabinet-Uniform-Direct-v0" }
                "AbsoluteDifficulty" { "Isaac-Franka-Cabinet-Absolute-Difficulty-Direct-v0" }
                "Fixed5050Blend" { "Isaac-Franka-Cabinet-Fixed-50-50-Blend-Direct-v0" }
                "ArgmaxOnly" { "Isaac-Franka-Cabinet-Argmax-Only-Direct-v0" }
                "SoftmaxOnly" { "Isaac-Franka-Cabinet-Softmax-Only-Direct-v0" }
                "NoEMA" { "Isaac-Franka-Cabinet-No-EMA-Direct-v0" }
                "SamplingRatio05" { "Isaac-Franka-Cabinet-Sampling-Ratio-0-5-Direct-v0" }
                "Delay15" { "Isaac-Franka-Cabinet-Delay-15-Direct-v0" }
                "BufferSize1" { "Isaac-Franka-Cabinet-Buffer-Size-1-Direct-v0" }
                "VelocityWorld" { "Isaac-Franka-Cabinet-Velocity-World-Direct-v0" }
                "SimplifiedSubtasks" { "Isaac-Franka-Cabinet-Simplified-Subtasks-Direct-v0" }
                default { "Isaac-Franka-Cabinet-Direct-v0" }
            }
        }
        "Lift" { "Isaac-Lift-Cube-Franka-v0" }
        "Factory" { "Isaac-Factory-NutThread-Direct-v0" }
    }
    $taskSlug = $Task.ToLowerInvariant()
    return [pscustomobject]@{
        Task = $Task
        TaskSlug = $taskSlug
        Method = $Method
        MethodSlug = $methodSlug
        Mode = $mode
        Enabled = $enabled
        TaskId = $taskId
        NumEnvs = switch ($Task) {
            "Cabinet" { $CabinetNumEnvs }
            "Lift" { $LiftNumEnvs }
            "Factory" { $FactoryNumEnvs }
        }
        Iterations = switch ($Task) {
            "Cabinet" { $CabinetIterations }
            "Lift" { $LiftIterations }
            "Factory" { $FactoryIterations }
        }
        Framework = if ($Task -eq "Factory") { "rl_games" } else { "rsl_rl" }
    }
}

$pythonExecutable = Resolve-IsaacLabPython
$journalRootPath = if ([string]::IsNullOrWhiteSpace($JournalRoot)) {
    Join-Path $repoRoot "logs\journal"
} elseif ([IO.Path]::IsPathRooted($JournalRoot)) {
    [IO.Path]::GetFullPath($JournalRoot)
} else {
    [IO.Path]::GetFullPath((Join-Path $repoRoot $JournalRoot))
}
if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $journalRootPath | Out-Null }

$expectedJournalRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot "logs\journal"))
if ([IO.Path]::GetFullPath($journalRootPath) -ne $expectedJournalRoot) {
    Write-Warning "Using non-default journal root: $journalRootPath"
}

Write-Host "[PYTHON]  $pythonExecutable"
Write-Host "[OUTPUT]  $journalRootPath\training"
Write-Host "[ORDER]   $($Tasks -join ' -> ')"
Write-Host "[METHODS] $($Methods -join ', ')"
Write-Host "[SEEDS]   $($Seeds -join ', ')"
if (-not [string]::IsNullOrWhiteSpace($MethodDirectorySuffix)) {
    Write-Host "[SUFFIX]  $MethodDirectorySuffix"
}
if ($AppendBudgetToMethodDirectory) {
    Write-Host "[SUFFIX]  append per-task environment and iteration budgets"
}
if ($IncludeCabinetAblations) {
    Write-Host "[ABLATIONS] Cabinet: $($ablationNames -join ', ')"
}

Push-Location $repoRoot
try {
    foreach ($task in @("Cabinet", "Lift", "Factory")) {
        if ($task -notin $Tasks) { continue }
        $taskMethods = @($Methods)
        if ($task -eq "Cabinet" -and $IncludeCabinetAblations) {
            $taskMethods += $ablationNames
        }
        $taskMethods = @($taskMethods | Select-Object -Unique)

        foreach ($method in $taskMethods) {
            $spec = Get-MethodSpec $task $method
            $methodDirectory = "$($spec.MethodSlug)$MethodDirectorySuffix"
            if ($AppendBudgetToMethodDirectory) {
                $methodDirectory += "_$($spec.NumEnvs)env_$($spec.Iterations)iter"
            }
            $groupDir = Join-Path $journalRootPath "training\$($spec.TaskSlug)\$methodDirectory"
            if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $groupDir | Out-Null }

            foreach ($seed in $Seeds) {
                if ($spec.Framework -eq "rsl_rl") {
                    $runs = @(Get-ChildItem -LiteralPath $groupDir -Directory -ErrorAction SilentlyContinue |
                        Where-Object { $_.Name -match "_seed_$seed$" })
                    $finalCheckpoint = "model_$($spec.Iterations - 1).pt"
                    $complete = @($runs | Where-Object {
                        Test-Path -LiteralPath (Join-Path $_.FullName $finalCheckpoint) -PathType Leaf
                    })
                    if ($complete.Count -eq 1 -and $runs.Count -eq 1) {
                        Write-Host "[SKIP] $task/$method seed $seed already has $finalCheckpoint"
                        continue
                    }
                    if ($runs.Count -gt 0) {
                        throw "Existing incomplete or ambiguous run for $task/$method seed $seed under $groupDir. Preserve it and resolve it before relaunching."
                    }

                    $experimentPath = [IO.Path]::GetFullPath($groupDir).Replace('\', '/')
                    $arguments = @(
                        $rslTrainer, "--task", $spec.TaskId, "--headless",
                        "--num_envs", $spec.NumEnvs, "--max_iterations", $spec.Iterations,
                        "--seed", $seed, "--experiment_name", $experimentPath,
                        "--run_name", "seed_$seed",
                        "env.reset_state_curriculum_enabled=$($spec.Enabled.ToString().ToLowerInvariant())",
                        "env.reset_curriculum_mode=$($spec.Mode)"
                    )
                } else {
                    $runDir = Join-Path $groupDir "seed_$seed"
                    $factoryCheckpointPattern = "last_Factory_ep_$($spec.Iterations)*.pth"
                    $finalCheckpoints = @(Get-ChildItem -LiteralPath (Join-Path $runDir "nn") -File `
                        -Filter $factoryCheckpointPattern -ErrorAction SilentlyContinue)
                    if ($finalCheckpoints.Count -ge 1) {
                        Write-Host "[SKIP] $task/$method seed $seed already has an epoch-$($spec.Iterations) checkpoint"
                        continue
                    }
                    if (Test-Path -LiteralPath $runDir) {
                        throw "Existing incomplete Factory run for $method seed $seed at $runDir. Preserve it and resolve it before relaunching."
                    }
                    $experimentPath = [IO.Path]::GetFullPath($runDir).Replace('\', '/')
                    $arguments = @(
                        $factoryTrainer, "--task", $spec.TaskId, "--headless",
                        "--num_envs", $spec.NumEnvs, "--max_iterations", $spec.Iterations,
                        "--seed", $seed,
                        "env.reset_state_curriculum_enabled=$($spec.Enabled.ToString().ToLowerInvariant())",
                        "env.reset_curriculum_mode=$($spec.Mode)",
                        "agent.params.config.full_experiment_name=$experimentPath"
                    )
                }

                Write-Host "[TRAIN] $task/$method seed $seed -- $($spec.NumEnvs) envs, $($spec.Iterations) iterations"
                if ($DryRun) {
                    Write-Host "        $pythonExecutable $($arguments -join ' ')"
                    continue
                }
                & $pythonExecutable @arguments
                if ($LASTEXITCODE -ne 0) {
                    throw "Training failed for $task/$method seed $seed (exit code $LASTEXITCODE)."
                }
            }
        }
    }
} finally {
    Pop-Location
}

if ($DryRun) {
    Write-Host "[DONE] Dry-run completed; no training was launched."
} else {
    Write-Host "[DONE] Requested training campaign completed."
}
