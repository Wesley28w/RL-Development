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
    [ValidateRange(1, 1000000)]
    [int] $Episodes = 10000,
    [ValidateRange(1, 65536)]
    [int] $NumEnvs = 512,
    [int] $EvalSeed = 12345,
    [switch] $DeterministicFactory,
    [switch] $AllowMissing,
    [switch] $Overwrite,
    [switch] $DryRun,
    [string] $JournalRoot,
    [string] $PythonExe
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$rslEvaluator = Join-Path $PSScriptRoot "evaluate_rsl_rl.py"
$factoryEvaluator = Join-Path $PSScriptRoot "evaluate_factory_rl_games.py"
$ablationNames = @(
    "Uniform", "AbsoluteDifficulty", "Fixed5050Blend", "ArgmaxOnly", "SoftmaxOnly", "NoEMA",
    "SamplingRatio05", "Delay15", "BufferSize1", "VelocityWorld", "SimplifiedSubtasks"
)

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
        if (-not (Test-IsaacLabPython $candidate)) { throw "Invalid Isaac Lab Python: $candidate" }
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
    return [pscustomobject]@{
        TaskSlug = $Task.ToLowerInvariant()
        MethodSlug = $methodSlug
        TaskId = $taskId
        Framework = if ($Task -eq "Factory") { "rl_games" } else { "rsl_rl" }
        Metric = switch ($Task) {
            "Cabinet" { "success_rate_margin" }
            "Lift" { "lift_success_rate" }
            "Factory" { "dones_success_rate" }
        }
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
$trainingRoot = Join-Path $journalRootPath "training"
$evaluationRoot = Join-Path $journalRootPath "evaluations"
if (-not (Test-Path -LiteralPath $trainingRoot -PathType Container) -and -not $DryRun) {
    throw "Training campaign root does not exist: $trainingRoot"
}

$jobs = @()
$missing = @()
foreach ($task in @("Cabinet", "Lift", "Factory")) {
    if ($task -notin $Tasks) { continue }
    foreach ($method in $Methods) {
        $spec = Get-MethodSpec $task $method
        $groupDir = Join-Path $trainingRoot "$($spec.TaskSlug)\$($spec.MethodSlug)"
        foreach ($seed in $Seeds) {
            try {
                if ($spec.Framework -eq "rsl_rl") {
                    $runs = @(Get-ChildItem -LiteralPath $groupDir -Directory -ErrorAction SilentlyContinue |
                        Where-Object { $_.Name -match "_seed_$seed$" })
                    if ($runs.Count -ne 1) {
                        throw "Expected one run ending in _seed_$seed, found $($runs.Count) under $groupDir."
                    }
                    $checkpoint = Join-Path $runs[0].FullName "model_1499.pt"
                } else {
                    $runDir = Join-Path $groupDir "seed_$seed"
                    $checkpoints = @(Get-ChildItem -LiteralPath (Join-Path $runDir "nn") -File `
                        -Filter "last_Factory_ep_200*.pth" -ErrorAction SilentlyContinue |
                        Where-Object { $_.Name -match '^last_Factory_ep_200(?:_rew_.*)?\.pth$' })
                    if ($checkpoints.Count -ne 1) {
                        throw "Expected one Factory epoch-200 checkpoint, found $($checkpoints.Count) under $runDir\nn."
                    }
                    $checkpoint = $checkpoints[0].FullName
                }
                if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
                    throw "Missing final checkpoint: $checkpoint"
                }
                $jobs += [pscustomobject]@{
                    Task = $task
                    Method = $method
                    Seed = $seed
                    Spec = $spec
                    Checkpoint = $checkpoint
                    OutputDir = Join-Path $evaluationRoot "$($spec.TaskSlug)\$($spec.MethodSlug)\eval_seed_$EvalSeed"
                }
            } catch {
                $missing += "$task/$method/seed_$seed`: $($_.Exception.Message)"
            }
        }
    }
}
if ($missing.Count -gt 0) {
    $message = "Checkpoint preflight found missing or ambiguous runs:`n  " + ($missing -join "`n  ")
    if (-not $AllowMissing) { throw $message + "`nUse -AllowMissing to evaluate only completed runs." }
    Write-Warning $message
}
if ($jobs.Count -eq 0) { throw "No evaluable checkpoints were found." }

Write-Host "[PYTHON]   $pythonExecutable"
Write-Host "[PROTOCOL] $Episodes completed episodes per checkpoint; $NumEnvs parallel environments"
Write-Host "[OUTPUT]   $evaluationRoot"
foreach ($job in $jobs) {
    if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $job.OutputDir | Out-Null }
    $summary = Join-Path $job.OutputDir "seed_$($job.Seed)_summary.json"
    if ((Test-Path -LiteralPath $summary) -and -not $Overwrite) {
        $existing = Get-Content -LiteralPath $summary -Raw | ConvertFrom-Json
        $expectedDeterministic = if ($job.Task -eq "Factory") { [bool]$DeterministicFactory } else { $true }
        $savedCheckpoint = if ($null -ne $existing.checkpoint) {
            [IO.Path]::GetFullPath([string]$existing.checkpoint)
        } else { "" }
        if ($existing.schema_version -ne 2 -or
            $null -eq $existing.PSObject.Properties[$job.Spec.Metric] -or
            $existing.requested_episodes -ne $Episodes -or
            $existing.num_envs -ne $NumEnvs -or
            $existing.evaluation_seed -ne $EvalSeed -or
            $existing.deterministic_policy -ne $expectedDeterministic -or
            $existing.reset_state_curriculum_enabled -ne $false -or
            $savedCheckpoint -ne [IO.Path]::GetFullPath($job.Checkpoint)) {
            throw "Existing evaluation uses a different checkpoint or protocol: $summary"
        }
        Write-Host "[SKIP] $($job.Task)/$($job.Method) seed $($job.Seed) already evaluated"
        continue
    }

    $evaluator = if ($job.Task -eq "Factory") { $factoryEvaluator } else { $rslEvaluator }
    $arguments = @(
        $evaluator, "--task", $job.Spec.TaskId, "--checkpoint", $job.Checkpoint,
        "--train_seed", $job.Seed, "--seed", $EvalSeed, "--num_envs", $NumEnvs,
        "--episodes", $Episodes, "--output_dir", $job.OutputDir, "--headless"
    )
    if ($Overwrite) { $arguments += "--overwrite" }
    if ($job.Task -eq "Factory" -and $DeterministicFactory) { $arguments += "--deterministic" }
    Write-Host "[EVAL] $($job.Task)/$($job.Method) seed $($job.Seed)"
    Write-Host "       checkpoint: $($job.Checkpoint)"
    if ($DryRun) {
        Write-Host "       $pythonExecutable $($arguments -join ' ')"
        continue
    }
    & $pythonExecutable @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed for $($job.Task)/$($job.Method) seed $($job.Seed) (exit code $LASTEXITCODE)."
    }
}

if (-not $DryRun) {
    foreach ($task in $Tasks) {
        foreach ($method in $Methods) {
            if ($task -ne "Cabinet" -and $method -in $ablationNames) { continue }
            $spec = Get-MethodSpec $task $method
            $outputDir = Join-Path $evaluationRoot "$($spec.TaskSlug)\$($spec.MethodSlug)\eval_seed_$EvalSeed"
            $summaries = @(Get-ChildItem -LiteralPath $outputDir -Filter "seed_*_summary.json" -File `
                -ErrorAction SilentlyContinue | ForEach-Object {
                    Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
                } | Sort-Object train_seed)
            if ($summaries.Count -gt 0) {
                $summaries | Select-Object -Property * -ExcludeProperty metric_definitions |
                    Export-Csv -LiteralPath (Join-Path $outputDir "all_seeds_summary.csv") -NoTypeInformation
            }
        }
    }
    Write-Host "[DONE] Evaluation campaign completed under $evaluationRoot"
} else {
    Write-Host "[DONE] Evaluation dry-run completed; no simulation was launched."
}
