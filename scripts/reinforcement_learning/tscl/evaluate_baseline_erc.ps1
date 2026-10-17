[CmdletBinding()]
param(
    [ValidateSet("Baseline", "ERC")]
    [string[]] $Methods = @("Baseline", "ERC"),
    [ValidateSet("Lift", "Cabinet", "Factory")]
    [string[]] $Tasks = @("Lift", "Cabinet", "Factory"),
    [int[]] $Seeds = (42..53),
    [ValidateRange(1, 1000000)]
    [int] $Episodes = 10000,
    [ValidateRange(1, 65536)]
    [int] $NumEnvs = 512,
    [int] $EvalSeed = 12345,
    [switch] $DeterministicFactory,
    [switch] $AllowProtocolOverride,
    [switch] $AllowMissing,
    [switch] $Overwrite,
    [switch] $DryRun,
    [string] $JournalRoot,
    [string] $ResultsRoot,
    [string] $PythonExe
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$rslEvaluator = Join-Path $PSScriptRoot "evaluate_rsl_rl.py"
$factoryEvaluator = Join-Path $PSScriptRoot "evaluate_factory_rl_games.py"

if (-not $AllowProtocolOverride) {
    $protocolDifferences = @()
    if ($Episodes -ne 10000) { $protocolDifferences += "Episodes=$Episodes (TSCL uses 10000)" }
    if ($NumEnvs -ne 512) { $protocolDifferences += "NumEnvs=$NumEnvs (TSCL used 512)" }
    if ($EvalSeed -ne 12345) { $protocolDifferences += "EvalSeed=$EvalSeed (TSCL used 12345)" }
    if ($DeterministicFactory) { $protocolDifferences += "DeterministicFactory=True (TSCL used its saved stochastic player setting)" }
    if ($protocolDifferences.Count -gt 0) {
        throw "Requested settings do not match the completed TSCL evaluations:`n  $($protocolDifferences -join "`n  ")`nUse -AllowProtocolOverride only for an intentionally separate experiment."
    }
}

function Test-IsaacLabPython {
    param([string] $Candidate)
    if ([string]::IsNullOrWhiteSpace($Candidate) -or -not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
        return $false
    }
    $requiredModules = @("isaaclab", "isaaclab_tasks", "isaaclab_rl", "isaacsim")
    if ($Tasks -contains "Factory") { $requiredModules += "rl_games" }
    if ($Tasks -contains "Lift" -or $Tasks -contains "Cabinet") { $requiredModules += "rsl_rl" }
    $moduleList = $requiredModules -join ","
    & $Candidate -c "import importlib.util,sys; missing=[m for m in '$moduleList'.split(',') if importlib.util.find_spec(m) is None]; sys.exit(1 if missing else 0)" 2>$null
    return $LASTEXITCODE -eq 0
}

function Resolve-IsaacLabPython {
    if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
        $explicitPath = if ([IO.Path]::IsPathRooted($PythonExe)) {
            [IO.Path]::GetFullPath($PythonExe)
        } else {
            [IO.Path]::GetFullPath((Join-Path $repoRoot $PythonExe))
        }
        if (-not (Test-IsaacLabPython $explicitPath)) {
            throw "The requested Python cannot import the Isaac Lab and selected RL-framework modules: $explicitPath"
        }
        return $explicitPath
    }

    $candidates = [System.Collections.Generic.List[string]]::new()
    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
        $candidates.Add((Join-Path $env:CONDA_PREFIX "python.exe"))
    }
    $command = Get-Command python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $command) { $candidates.Add($command.Source) }

    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if ($null -ne $conda) {
        try {
            $condaInfo = (& conda env list --json 2>$null | Out-String) | ConvertFrom-Json
            $namedEnvs = @($condaInfo.envs | Where-Object { (Split-Path $_ -Leaf) -eq "env_isaaclab" })
            foreach ($envPath in $namedEnvs) { $candidates.Add((Join-Path $envPath "python.exe")) }
        } catch {
            Write-Verbose "Could not query Conda environments: $($_.Exception.Message)"
        }
    }

    foreach ($candidate in @($candidates | Select-Object -Unique)) {
        if (Test-IsaacLabPython $candidate) { return [IO.Path]::GetFullPath($candidate) }
    }
    throw "No Python interpreter with Isaac Lab and the selected RL framework was found. Activate env_isaaclab or pass -PythonExe <path-to-python.exe>."
}

function Resolve-RslCheckpoint {
    param(
        [string] $JournalGroup,
        [int] $Seed,
        [string] $FinalCheckpoint
    )
    $root = Join-Path $journalRootPath $JournalGroup
    $matchingRuns = @()
    foreach ($run in @(Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue)) {
        $agentConfig = Join-Path $run.FullName "params\agent.yaml"
        if (-not (Test-Path -LiteralPath $agentConfig -PathType Leaf)) { continue }
        $configText = Get-Content -LiteralPath $agentConfig -Raw
        if ($configText -match "(?m)^seed:\s*$Seed\s*$") { $matchingRuns += $run }
    }
    if ($matchingRuns.Count -ne 1) {
        throw "Expected one $JournalGroup run whose saved agent config has seed $Seed, found $($matchingRuns.Count) under $root."
    }
    $checkpoint = Join-Path $matchingRuns[0].FullName $FinalCheckpoint
    if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
        throw "Missing fixed-budget checkpoint: $checkpoint"
    }
    return $checkpoint
}

function Resolve-FactoryCheckpoint {
    param(
        [string] $JournalGroup,
        [string] $RunPrefix,
        [int] $Seed
    )
    $runDir = Join-Path $journalRootPath "$JournalGroup\$RunPrefix$Seed"
    if (-not (Test-Path -LiteralPath $runDir -PathType Container)) {
        throw "Missing Factory journal run: $runDir"
    }

    # Prefer a checkpoint colocated with the journal run if the artifacts are reorganized later.
    $local = @(Get-ChildItem -LiteralPath $runDir -Recurse -File -Filter "last_Factory_ep_200_rew_*.pth" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^last_Factory_ep_200_rew_(?!_).+\.pth$' })
    if ($local.Count -eq 1) { return $local[0].FullName }
    if ($local.Count -gt 1) {
        throw "Ambiguous colocated Factory epoch-200 checkpoints for $JournalGroup seed $Seed under $runDir."
    }

    # The historical Baseline/ERC Factory jobs used the shared RL-Games experiment name `test`.
    # Associate the one canonical epoch-200 checkpoint written during this run's TensorBoard time window.
    $events = @(Get-ChildItem -LiteralPath $runDir -File -Filter "events.out.tfevents.*")
    if ($events.Count -ne 1 -or $events[0].Name -notmatch '^events\.out\.tfevents\.(\d+)\.') {
        throw "Expected one timestamped TensorBoard event file for $JournalGroup seed $Seed under $runDir."
    }
    $runStart = [DateTimeOffset]::FromUnixTimeSeconds([int64] $matches[1]).LocalDateTime
    $runEnd = $events[0].LastWriteTime.AddMinutes(30)
    $sharedNnDir = Join-Path $repoRoot "logs\rl_games\Factory\test\nn"
    $matchesForRun = @(Get-ChildItem -LiteralPath $sharedNnDir -File -Filter "last_Factory_ep_200_rew_*.pth" -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -match '^last_Factory_ep_200_rew_(?!_).+\.pth$' -and
            $_.LastWriteTime -ge $runStart -and
            $_.LastWriteTime -le $runEnd
        })
    if ($matchesForRun.Count -ne 1) {
        throw "Expected one canonical Factory epoch-200 checkpoint for $JournalGroup seed $Seed in its saved run window [$runStart, $runEnd], found $($matchesForRun.Count) under $sharedNnDir."
    }
    return $matchesForRun[0].FullName
}

$pythonExecutable = Resolve-IsaacLabPython
Write-Host "[INFO] Evaluation Python: $pythonExecutable"

if ([string]::IsNullOrWhiteSpace($JournalRoot)) {
    $journalRootPath = Join-Path $repoRoot "logs\journal"
} elseif ([IO.Path]::IsPathRooted($JournalRoot)) {
    $journalRootPath = [IO.Path]::GetFullPath($JournalRoot)
} else {
    $journalRootPath = [IO.Path]::GetFullPath((Join-Path $repoRoot $JournalRoot))
}
if (-not (Test-Path -LiteralPath $journalRootPath -PathType Container)) {
    throw "Journal root does not exist: $journalRootPath"
}

if ([string]::IsNullOrWhiteSpace($ResultsRoot)) {
    $resultsRootPath = Join-Path $journalRootPath "evaluations"
} elseif ([IO.Path]::IsPathRooted($ResultsRoot)) {
    $resultsRootPath = [IO.Path]::GetFullPath($ResultsRoot)
} else {
    $resultsRootPath = [IO.Path]::GetFullPath((Join-Path $repoRoot $ResultsRoot))
}

$jobs = @()
$missing = @()
foreach ($methodName in $Methods) {
    $methodSlug = $methodName.ToLowerInvariant()
    foreach ($taskName in $Tasks) {
        foreach ($seed in $Seeds) {
            try {
                switch ("$methodName/$taskName") {
                    "Baseline/Lift" {
                        $checkpoint = Resolve-RslCheckpoint "training\lift\baseline" $seed "model_1499.pt"
                        $taskId = "Isaac-Lift-Cube-Franka-v0"
                        $evaluator = $rslEvaluator
                        $slug = "franka_lift"
                    }
                    "Baseline/Cabinet" {
                        $checkpoint = Resolve-RslCheckpoint "training\cabinet\baseline" $seed "model_1499.pt"
                        $taskId = "Isaac-Franka-Cabinet-Direct-v0"
                        $evaluator = $rslEvaluator
                        $slug = "franka_cabinet"
                    }
                    "Baseline/Factory" {
                        $checkpoint = Resolve-FactoryCheckpoint "training\factory\baseline" "seed_" $seed
                        $taskId = "Isaac-Factory-NutThread-Direct-v0"
                        $evaluator = $factoryEvaluator
                        $slug = "franka_factory_nut_thread"
                    }
                    "ERC/Lift" {
                        $checkpoint = Resolve-RslCheckpoint "training\lift\erc" $seed "model_1499.pt"
                        $taskId = "Isaac-Lift-Cube-Franka-v0"
                        $evaluator = $rslEvaluator
                        $slug = "franka_lift"
                    }
                    "ERC/Cabinet" {
                        $checkpoint = Resolve-RslCheckpoint "training\cabinet\erc" $seed "model_1499.pt"
                        $taskId = "Isaac-Franka-Cabinet-Direct-v0"
                        $evaluator = $rslEvaluator
                        $slug = "franka_cabinet"
                    }
                    "ERC/Factory" {
                        $checkpoint = Resolve-FactoryCheckpoint "training\factory\erc" "seed_" $seed
                        $taskId = "Isaac-Factory-NutThread-Direct-v0"
                        $evaluator = $factoryEvaluator
                        $slug = "franka_factory_nut_thread"
                    }
                }
                $outputDir = Join-Path $resultsRootPath "$methodSlug\$slug\eval_seed_$EvalSeed"
                $jobs += [pscustomobject]@{
                    Method = $methodName
                    Task = $taskName
                    TaskId = $taskId
                    Seed = $seed
                    Checkpoint = $checkpoint
                    Evaluator = $evaluator
                    OutputDir = $outputDir
                }
            } catch {
                $missing += "${methodName}/${taskName}/seed_${seed}: $($_.Exception.Message)"
            }
        }
    }
}

if ($missing.Count -gt 0) {
    $message = "Checkpoint preflight found missing or ambiguous runs:`n  " + ($missing -join "`n  ")
    if (-not $AllowMissing) {
        throw $message + "`nUse -AllowMissing to evaluate only unambiguous completed runs."
    }
    Write-Warning $message
}
if ($jobs.Count -eq 0) { throw "No evaluable Baseline/ERC checkpoints were found." }

foreach ($job in $jobs) {
    if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $job.OutputDir | Out-Null }
    $summary = Join-Path $job.OutputDir "seed_$($job.Seed)_summary.json"
    if ((Test-Path -LiteralPath $summary) -and -not $Overwrite) {
        $existing = Get-Content -LiteralPath $summary -Raw | ConvertFrom-Json
        $expectedDeterministic = if ($job.Task -eq "Factory") { [bool] $DeterministicFactory } else { $true }
        $primaryMetric = switch ($job.Task) {
            "Lift" { "lift_success_rate" }
            "Cabinet" { "success_rate_margin" }
            "Factory" { "dones_success_rate" }
        }
        $existingCheckpoint = if ($null -ne $existing.checkpoint) {
            [IO.Path]::GetFullPath([string] $existing.checkpoint)
        } else { "" }
        if ($existing.schema_version -ne 2 -or
            $null -eq $existing.PSObject.Properties[$primaryMetric] -or
            $existing.task -ne $job.TaskId -or
            $existing.train_seed -ne $job.Seed -or
            $existing.requested_episodes -ne $Episodes -or
            $existing.num_envs -ne $NumEnvs -or
            $existing.evaluation_seed -ne $EvalSeed -or
            $existing.deterministic_policy -ne $expectedDeterministic -or
            $existing.reset_state_curriculum_enabled -ne $false -or
            $existingCheckpoint -ne [IO.Path]::GetFullPath($job.Checkpoint)) {
            throw "Existing result has different checkpoint or evaluation settings; inspect it or use -Overwrite: $summary"
        }
        Write-Host "[SKIP] $($job.Method) $($job.Task) seed $($job.Seed): matching result already exists."
        continue
    }

    $arguments = @(
        $job.Evaluator,
        "--task", $job.TaskId,
        "--checkpoint", $job.Checkpoint,
        "--train_seed", $job.Seed,
        "--seed", $EvalSeed,
        "--num_envs", $NumEnvs,
        "--episodes", $Episodes,
        "--output_dir", $job.OutputDir,
        "--headless"
    )
    if ($Overwrite) { $arguments += "--overwrite" }
    if ($job.Task -eq "Factory" -and $DeterministicFactory) { $arguments += "--deterministic" }

    Write-Host "[EVAL] $($job.Method) $($job.Task) seed $($job.Seed)"
    Write-Host "       checkpoint: $($job.Checkpoint)"
    Write-Host "       output:     $($job.OutputDir)"
    if ($DryRun) {
        Write-Host "       command:    $pythonExecutable $($arguments -join ' ')"
        continue
    }

    $previousWriteTime = if (Test-Path -LiteralPath $summary) {
        (Get-Item -LiteralPath $summary).LastWriteTimeUtc
    } else { [datetime]::MinValue }
    & $pythonExecutable @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed for $($job.Method) $($job.Task) seed $($job.Seed) (exit code $LASTEXITCODE)."
    }
    if (-not (Test-Path -LiteralPath $summary) -or
        (Get-Item -LiteralPath $summary).LastWriteTimeUtc -le $previousWriteTime) {
        throw "Evaluation did not produce a new summary for $($job.Method) $($job.Task) seed $($job.Seed): $summary"
    }
}

if (-not $DryRun) {
    foreach ($methodName in $Methods) {
        $methodSlug = $methodName.ToLowerInvariant()
        foreach ($taskName in $Tasks) {
            $slug = switch ($taskName) {
                "Lift" { "franka_lift" }
                "Cabinet" { "franka_cabinet" }
                "Factory" { "franka_factory_nut_thread" }
            }
            $outputDir = Join-Path $resultsRootPath "$methodSlug\$slug\eval_seed_$EvalSeed"
            $summaries = @(Get-ChildItem -LiteralPath $outputDir -Filter "seed_*_summary.json" -File -ErrorAction SilentlyContinue |
                ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json } |
                Sort-Object train_seed)
            if ($summaries.Count -eq 0) { continue }

            $settingSignatures = @($summaries | ForEach-Object {
                "$($_.schema_version)|$($_.task)|$($_.evaluation_seed)|$($_.requested_episodes)|$($_.num_envs)|$($_.deterministic_policy)|$($_.reset_state_curriculum_enabled)"
            } | Sort-Object -Unique)
            if ($settingSignatures.Count -ne 1) {
                throw "Refusing to aggregate mixed evaluation settings in $outputDir."
            }
            $aggregate = Join-Path $outputDir "all_seeds_summary.csv"
            $summaries | Select-Object -Property * -ExcludeProperty metric_definitions |
                Export-Csv -LiteralPath $aggregate -NoTypeInformation
            Write-Host "[DONE] Wrote $aggregate"
        }
    }
}
