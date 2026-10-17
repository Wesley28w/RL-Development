# TSCL fixed-checkpoint evaluation

## Sixteen-seed training campaign

The canonical all-task launcher uses seeds 42--57 and always writes below `logs/journal/training/`. `train_all.ps1`
runs Cabinet first (16,384 environments, 1,500 iterations), Lift second (4,096 environments, 1,500 iterations), and
Factory last (512 environments, 75 iterations). Cabinet includes Baseline, ERC, TSCL, and every supported ERC
ablation; Lift and Factory run the matched Baseline/ERC/TSCL trio because the ablation task registrations are
Cabinet-specific. Completed runs are skipped, while partial or ambiguous runs stop the launcher so they cannot be
silently mixed with a replacement. Output method directories include their environment and iteration budgets.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reinforcement_learning/tscl/train_all.ps1
powershell -ExecutionPolicy Bypass -File scripts/reinforcement_learning/tscl/train_cabinet.ps1
powershell -ExecutionPolicy Bypass -File scripts/reinforcement_learning/tscl/train_lift.ps1
powershell -ExecutionPolicy Bypass -File scripts/reinforcement_learning/tscl/train_factory.ps1
powershell -ExecutionPolicy Bypass -File scripts/reinforcement_learning/tscl/train_cabinet_ablations.ps1
```

Add `-DryRun` to inspect all resolved commands without launching Isaac Sim. Individual methods and ablations can be
selected with `-Methods` and `-Ablations`. Matching Bash wrappers with the same base names are also provided.

The matching evaluation launchers use 10,000 completed episodes per checkpoint, 512 evaluation environments, and
evaluation seed 12345. Results are written below `logs/journal/evaluations/<task>/<method>/`.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reinforcement_learning/tscl/evaluate_all.ps1
powershell -ExecutionPolicy Bypass -File scripts/reinforcement_learning/tscl/evaluate_cabinet.ps1
powershell -ExecutionPolicy Bypass -File scripts/reinforcement_learning/tscl/evaluate_lift.ps1
powershell -ExecutionPolicy Bypass -File scripts/reinforcement_learning/tscl/evaluate_factory.ps1
powershell -ExecutionPolicy Bypass -File scripts/reinforcement_learning/tscl/evaluate_cabinet_ablations.ps1
```

This directory evaluates the final, fixed-budget checkpoint from every TSCL training seed. Evaluation disables
reset-state curriculum sampling and uses each task's original reset distribution, reward, observation, termination,
and success definition. It does not select checkpoints using evaluation performance.

From the IsaacLab repository root with the Isaac Lab environment activated, evaluate all completed seeds for one task:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reinforcement_learning/tscl/evaluate_tscl.ps1 -Tasks Cabinet
```

Evaluate all three tasks and require every seed from 42 through 57 to be present:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reinforcement_learning/tscl/evaluate_tscl.ps1
```

If training is still in progress, add `-AllowMissing` to evaluate only checkpoints that have reached the exact final
budget. Add `-DryRun` to inspect every resolved checkpoint and command without launching Isaac Sim.

The launcher validates the Python environment before starting. It first checks the active interpreter, then an
installed Conda environment named `env_isaaclab`. To select one explicitly, pass
`-PythonExe C:\path\to\env_isaaclab\python.exe`. This avoids `isaaclab.bat` accidentally using base Conda when a
PowerShell prompt displays a stale environment name.

The launcher reads the organized journal layout directly:

```text
logs/journal/
  training/cabinet/tscl/<timestamp>_seed_<42-53>/model_1499.pt
  training/lift/tscl/<timestamp>_seed_<42-53>/model_1499.pt
  training/factory/tscl/seed_<42-53>/nn/last_Factory_ep_200_rew_*.pth
```

Defaults are 10,000 episodes per trained seed, 512 parallel environments, and the same evaluation seed (12345) for
every policy. Per-episode CSV files, per-seed JSON summaries, checkpoint hashes, Wilson 95% intervals, and a combined
CSV are written below `logs/journal/evaluations/tscl/`, separate from immutable training artifacts. Use `-JournalRoot`
or `-ResultsRoot` only when intentionally working from a different location; relative paths are resolved from the
repository root.

Headline metrics are the original task metrics: Cabinet `dones/success_rate_margin`, Lift `lift_success_rate`, and
Factory `dones/success_rate`. The files also include strict binary completion, return, length, timeout/termination
rates where applicable, policy-action magnitude, final and ever-achieved subtask success rates, and task-specific
physical diagnostics (drawer progress and gripper distance; cube/goal distances and height; nut XY/Z/yaw errors).
Every JSON summary contains human-readable metric definitions, while its episode CSV contains the underlying rows.

RSL-RL inference is deterministic. Factory preserves its saved RL-Games player setting (currently stochastic); pass
`-DeterministicFactory` only if that same inference setting will also be used for every compared Factory method.

## Matched Baseline and ERC evaluation

Use `evaluate_baseline_erc.ps1` to evaluate the final fixed-budget Baseline and ERC checkpoints with the same protocol
as TSCL: 10,000 episodes per trained seed, 512 parallel environments, evaluation seed 12345, deterministic RSL-RL
inference, and the saved stochastic RL-Games Factory player setting. Curriculum reset sampling is disabled, so every
method is measured from the original task reset distribution.

Preflight every seed and resolved checkpoint without launching Isaac Sim:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reinforcement_learning/tscl/evaluate_baseline_erc.ps1 -DryRun
```

Evaluate all seeds for all three tasks, first Baseline and then ERC:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reinforcement_learning/tscl/evaluate_baseline_erc.ps1
```

Methods and tasks can be run separately, which is usually easier for long evaluations:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reinforcement_learning/tscl/evaluate_baseline_erc.ps1 -Methods Baseline -Tasks Cabinet
powershell -ExecutionPolicy Bypass -File scripts/reinforcement_learning/tscl/evaluate_baseline_erc.ps1 -Methods ERC -Tasks Cabinet
```

Outputs are kept separate under `logs/journal/evaluations/baseline/` and `logs/journal/evaluations/erc/`. Existing
matching summaries are skipped, so an interrupted command can be run again to resume. A result with a different
checkpoint, task, episode count, environment count, evaluation seed, inference mode, or curriculum-reset state is
rejected rather than silently mixed into the aggregate. The launcher also rejects changes to the matched TSCL
protocol unless `-AllowProtocolOverride` is supplied for an intentionally separate experiment.

The historical Factory Baseline/ERC jobs stored their checkpoints in the shared `logs/rl_games/Factory/test/nn`
directory. The launcher maps them back to their per-seed journal runs using the TensorBoard run timestamps and
requires exactly one canonical epoch-200 checkpoint in each run window. It stops on any missing or ambiguous mapping.

## Unified 10,000-episode full benchmark

`evaluate_full_benchmark.ps1` reruns every final Baseline, ERC, and TSCL policy for Lift, Cabinet, and Factory at one
fixed protocol: seeds 42 through 53, 10,000 completed episodes per checkpoint, 512 parallel environments, and
evaluation seed 12345. Results are written only below `logs/journal/full_eval/`, organized by method and task.

Run the full checkpoint and command preflight without launching Isaac Sim:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reinforcement_learning/tscl/evaluate_full_benchmark.ps1 -DryRun
```

Run the complete benchmark sequentially:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reinforcement_learning/tscl/evaluate_full_benchmark.ps1
```

The launcher evaluates 108 final policies: three methods, three tasks, and twelve training seeds. Each policy gets
exactly 10,000 completed episodes. It is resumable: matching completed summaries are skipped on the next invocation.
Use `-Overwrite` only when deliberately replacing already completed results. `-Methods`, `-Tasks`, and `-Seeds` can
restrict a run without changing the evaluation protocol.

## Corrected Factory dones/success_rate evaluation

`evaluate_factory_dones.ps1` evaluates only Factory Nut-Thread for Baseline, ERC, and TSCL. By default it evaluates
seeds 42 through 53 for 2,000 completed episodes per checkpoint, using 512 environments, evaluation seed 12345, the
original reset distribution, curriculum-disabled resets, and the saved stochastic RL-Games player setting.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reinforcement_learning/tscl/evaluate_factory_dones.ps1
```

Results replace the earlier Factory-only artifacts in the existing hierarchy under
`logs/journal/full_eval/{baseline,erc,tscl}/franka_factory_nut_thread/`. Lift and Cabinet results are not touched. For
every checkpoint the evaluator writes:

- `dones_success_rate`: the episode-balanced mean fraction of steps satisfying Factory's original
  `overall_success` predicate;
- `dones_success_rate_all_env_steps`: the same per-step environment-population signal averaged across the full eval;
- `seed_<seed>_dones_success_windows.csv`: consecutive 128-step averages matching RL-Games' TensorBoard windowing;
- `terminal_success_rate`: the separate fraction of episodes successful at timeout; and
- the existing per-episode subtask and physical diagnostics.

Use `-DryRun` to validate all checkpoint mappings without launching simulation. This dedicated correction launcher
always replaces existing Factory summary and episode files because their former `dones_success_rate` field had the
wrong meaning; its Factory-only task filter prevents replacement of Lift or Cabinet artifacts.
