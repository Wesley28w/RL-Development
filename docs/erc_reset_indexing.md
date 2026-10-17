# ERC reset indexing and training defaults

Completion bank `i` contains states recorded after milestone `i` completes.
For ACES and its difficulty-driven ablations, scheduler category `i` is the
milestone to practice, not the completion bank to replay:

| Category (zero-based) | Reset source |
| --- | --- |
| 0: reach A | Original baseline reset |
| 1: reach B | A-completion bank 0 |
| 2: reach C | B-completion bank 1 |

The shared implementation is `isaaclab_tasks/utils/reset_buffer.py`. Availability
is checked against the required predecessor bank, so reaching B need not have
succeeded before A states can be used to practice it. Category 0 never wraps
around to the final bank. Only populated slots are sampled, and unavailable
targets receive no reset probability after availability masking.

This mapping is used by direct Cabinet, Factory, Lift, and the manager-based
Cabinet port (including its OpenArm configuration). Cabinet's ablation variants
inherit the same reset path. TSCL already chooses reset banks directly, rather
than milestone difficulties, so its bank-category semantics remain unchanged.

## Reset and noise behavior

- No joint perturbation is added to empirical replay states. The
  `curriculum_dr` option and curriculum-added action/observation noise are removed.
- Original baseline reset randomization is retained for natural episodes only.
  In manager-based tasks, replay restoration runs before the natural-only reset
  events, which exclude replayed environments.
- Factory randomizes only the natural subset, clears the original sensor offset
  for replay episodes, refreshes the fixed-asset observation/action frame, and
  clears teleport-induced finite-difference motion in replay observations.
- A category-0 reset is a normal baseline reset: it is marked natural and may
  contribute completion states and unbiased completion statistics.
- `curriculum/sample_rate` records actual replays, not category-0 selections.
  The configured sampling ratio is therefore an upper bound on replay fraction.

## New training defaults

| Franka benchmark | Parallel environments | Training iterations |
| --- | ---: | ---: |
| Factory | 256 | 150 rl-games epochs |
| Lift | 4,096 | 1,500 RSL-RL iterations |
| Cabinet | 8,192 | 1,250 RSL-RL iterations |

Factory's progress clock uses its 128-step rollout horizon. Cabinet's delay clock
uses the new 1,250-iteration budget. TSCL teacher parameters and update cadence
are otherwise unchanged. Play/evaluation environment counts are not changed.
Command-line overrides still take precedence over these defaults.

These changes apply to new training runs. Existing checkpoints, evaluation logs,
and manuscript results describe the previous implementation and are not rewritten.

## Regression checks

Use the IsaacLab Python environment for these commands:

```powershell
python -m pytest -q source/isaaclab_tasks/test/test_reset_buffer_indexing.py
python -m pytest -q source/isaaclab_tasks/test/test_reset_buffer_simulation.py
python -m pytest -q source/isaaclab_tasks/test/test_franka_cabinet_uniform_curriculum.py
python -m pytest -q source/isaaclab_tasks/test/test_factory_aces_curriculum.py source/isaaclab_tasks/test/test_tscl_window_teacher.py
```

The simulator checks use four environments per task, no policy training, and
separate simulator processes to avoid cross-task stage-teardown deadlocks.
