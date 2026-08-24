# Empirical Reset Curriculum — Supplemental Environments

This folder is a NeurIPS workshop supplemental containing the IsaacLab task code
for an **empirical, success-rate-driven reset-pose curriculum**: a fraction of
episode resets replay a previously-recorded, partially-completed state for a
robot's harder subtasks (sampled in proportion to how often each subtask is
currently *failing*), instead of always resetting from scratch. This is
implemented and tuned across three manipulation tasks:

- **`factory/`** — Factory contact-rich assembly (`Isaac-Factory-NutThread-Direct-v0`,
  and its siblings `Isaac-Factory-PegInsert-Direct-v0` / `Isaac-Factory-GearMesh-Direct-v0`),
  a Direct-workflow IsaacLab task.
- **`franka_cabinet/`** — Franka drawer-opening (`Isaac-Franka-Cabinet-Direct-v0`),
  a Direct-workflow IsaacLab task.
- **`lift/`** — Franka cube lift-to-target (`Isaac-Lift-Cube-Franka-v0`), a
  manager-based IsaacLab task (the curriculum is wired in as an `EventTermCfg` +
  `CurriculumTermCfg` pair rather than hand-rolled in the env class).

Each environment implements the same underlying idea independently (Factory's and
Franka Cabinet's code share a near-identical structure since one was adapted from
the other; Lift's is a from-scratch manager-based port with the cleanest
documentation of the three — read `lift/mdp/curriculums.py` and `lift/mdp/events.py`
first if you want the clearest explanation of the mechanism).

This folder is **not a standalone runnable repo**. It's a drop-in source
supplement: you need a working [IsaacLab](https://github.com/isaac-sim/IsaacLab)
installation (with Isaac Sim) to run it.

## 1. Requirements

A working IsaacLab install (see the main [IsaacLab installation
docs](https://isaac-sim.github.io/IsaacLab/)) with the `isaaclab_tasks` package
installed in editable mode, as is standard for a source checkout.

## 2. Installing these environments into your IsaacLab checkout

From your IsaacLab root (`<ISAACLAB_ROOT>/source/isaaclab_tasks/isaaclab_tasks/`):

**Factory** — copy the whole folder in, replacing the existing task:
```
<ISAACLAB_ROOT>/source/isaaclab_tasks/isaaclab_tasks/direct/factory/
```
i.e. `neurips/factory/*` → `direct/factory/*` (overwrite).

**Franka Cabinet** — same, straight folder copy:
```
neurips/franka_cabinet/*  →  <ISAACLAB_ROOT>/source/isaaclab_tasks/isaaclab_tasks/direct/franka_cabinet/*
```

**Lift** — this one is a **merge**, not a straight copy, since the `lift/`
package in IsaacLab also hosts task variants this supplement doesn't include
(IK control, teddy bear object, OpenArm robot, etc.):
```
neurips/lift/lift_env_cfg.py        → manager_based/manipulation/lift/lift_env_cfg.py            (replace)
neurips/lift/mdp/*.py               → manager_based/manipulation/lift/mdp/*.py                     (replace/add)
neurips/lift/__init__.py            → manager_based/manipulation/lift/config/franka/__init__.py    (replace)
neurips/lift/joint_pos_env_cfg.py   → manager_based/manipulation/lift/config/franka/joint_pos_env_cfg.py (replace)
neurips/lift/agents/*               → manager_based/manipulation/lift/config/franka/agents/*        (replace)
```

After copying, no reinstall is needed if `isaaclab_tasks` is already an editable
install (`pip install -e source/isaaclab_tasks`, the default IsaacLab setup) —
gym task registration happens at import time.

## 3. Running training

Run from `<ISAACLAB_ROOT>`. Each task uses a different RL library entry point —
Factory only registers an `rl_games` agent config, Cabinet and Lift use `rsl_rl`:

```bat
:: Factory (NutThread) — rl_games
scripts\reinforcement_learning\rl_games\train.py --task Isaac-Factory-NutThread-Direct-v0 --num_envs 512 --headless --seed 42

:: Franka Cabinet — rsl_rl
scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Franka-Cabinet-Direct-v0 --num_envs 16384 --headless --seed 42

:: Franka Lift — rsl_rl
scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Lift-Cube-Franka-v0 --num_envs 8192 --headless --seed 42
```

Factory's other two tasks use the identical curriculum code — swap the task name
to try either:
- `Isaac-Factory-PegInsert-Direct-v0`
- `Isaac-Factory-GearMesh-Direct-v0`

## 4. Key curriculum knobs

All three environments expose the same set of hyperparameters (Factory and
Franka Cabinet as class attributes on their env cfg; Lift as attributes on
`LiftEnvCfg` in `lift_env_cfg.py`):

| Field | What it does |
|---|---|
| `reset_state_curriculum_enabled` | Master on/off switch. Set `False` for a curriculum-free baseline — everything below becomes a no-op. |
| `sampling_ratio` | Fraction of resets on a given step that replay a curriculum (partially-completed) state instead of a normal reset. |
| `success_buffer_size` | Number of recorded partially-completed poses kept per subtask to sample replay states from. |
| `curriculum_dr` | Domain randomization noise added to a replayed pose before use, so replays aren't bit-identical repeats. |
| `success_rate_alpha` | Momentum/EMA coefficient for the per-subtask success-rate estimate driving the sampling distribution. |
| `prob_exp` | Sharpening exponent applied to the sampling distribution (1 = uniform over subtask difficulty, higher = more concentrated on the hardest subtask). |
| `greedy_margin` | Margin between the top two subtask scores below which sampling blends toward softmax instead of picking the hardest greedily. |
| `action_std` / `observation_std` *(Factory & Cabinet only)* | Optional action/observation noise applied only to curriculum (replayed) episodes. |

Franka Cabinet and Factory also read/write these directly on `self`/`self.cfg`
inside `franka_cabinet_env.py` / `factory_env.py`. Lift wires the same fields
into the reset event (`mdp/events.py:sample_curriculum_reset_state`) and the
bookkeeping term (`mdp/events.py:subtask_progression_tracker`), both gated by
`reset_state_curriculum_enabled`.

## 5. What to look at in TensorBoard / W&B

Each environment logs curriculum diagnostics under a `curriculum/*` (Factory,
Cabinet) or `Curriculum/*` (Lift, via the curriculum manager) namespace:

- `curriculum/success_rate_<i>`, `curriculum/difficulty_<i>` — per-subtask
  success rate and derived difficulty (`1 - success_rate`).
- `curriculum/distribution_<i>` — sampling probability assigned to subtask `i`.
- `curriculum/blend`, `curriculum/margin`, `curriculum/selected` — the
  greedy/softmax blend factor, the margin that drove it, and the subtask
  actually picked most often.
- `curriculum/natural`, `curriculum/sample_rate` — fraction of resets that were
  ordinary (non-curriculum) vs. curriculum replays on a given step.
- Factory additionally logs `env_compare/*` — success rate split between
  curriculum-replay environments and natural (eval) environments, and the gap
  between them, per subtask.
