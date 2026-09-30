# Reverse Curriculum Generation on Franka Lift

Implementation notes for the RCG arm of the curriculum benchmark on the lift task.

**Reference.** C. Florensa, D. Held, M. Wulfmeier, M. Zhang, P. Abbeel, *Reverse Curriculum
Generation for Reinforcement Learning*, CoRL 2017. [arXiv:1707.05300](https://arxiv.org/abs/1707.05300)

RCG does not change the reward, the observation, the termination condition or the policy. It
changes **where episodes start**. Training begins from states close to a known goal state and
the start distribution expands backwards as the policy improves, so the agent always trains on
starts it can sometimes but not always solve.

The curriculum logic is the same code that runs on Franka Cabinet — it lives in
`isaaclab_tasks/utils/rcg/` and is shared, not copied, which is what makes "the same curriculum on
two tasks" a checkable claim rather than an assertion. Read
[`direct/franka_cabinet/RCG.md`](../../../direct/franka_cabinet/RCG.md) for the algorithm mapping,
the stage-scheduling argument and the deviations that apply to both tasks. **This file covers only
what is specific to lifting a cube**, which is most of what is interesting: lift is a different
workflow, has a goal-conditioned MDP, has a free-floating object, and never terminates on success.

---

## Tasks

| id | arm | env cfg | runner cfg | log dir |
|---|---|---|---|---|
| `Isaac-Lift-Cube-Franka-v0` | — | `FrankaCubeLiftEnvCfg` | `LiftCubePPORunnerCfg` | `franka_lift` |
| `Isaac-Lift-Cube-Franka-Baseline-v0` | PPO baseline | `FrankaCubeLiftBaselineEnvCfg` | `LiftCubeBaselinePPORunnerCfg` | `franka_lift_baseline` |
| `Isaac-Lift-Cube-Franka-RCG-v0` | RCG | `FrankaCubeLiftRCGEnvCfg` | `LiftCubeRCGPPORunnerCfg` | `franka_lift_rcg` |

`Isaac-Lift-Cube-Franka-v0` is **untouched** — same entry point, same config, same behaviour as
upstream ships it. The benchmark uses the other two, which share one environment class
(`LiftRCGEnv`) and one config ancestry and differ by exactly one boolean, `rcg.enabled`. So the
measured baseline is *code*-identical to the RCG arm rather than merely configured to look like it,
and nothing outside the benchmark changes.

The two benchmark arms differ from upstream Franka Lift only by the `rcg_success_tracker` reward
term. Its return value is a tensor of exact zeros on every step, so the reward signal is
bit-identical, and it draws nothing from the RNG — see "Why a reward term" below, because that last
point is not incidental.

---

## Files

| File | Role |
|---|---|
| `lift_rcg_env.py` | `LiftRCGEnv`. The three task hooks, the reset path, the success metrics. |
| `mdp/rcg.py` | `rcg_success_tracker`. The one per-step hook, as a zero-valued reward term. |
| `lift_env_cfg.py` | `success_threshold`, `success_rate_ema_alpha`, `rcg`, `progress_reference_distance`. Inert unless the env class is `LiftRCGEnv`. |
| `config/franka/lift_rcg_env_cfg.py` | The two benchmark arms. |
| `config/franka/agents/rsl_rl_ppo_cfg.py` | Both runner configs — identical PPO hyperparameters. |
| `scripts/rcg/quick_lift_eval.py` | Per-seed lift report from `rho_0`, with per-group mean ± sd. |
| `scripts/rcg/run_lift_seeds.ps1` | Seed sweep, one arm per invocation. |

`record_goal_states.py`, `test_state_roundtrip.py`, `evaluate.py` and `permutation_test.py` under
`scripts/rcg/` are task-agnostic and serve both tasks.

---

## The state

23 numbers by default:

| field | shape | why |
|---|---|---|
| `robot_joint_pos` | `(N, 9)` | 7 arm joints + 2 gripper fingers |
| `object_pos` | `(N, 3)` | cube position, **relative to `scene.env_origins`** |
| `object_quat` | `(N, 4)` | cube orientation |
| `goal_pose_b` | `(N, 7)` | the commanded goal, in the robot's root frame |

and 38 with `rcg.capture_full_state = True`, which adds `robot_joint_vel` `(N, 9)` and `object_vel`
`(N, 6)`. See "Verify capture/restore" below: that flag is the difference between a start state that
is merely *valid* and one that is the state that was recorded, and the gap between them is measured
rather than assumed.

Three things here that Franka Cabinet did not have to deal with.

### The goal is part of the state

`object_pose` is a `UniformPoseCommand`, resampled at every reset, and it enters both the
observation and three of the six reward terms. A start state restored *without* its goal would be
paired with a fresh random one — so `R(pi_i, s_0)`, the success probability of that start state and
the entire basis of the good-start criterion, would not be a property of the start state at all. A
cube 1 cm from goal A tells you nothing about goal B.

So the commanded pose travels with the state, stored in the robot's root frame exactly as the
command term's `pose_command_b` buffer holds it, which makes capture and restore bit-exact
inverses. `time_left` is deliberately *not* stored: a restored state begins a new episode, so its
resampling clock starts from the full interval, as a natural episode's does. (With
`resampling_time_range = (5.0, 5.0)` and `episode_length_s = 5.0` the command never resamples
mid-episode anyway, but storing `time_left` would break that if either value ever changed.)

**This is a deliberate difference from the reset-pose curriculum**, which replays a recorded pose
against a freshly sampled goal. It is not a free choice: without it the good-start criterion
estimates a start state's success probability marginalised over goals, which flattens exactly the
signal the curriculum runs on.

### There is a free-floating object

`object_pos` is stored relative to `scene.env_origins`, because environment clones sit at different
world offsets and storing an absolute world position is the easiest way to break a state pool
silently — it would work perfectly in environment 0 and teleport the cube into the neighbouring
table everywhere else.

### There is no action-target buffer

Franka Cabinet needed `robot_dof_targets` in its state, and needed its reset path fixed, because
`_pre_physics_step` integrates from that buffer. Franka Lift's `JointPositionAction` writes an
*absolute* target, `action * scale + default_joint_pos`, on every step. Nothing is integrated across
episodes and nothing leaks, so `rcg.reset_dof_targets` is simply not consulted here.

One consequence worth knowing: on the first step after a curriculum restore, the target is computed
from the policy's action about the *default* pose, not about the restored pose. The arm therefore
gets one step of transient toward wherever that action points. This is not a bug introduced by RCG —
it is what the action term does at every reset — and the policy sees the restored joint positions in
its observation, so it can command accordingly. The reset-pose curriculum has the identical property.

Velocities are not stored by default, and are set to zero on restore. Same deviation from the paper
as on cabinet, and for the same reason: it keeps the two curricula's start states carrying the same
information. `rcg.capture_full_state = True` stores them.

---

## Success, without a success termination

Upstream Franka Lift terminates only on `time_out` and `object_dropping`. That is **left alone** —
adding a success termination would change the task rather than the curriculum, and would hand RCG a
different MDP from the published one. Success is therefore something *measured*, not something the
episode ends on:

- `_rcg_is_solved()` — the object is within `success_threshold` (0.02 m) of the commanded goal
  position. One definition, used by the tracker, by the metrics, and by the curriculum's
  already-solved candidate filter, so they cannot drift apart.
- `rcg_currently_solved` — the instantaneous test, refreshed every step by the tracker term. Read
  from `_reset_idx`, where it still holds the *final* step's value: terminal success.
- `rcg_episode_solved` — the same thing made sticky for an episode: "did this episode ever reach the
  goal".

**The curriculum scores a finished episode by `ever`, not by `terminal`**
(`rcg.episode_success_mode = "ever"`). This matters more than it looks. The paper's episodes end the
moment the goal set is entered, so `R(pi_i, s_0)` there is the probability of *reaching* the goal
from `s_0`. Under `"terminal"` on a task that always runs to its time limit, a start state one step
from the goal is instead scored on whether the policy can *hold* the cube in place for 250 steps —
a far harder question, which answers `0` for nearly every start early in training. `select()` then
returns empty every stage and the curriculum never leaves stage 0. `"terminal"` remains available as
an ablation.

Why 0.02 m: it is the threshold the reset-pose curriculum's `lift_success_rate` metric uses, so the
two curricula's curves are on one scale, and it sits well inside the 0.05 `std` of the
`object_goal_tracking_fine_grained` reward term. No separate height test is needed — the goal is
sampled at `z` in `(0.25, 0.5)` and the cube rests at `z ≈ 0.055`, so the object cannot be within
2 cm of the goal without having been lifted.

### Why a reward term

The manager-based workflow has no per-step override point on the environment class, the way
`DirectRLEnv._get_dones` does, so the per-step goal evaluation has to come from a manager term.
`RewardTermCfg` rather than `EventTermCfg(mode="interval")`, because registering *any* interval
event makes `EventManager` call `torch.rand` on the global RNG — once per reset and once per step,
unconditionally, regardless of what the term itself does. That shift alone is enough to make the
baseline arm diverge from an unmodified upstream run, since everything else drawing on the same
generator moves downstream of it. `RewardManager` has no such hidden timer.

The weight has to be non-zero (`1.0`): `RewardManager.compute` skips zero-weight terms as a
micro-optimisation, and a term that is never called tracks nothing. `1.0 × 0 × dt` is still exactly
`0`.

This is the same reasoning, and the same shape, as the reset-pose curriculum's
`subtask_progression_tracker`.

---

## Configuration differences from cabinet

Everything else in `RCGCfg` is at the paper's value. These three are task-sized:

| field | cabinet | lift | why |
|---|---|---|---|
| `policy_steps_per_stage` | 1 200 000 | 600 000 | 300 starts × 8 episodes × **250**-step episodes |
| `max_policy_steps_per_stage` | 4 000 000 | 2 000 000 | same halving |
| `episode_success_mode` | `terminal` | `ever` | lift does not terminate on success — see above |
| `brownian_horizon` | 50 (the paper's `T_B`) | **10** | a free cube has no restoring force — see below |

### `brownian_horizon` is the one that decides whether the curriculum works at all

The paper's `T_B = 50` does not transfer to this task, and using it is the difference between a
curriculum that works and one that does nothing. Fifty steps at 50 Hz is a full second of random
joint targets applied to an arm holding a cube: the cube is long gone and the arm has wandered
anywhere. Measured with `--dry_run_expand` over 256 environments, progress of the generated starts
(`1.0` = in the goal set; the recorded goal states measure `0.99`):

| `T_B` | mean progress | distribution | `≥ 0.5 × goal` | verdict |
|---|---|---|---|---|
| 50 | 0.20 | 56% of starts at progress < 0.1, rest spread flat | 0.18 | indistinguishable from random |
| 10 | 0.75 | concentrated in [0.5, 1.0] | 0.93 | a real difficulty gradient |
| 3 | 0.90 | all inside [0.8, 1.0] | 1.00 | too close, mostly mastered |

At `T_B = 50` the new starts are unsolvable and the replayed archive starts are trivial, so every
start's success rate is exactly `0` or exactly `1`, `select()` finds nothing in `(r_min, r_max)`, and
the curriculum only ever advances on its replay path. The tell in TensorBoard is unmistakable:

```
rcg/frac_good_starts     pinned at 0.000
rcg/mean_success_rate    pinned at n_old / (n_new + n_old) = 100/300 = 0.333
```

Franka Cabinet tolerates `T_B = 50` because its drawer is spring-loaded and the arm stays near the
handle; a free cube has no such restoring force. **Re-run `--dry_run_expand` if the object, the
gripper or the episode rate changes** — the right value depends on all three.

`policy_steps_per_stage` is the other number that may need retuning, and the formula is what to
retune it by: `(n_new + n_old) × attempts × episode_length`. Too small and `select()` cannot estimate
a 0.1 success rate, so it returns empty and `rcg/stage` stays flat; too large and the run spends its
whole budget in a handful of stages.

### `clip_actions = 10.0`

Both arms set it. A circuit breaker, not a tuning knob: the action term maps an action to
`action * 0.5 + default_joint_pos` and the arm's joints span about ±2.9 rad, so anything beyond
±6 is already meaningless and `10.0` never binds on a working policy.

What it stops: `last_action` and `joint_vel_rel` are both unbounded observation terms, and
`action_rate_l2` / `joint_vel_l2` are unbounded quadratic penalties whose weights the task's own
`CurriculumCfg` multiplies by 1000 after 10k steps. A policy whose mean drifts outward therefore
drives joint targets far past their limits → joint velocities in the hundreds of rad/s → straight
back into the observation → further drift, with the quadratics turning it into per-step rewards of
−26 against a normal ceiling of +0.74. Measured without the bound, observations and actions reached
122 and `Loss/value` overflowed to `inf` within a few hundred iterations, killing roughly half of
all runs with `RuntimeError: normal expects all elements of std >= 0.0` — which is the *downstream*
symptom of `NaN` parameters, not the cause.

The baseline arm never enters that loop, so the bound is invisible there; it is applied to both arms
anyway, because a wrapper setting that differed between them would confound the comparison.

### `noise_std_type = "log"`

Both arms set it, in `LiftCubeBenchmarkPPORunnerCfg`. rsl-rl's default, `"scalar"`, makes the action
standard deviation a raw `nn.Parameter` that nothing constrains to be positive; a single bad update
can drive one of its eight components to zero, below it, or to `NaN`, and training then dies inside
`alg.update()` with `RuntimeError: normal expects all elements of std >= 0.0` — several frames away
from whatever actually went wrong. It is insurance against a confusing error message, not a cure for
divergence.

---

## Diagnostics to watch

Logged under `rcg/` and `dones/` in TensorBoard. The generic `rcg/*` keys are documented in the
cabinet file; these are the lift-specific ones:

| key | what it tells you |
|---|---|
| `rcg/pool_goal_distance_mean` | **the curriculum working.** Should drift *up* across stages. |
| `rcg/pool_object_height_mean` | should drift *down* — the cube starts closer to the table each stage. |
| `rcg/pool_goal_distance_min` / `_max` | the spread of the frontier. |
| `dones/success_rate` | terminal success, EMA over completed episodes. The primary metric. |
| `dones/success_rate_ever` | reached the goal at any point. Saturates; a yes/no, not a comparison axis. |
| `dones/eval_success_rate` | the same, on the environments held out of the curriculum. Only present when `rcg.eval_env_fraction > 0`. |
| `rcg/frac_good_starts` | **the health check.** Pinned at 0 means `select()` never finds anything and the curriculum is only advancing on its replay path. See `brownian_horizon` above. |
| `Policy/mean_std` (rsl-rl) | not an RCG metric, but the early warning for a diverging run: a healthy run dips below ~0.85 and stays there; a run that climbs monotonically from 1.0 with `Loss/learning_rate` pinned at its `1e-5` floor is on its way to a non-finite `Loss/value`. |

Training-time success rate on the RCG arm is **not** comparable to the baseline's: RCG trains on
curriculum starts. Either set `rcg.eval_env_fraction = 0.0625` for a live held-out curve, or use
`quick_lift_eval.py` / `evaluate.py`, which is the number to report.

A short smoke run (64 envs, 50-step episodes, `r_min`/`r_max` widened so every stage advances) shows
the mechanism working:

```
stage                        0       1       2       3
rcg/pool_goal_distance_mean  0.1996  0.2389  0.2500  0.2710
```

The pool marches *away* from the goal across stages, which is exactly the behaviour a reverse
curriculum is supposed to produce. On this task the number to watch rises, unlike cabinet's
`rcg/pool_drawer_mean`, which falls — both mean "further from the goal".

Note that these metrics are published from the `rcg_success_tracker` term on **every** step, not from
`_reset_idx`. That is not incidental: `advance_rcg_stage` teleports every environment at once, so for
a full episode afterwards nothing resets, and a reset-only publisher would report the previous stage
for ten iterations. See `LiftRCGEnv.publish_rcg_log`.

---

## Reproducing the benchmark

Commands are on one line because the shell here is PowerShell, where a trailing `\` is a parse error
(`Missing expression after unary operator '--'`). PowerShell's continuation character is a backtick,
and nothing — not even a space — may follow it on the line. Activate the environment first, or
`isaaclab.bat` resolves to base conda Python and fails with `No module named 'isaacsim'`:

```powershell
conda activate env_isaaclab
```

### 1. Baseline PPO (also produces the goal states)

```powershell
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lift-Cube-Franka-Baseline-v0 --headless
```

### 2. Record `s^g`

```powershell
isaaclab.bat -p scripts/rcg/record_goal_states.py --task Isaac-Lift-Cube-Franka-RCG-v0 --checkpoint logs/rsl_rl/franka_lift_baseline/<run>/model_1499.pt --num_states 1000 --headless
```

Writes `lift/data/goal_states_franka_lift.pt` and prints the per-field spread of the recorded set. A
near-zero `robot_joint_pos` spread means the goal set collapsed to one pose and the curriculum will
expand from a single configuration — record again with more environments, or from a less converged
checkpoint.

The recorder switches the curriculum off, so it records from `rho_0` even though it is pointed at
the RCG task id. A partially trained checkpoint is fine; only a few hundred success states are
needed. But do **not** hand-build a goal state with the cube parked at the goal and the arm at its
default pose: the informative curriculum dimension is the arm configuration, and expanding backwards
from such a state never produces starts with the gripper anywhere near the cube.

### 3. Verify capture/restore before trusting anything

```powershell
isaaclab.bat -p scripts/rcg/test_state_roundtrip.py --task Isaac-Lift-Cube-Franka-RCG-v0 --num_envs 64 --headless
isaaclab.bat -p scripts/rcg/test_state_roundtrip.py --task Isaac-Lift-Cube-Franka-RCG-v0 --num_envs 64 --headless env.rcg.capture_full_state=true
isaaclab.bat -p scripts/rcg/test_state_roundtrip.py --task Isaac-Lift-Cube-Franka-RCG-v0 --num_envs 256 --dry_run_expand --headless
```

Measured on 32 environments, 20 replay steps, both tasks and both state schemas:

| task | `rcg.capture_full_state` | round trip | determinism control | replay residual / motion |
|---|---|---|---|---|
| Franka Cabinet | `False` (default) | `0.0` on every field | `0.0` | **1.19** |
| Franka Cabinet | `True` | `0.0` on every field | `0.0` | `1.5e-6` |
| Franka Lift | `False` (default) | `0.0` on every field | `0.0` | **0.144** |
| Franka Lift | `True` | `0.0` on every field | `0.0` | `2.9e-6` |

Read that table carefully, because it says something that is easy to get backwards.

Restore is **exactly** reproducible on both tasks and at both settings — the control is `0.0`
everywhere, so nothing here is simulator nondeterminism. With the full state, a replayed trajectory
matches the natural one to `1.5e-6` of the distance it travelled on cabinet and `2.9e-6` on lift:
capture/restore is complete, and there is no meaningful PhysX residual left over, not even with a
cube held between two fingers.

The large numbers in the `False` rows are therefore **not** drift and **not** a defect. They are the
direct, measured cost of the positions-only decision: a restore zeroes velocities, so a replay
starts from rest where the natural trajectory had momentum, and 20 steps later the arm is somewhere
else entirely. This is exactly what the run-time restore does on every curriculum reset. A
position-only start state is still a *valid* state — it is the same class of state the task's own
reset produces, which is why the curriculum works — but it is **not** the state that was captured,
and "restore and continue is equivalent to having arrived here naturally" is false at that setting.

So the dynamics-equivalence check gates only when `rcg.capture_full_state` is set, and at the
default it prints the residual as a measurement. The round trip and the determinism control gate
always. **If either of those fails, stop.**

Which setting to use is an experimental-design choice, not a correctness one:

- `False` keeps RCG's start states carrying exactly the information the reset-pose curriculum's
  carry, so a comparison between the two curricula is not confounded by RCG having extra state.
  This is the benchmark default, and it is what the existing cabinet results were produced with.
- `True` is faithful to Florensa et al., whose start states are genuine visited states including
  velocity, and is the only setting at which the pool provably contains the states it recorded.

Run the gate test **both ways** on any new task: at the default to see what positions-only costs,
and with `env.rcg.capture_full_state=true` to check that the schema is actually complete, which is
where a real bug would show up.

The lift-specific result worth noting: with the full state, a replayed trajectory matches the natural
one to `2.9e-6` of its motion **even with the cube held between the fingers**. Frictional contact was
the obvious candidate for irreducible residual on this task, and it measures as negligible. So the
`0.144` at the default setting is entirely the zeroed velocities, not contact state.

The `--dry_run_expand` pass is the one that matters most on a new task, and its gate is **two
sided**:

- at least 10% of generated starts must sit *further* from the goal than the goal states — otherwise
  the curriculum has nowhere to expand to and every candidate is already solved;
- at least 30% must remain *within half* the goal states' progress — otherwise SampleNearby is
  producing states that are effectively random rather than nearby, which is the `T_B` failure
  described above. A one-sided check passes that case happily, which is exactly how it was missed
  the first time.

It also verifies that no start in the pool already satisfies the success condition, by restoring the
states and asking `_rcg_is_solved` rather than by inspecting the state dict, so it holds for any
task's notion of success.

### 4. RCG training

```powershell
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lift-Cube-Franka-RCG-v0 --headless
```

### 5. Comparable evaluation

```powershell
isaaclab.bat -p scripts/rcg/quick_lift_eval.py --run_glob "logs/rsl_rl/franka_lift_baseline/*" "logs/rsl_rl/franka_lift_rcg/*" --checkpoint_name model_1499.pt --output lift_eval.csv --headless
python scripts/rcg/permutation_test.py --csv lift_eval.csv --metrics success min_dist
```

`evaluate.py` is the per-checkpoint variant, for a learning curve on `rho_0` rather than one number
per run:

```powershell
isaaclab.bat -p scripts/rcg/evaluate.py --task Isaac-Lift-Cube-Franka-RCG-v0 --run_dir logs/rsl_rl/franka_lift_rcg/<run> --episodes 512 --headless
```

Both force `rcg.enabled = False`, and so does `play.py`, so every reported number is from the task's
own start distribution.

---

## Known limitations

Everything in the cabinet file's "Known limitations" applies here too — `--resume` restarts the
curriculum, recurrent policies are rejected, `advance_rcg_stage()` discards in-flight episodes, and
restore is not bit-identical to the natural trajectory. Two additions specific to lift:

- **The restore transient described above.** One step of arm motion toward the action's absolute
  target. Shared with the reset-pose curriculum, so it does not confound the comparison between
  them, but it does mean a restored state is not held perfectly still for one step.
- **Contact state is not captured.** The cube's pose and the finger joints are restored, but the
  contact impulses between them are not, and `Articulation`/`RigidObject` expose no way to. Measured,
  this costs `2.9e-6` of the motion over 20 steps, so it is not worth worrying about on this task —
  but it is the thing to suspect first if a future object, or a stiffer grasp, ever misbehaves. The
  fix would be a simulator-level state snapshot, not more fields in `_rcg_capture_state`.
