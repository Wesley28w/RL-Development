# Reverse Curriculum Generation on Franka Cabinet

Implementation notes for the RCG arm of the curriculum benchmark.

**Reference.** C. Florensa, D. Held, M. Wulfmeier, M. Zhang, P. Abbeel, *Reverse Curriculum
Generation for Reinforcement Learning*, CoRL 2017. [arXiv:1707.05300](https://arxiv.org/abs/1707.05300)

RCG does not change the reward, the observation, the termination condition or the policy. It
changes **where episodes start**. Training begins from states close to a known goal state and
the start distribution expands backwards as the policy improves, so the agent always trains on
starts it can sometimes but not always solve.

---

## Algorithm, as implemented

```
Algorithm 1 (Policy Training)                     where it lives
Input: pi_0, s^g, rho_0, N_new, N_old,
       R_min, R_max, Iter
starts_old <- [s^g]                               initialize_rcg()
starts, rews <- [s^g], [1]
for i <- 1 to Iter:
  starts <- SampleNearby(starts, N_new)           _brownian_expand()
  starts.append(sample(starts_old, N_old))        _sample_state_pool()
  rho_i <- Unif(starts)                           _rcg_reset_from_pool(), on every reset
  pi_i, rews <- train_pol(rho_i, pi_{i-1})        RCGOnPolicyRunner.learn()
  starts <- select(starts, rews, R_min, R_max)    _select_good_starts()
  starts_old.append(starts)                       advance_rcg_stage()

Procedure 2 (SampleNearby)                        _brownian_expand()
Input: starts, N_new, Sigma, T_B, M
while len(starts) < M:
  s_0 ~ Unif(starts)
  for t <- 1 to T_B:
    a_t = eps_t,  eps_t ~ N(0, Sigma)
    s_t ~ P(s_t | s_{t-1}, a_t)
    starts.append(s_t)
starts_new <- sample(starts, N_new)
```

Good starts are `S_0^i = {s_0 : R_min < R(pi_i, s_0) < R_max}`, with `R` estimated as the
empirical success rate from that start over the stage.

Three details of the paper that are easy to get wrong and are honoured here:

- `select()` runs over the **whole** pool — the `N_new` new starts *and* the `N_old` replayed
  ones — not only the new states.
- `SampleNearby`'s candidate set **includes the seed states**: Procedure 2 appends visited
  states to the same list that `s_0` is drawn from and that is finally subsampled.
- Candidate feasibility is guaranteed **by construction**. States come from real rollouts of
  the simulator, never from noise injected directly into state space.

Paper hyperparameters are the defaults in `rcg_cfg.py`: `R_min = 0.1`, `R_max = 0.9`,
`N_new = 200`, `N_old = 100`, `M = 10_000`, `T_B = 50`, `Sigma = I`.

---

## Files

| File | Role |
|---|---|
| `rcg_cfg.py` | `RCGCfg`. Pure config, no RL-library dependency. |
| `rcg_mixin.py` | `RCGMixin`. Pools, statistics, selection, Brownian expansion, stage scheduling. Task-agnostic apart from three hooks. |
| `franka_cabinet_env.py` | The three hooks, plus `FrankaCabinetRCGEnvCfg`. |
| `agents/rsl_rl_ppo_cfg.py` | `FrankaCabinetRCGPPORunnerCfg` — identical PPO hyperparameters, different runner. |
| `isaaclab_rl/rsl_rl/rcg_runner.py` | `RCGOnPolicyRunner`. Decides *when* a stage ends; nothing else. |
| `scripts/rcg/record_goal_states.py` | Produces `s^g` from a trained checkpoint. |
| `scripts/rcg/test_state_roundtrip.py` | Gate test for capture/restore, plus a SampleNearby dry run. |
| `scripts/rcg/evaluate.py` | Measures success rate from `rho_0`. The number the benchmark reports. |

The state is captured as joint coordinates only:

| field | shape | why |
|---|---|---|
| `robot_joint_pos` | `(N, 9)` | |
| `robot_joint_vel` | `(N, 9)` | a genuine visited state, so velocities are kept |
| `robot_dof_targets` | `(N, 9)` | the environment's own integrator state (see below) |
| `cabinet_joint_pos` | `(N, 4)` | all four joints, not only the top drawer |
| `cabinet_joint_vel` | `(N, 4)` | |

Both articulations are fixed-base and their root poses are never written, so root state is not
part of the MDP state here. Nothing in the dict is a world-frame position, so the
`scene.env_origins` conversion that other tasks need does not arise — `_rcg_capture_state`'s
docstring spells out the rule for tasks where it does.

---

## Documented deviations from the paper

Each is a config flag, defaulting to the behaviour stated here.

1. **Dense reward for training, binary success for the curriculum.** The paper trains on the
   sparse indicator `r(s) = 1{s in S^g}`. Isaac Lab's Franka Cabinet uses a five-term dense
   reward, and keeping it is what makes RCG and the PPO baseline optimise the same objective.
   The `R_min`/`R_max` criterion always uses **binary task success**, never the dense return —
   a threshold on a dense return that sums distance, rotation, opening, finger and action terms
   would not mean anything. `rcg.sparse_reward = True` runs the paper-faithful variant as a
   third arm.

2. **Already-solved candidates are rejected** (`reject_solved_candidates = True`). Franka
   Cabinet terminates on `drawer_top_joint > 0.39`, which is also the success test, so a
   candidate captured past that threshold is terminal: an episode from it ends at step 1 with
   guaranteed success. It can only ever be filtered out as "mastered" while occupying a pool
   slot. Equivalent to restricting `rho_i` to the complement of the goal set.

3. **A set of goal states rather than a single `s^g`** (`num_goal_states = None`). The recorder
   collects ~1000 success states. Set `num_goal_states = 1` for strict fidelity.

4. **Stage length is budget *and* coverage based.** See "Stage scheduling" below.

5. **`robot_dof_targets` is reset on every reset** (`reset_dof_targets = True`). Upstream
   `_reset_idx` calls `set_joint_position_target(joint_pos)` but never updates
   `self.robot_dof_targets`, which is the buffer `_pre_physics_step` actually integrates from —
   so the target leaks across episodes. RCG cannot work without this buffer being part of the
   restored state. The fix is applied to the **normal reset path as well**, so both arms of the
   benchmark share identical reset semantics; baseline numbers may therefore differ slightly
   from published Isaac Lab Franka Cabinet results. `reset_dof_targets = False` recovers the
   upstream behaviour for an ablation.

6. **Bounded replay archive** (`old_starts_capacity = 100_000`). The paper's `starts_old` list
   is unbounded; oldest entries are evicted here.

`brownian_state_noise_std` defaults to `0.0`, i.e. the paper's pure action-space random walk.
It exists only as a fallback for the risk described next, and measurement says it is not
needed.

---

## Stage scheduling

The paper's `train_pol` is a fixed training budget per iteration. Naively porting "250k steps
per stage" to this task does not work: with 500-step episodes and a 300-state pool, 250k steps
is about **1.7 episodes per start state**, and `R_min = 0.1` is not estimable from 1.7 samples,
so `select()` returns empty and the curriculum never advances.

The default is therefore `policy_steps_per_stage = 1_200_000`
(≈ 300 starts × 8 episodes × 500 steps ≈ 18 PPO iterations at 4096 envs × 16 steps), gated by a
coverage criterion: a stage ends once at least `min_attempts_coverage` (0.9) of the pool has
been attempted `min_attempts_per_start` (5) times, with `max_policy_steps_per_stage` (4M) as a
hard cap.

Coverage rather than a strict minimum is deliberate. Episodes are assigned to starts at random,
so the minimum attempt count lags the mean badly — at a mean of 10 attempts, about 3% of a
300-state pool still sits below 5, which would push every stage to the hard cap. The few
under-sampled starts are simply not selectable as good starts that round.

`init_at_random_ep_len=True` (already the default in `train.py`) is kept: it desynchronises
episode boundaries, which is what lets attempt counts accumulate smoothly instead of in
500-step bursts.

The stage boundary sits **after the policy update and before the next rollout**:

```
16 rollout steps -> compute returns -> PPO update -> [RCG stage boundary] -> next rollout
```

so the rollout storage only ever holds genuine on-policy transitions collected under a single
start distribution. `advance_rcg_stage()` teleports every environment, so the runner refreshes
its cached observations immediately afterwards; without that the first transition of the next
rollout would pair old observations with new states.

---

## The feasibility question, and its answer

The Brownian motion acts through the **action space**, which for this task means the arm's
joint-position targets. The drawer is under-actuated from the arm's point of view, so it is not
obvious that random arm motion produces starts that are *less* far along the task than the
goal. If every candidate kept `drawer ≈ 0.4`, all of them would be trivially successful, all
would be filtered as mastered, and the curriculum could never expand.

It works, for a mechanical reason: the cabinet's drawer joint is an `ImplicitActuator` with
`stiffness = 10`, `damping = 1`, and its position target is never commanded away from 0. An
open drawer is therefore actively sprung closed (torque ≈ 10 × 0.4 = 4 N) unless the gripper
holds it.

Measured on 32 environments, one `SampleNearby` from solved seed states:

```
new-start drawer opening: mean 0.2157   min 0.0000   max 0.3899
candidates generated 1268, rejected as already-solved 364
fraction of new starts below 0.9 * threshold: 0.825
```

So the pool genuinely regresses away from the goal, and `brownian_state_noise_std` can stay at
`0.0`. Re-run `test_state_roundtrip.py --dry_run_expand` if any of the cabinet actuator gains,
`brownian_horizon` or `drawer_open_threshold` change — the conclusion depends on all of them.

---

## Reproducing the benchmark

### 1. Baseline PPO (also produces the goal states)

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Franka-Cabinet-Direct-v0 --headless
```

### 2. Record `s^g`

```bash
isaaclab.bat -p scripts/rcg/record_goal_states.py \
    --task Isaac-Franka-Cabinet-Direct-v0 \
    --checkpoint logs/rsl_rl/franka_cabinet_direct/<run>/model_1499.pt \
    --num_states 1000 --headless
```

Writes `franka_cabinet/data/goal_states_franka_cabinet.pt` and prints the drawer opening and
per-joint spread of the recorded set. A near-zero arm-joint spread means the goal set collapsed
to one pose and the curriculum will expand from a single configuration — record again with more
environments.

A *partially* trained checkpoint is fine; only a few hundred success states are needed. But do
not hand-build a goal state with the drawer open and the arm at its default pose: the
informative curriculum dimension for this task is the arm configuration, and expanding
backwards from such a state never produces starts with the gripper near the handle.

### 3. Verify capture/restore before trusting anything

```bash
isaaclab.bat -p scripts/rcg/test_state_roundtrip.py --num_envs 64 --headless
isaaclab.bat -p scripts/rcg/test_state_roundtrip.py --num_envs 256 --dry_run_expand --headless
```

Measured on 64 environments:

| check | result |
|---|---|
| round trip (capture → scramble → restore → capture) | `0.0` on all five fields |
| determinism control (same restore replayed twice) | `0.0` on all five fields |
| replay residual vs. the natural trajectory | `6.1e-4` rad on joint positions, `2.6e-3` rad/s on joint velocities — `7e-4` of the `0.84` rad the trajectory travelled |

So restore is exactly reproducible, but a replayed trajectory drifts slightly from the natural
one. That remainder is **not** a missing field: it lives in PhysX articulation solver caches and
contact impulses, which Isaac Lab's `Articulation` API does not expose, so no addition to the
state dict can capture it. It is bounded, and at `7e-4` of the motion it is orders of magnitude
below the ±0.125 rad reset randomisation the task already applies to every episode — which is
the standard by which it should be judged.

The test's gate is therefore relative (residual ≤ 5% of the motion) rather than a fixed absolute
tolerance. **If it fails, stop.** Every part of RCG rests on this, and no amount of correct
curriculum logic can compensate for start states the policy can never actually be in.

### 4. RCG training

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Franka-Cabinet-RCG-Direct-v0 --headless
```

### 5. Comparable evaluation

Training-time success rate is **not** comparable across the two arms: RCG trains on curriculum
starts, the baseline on `rho_0`. Evaluate both from `rho_0`:

```bash
isaaclab.bat -p scripts/rcg/evaluate.py --task Isaac-Franka-Cabinet-Direct-v0 \
    --run_dir logs/rsl_rl/franka_cabinet_direct/<run> --episodes 512 --headless
isaaclab.bat -p scripts/rcg/evaluate.py --task Isaac-Franka-Cabinet-RCG-Direct-v0 \
    --run_dir logs/rsl_rl/franka_cabinet_rcg/<run> --episodes 512 --headless
```

Each writes `rho0_eval.csv` in the run directory. `play.py` also forces `rcg.enabled = False`,
so interactive inspection is always on `rho_0` too.

---

## Diagnostics to watch

Logged under `rcg/` in TensorBoard:

| key | what it tells you |
|---|---|
| `rcg/stage` | stage advances. Flat means `select()` keeps returning empty. |
| `rcg/pool_drawer_mean` | **the curriculum working.** Should drift *down* across stages. |
| `rcg/frac_good_starts` | fraction of the pool inside `(R_min, R_max)`. Near 0 stalls the curriculum. |
| `rcg/mean_success_rate` | mean success over attempted starts. |
| `rcg/frac_min_attempts_met` | should approach 1 before each advance; if not, stages are hitting the hard cap. |
| `rcg/frac_unattempted` | starts never sampled this stage. Should fall to ~0. |
| `rcg/candidates_rejected_solved` | how much of SampleNearby lands back in the goal set. |
| `rcg/pool_size`, `rcg/old_starts_size` | pool and archive sizes. |

Reward should not discontinuously collapse at a stage boundary. If it does, suspect the
observation refresh in the runner.

These metrics describe the *rollout*, which is collected before the stage boundary, so they lag
the `[RCG] Stage N: ...` console lines by one iteration. That is expected, not a bug.

A short smoke run (64 envs, shortened episodes, `r_min`/`r_max` widened so every stage advances)
shows the mechanism working:

```
stage 0 -> 1 -> 2 -> 3 -> 4
rcg/pool_drawer_mean  0.353  0.257  0.216  0.143  0.149
```

The pool marches away from the goal while the reward keeps rising, which is exactly the
behaviour a reverse curriculum is supposed to produce.

---

## Implementation gotchas

Two non-obvious constraints, both found by running the thing rather than by reading it:

- **RCG's simulator writes must run inside `torch.inference_mode()`, not `torch.no_grad()`.**
  `Articulation.write_joint_velocity_to_sim` updates `data.joint_acc` in place, and that buffer
  is an *inference tensor* whenever the RL runner last refreshed it inside its own
  `torch.inference_mode()` rollout. Writing to it from `no_grad` raises *"Inplace update to
  inference tensor outside InferenceMode"*. Isaac Lab's write APIs are only ever exercised from
  inside `env.step()`, so they implicitly assume that context; `initialize_rcg`,
  `advance_rcg_stage` and `_brownian_expand` establish it explicitly. Note that this failure is
  invisible to a test script that uses `no_grad` throughout — it only appears under the real
  runner.
- **Never mutate `self.actions` in place from a curriculum method.** `_pre_physics_step` rebinds
  it during the rollout, so it too is an inference tensor; `_rcg_reset_all_envs` allocates a
  fresh buffer instead.

## Known limitations

- **`--resume` restarts the curriculum.** The start-state pool and archive are not in the
  checkpoint, so resuming reloads the policy but rebuilds the pool from `s^g`. Resumed runs are
  not equivalent to uninterrupted ones.
- **Recurrent policies are rejected.** `advance_rcg_stage()` teleports every environment, which
  would require the policy's hidden states to be reset at the same moment.
  `RCGOnPolicyRunner.learn` raises rather than leaving this as a silent trap.
- **`advance_rcg_stage()` discards in-flight episodes.** This matches the paper's alternation
  between training and curriculum-generation phases, but it does mean a fraction of collected
  experience ends mid-episode at every stage boundary.
- **Restore is not bit-identical to the natural trajectory.** Reproducible, and within `7e-4` of
  the motion, but not exact — see the table above. The residual is PhysX solver state that is not
  reachable through the `Articulation` API. If a future task turns out to be sensitive to it, the
  fix is a simulator-level state snapshot, not more fields in `_rcg_capture_state`.
- **Lift and Factory are not wired up.** `RCGMixin` is task-agnostic behind three hooks
  (`_rcg_capture_state`, `_rcg_restore_state`, `_rcg_is_solved`), so adding them is a matter of
  declaring the state and the success condition — but no such code exists yet. Those tasks have
  free-floating objects, so their `_rcg_capture_state` **must** convert object positions to
  environment-local coordinates by subtracting `scene.env_origins[env_ids]`; environment clones
  sit at different world offsets, and storing absolute world positions is the easiest way to
  break this silently.
