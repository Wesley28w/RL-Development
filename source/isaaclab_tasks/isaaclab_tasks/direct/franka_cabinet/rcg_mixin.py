# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reverse Curriculum Generation, implemented as a mixin for :class:`~isaaclab.envs.DirectRLEnv`.

Reference:
    C. Florensa, D. Held, M. Wulfmeier, M. Zhang, P. Abbeel,
    "Reverse Curriculum Generation for Reinforcement Learning", CoRL 2017.
    https://arxiv.org/abs/1707.05300

Mapping of the paper onto this implementation::

    Algorithm 1 (Policy Training)                    this file
    ------------------------------------------------ ----------------------------------------
    starts_old <- [s^g]                              initialize_rcg()
    starts <- SampleNearby(starts, N_new)             _brownian_expand()
    starts.append(sample(starts_old, N_old))          _sample_state_pool()
    rho_i <- Unif(starts)                             _rcg_reset_from_pool()  (every reset)
    pi_i, rews <- train_pol(rho_i, pi_{i-1})          the RL runner, between rollouts
    starts <- select(starts, rews, R_min, R_max)      _select_good_starts()
    starts_old.append(starts)                         advance_rcg_stage()

    Procedure 2 (SampleNearby)                        _brownian_expand()

A start state is represented as a ``dict[str, torch.Tensor]`` whose tensors all share a
leading "pool" dimension. A collection of such states is called a *pool*. The concrete keys
are decided by the task through :meth:`RCGMixin._rcg_capture_state`, so this mixin never needs
to know what the task's state actually contains.

The environment owns physics, state capture/restore, success bookkeeping and Brownian
expansion. The RL runner owns only the decision of *when* a stage ends, and calls:

* :meth:`RCGMixin.initialize_rcg` once, before the first rollout;
* :meth:`RCGMixin.rcg_stage_ready` after each policy update;
* :meth:`RCGMixin.advance_rcg_stage` when the former returns ``True``.
"""

from __future__ import annotations

import math
import os

import torch

StatePool = dict[str, torch.Tensor]
"""A batch of start states: tensors sharing a leading pool dimension."""


class RCGMixin:
    """Reverse curriculum generation for a :class:`~isaaclab.envs.DirectRLEnv` subclass.

    The subclass must:

    1. call :meth:`_rcg_init_buffers` at the end of its ``__init__``,
    2. call :meth:`_record_rcg_episode_results` and :meth:`_rcg_reset_from_pool` from
       ``_reset_idx`` (see :mod:`franka_cabinet_env` for the canonical shape),
    3. implement the three hooks :meth:`_rcg_capture_state`, :meth:`_rcg_restore_state` and
       :meth:`_rcg_is_solved`,
    4. expose an ``rcg`` field of type :class:`~.rcg_cfg.RCGCfg` on its configuration.
    """

    _rcg_pool_ready: bool = False
    """Class-level default so that a reset happening before :meth:`_rcg_init_buffers` (for
    instance from a base-class constructor) falls back to the task's own start distribution
    instead of raising."""

    _rcg_goal_recording_capacity: int = 0
    """Class-level default: goal-state recording is off unless explicitly started."""

    ##
    # Hooks that the task must implement.
    ##

    def _rcg_capture_state(self, env_ids: torch.Tensor) -> StatePool:
        """Capture the full physical state of the given environments.

        The returned dict must contain everything needed so that restoring it and continuing
        the simulation is equivalent to having arrived at that state naturally -- including
        any buffer the environment itself integrates over, such as an action-target buffer.

        Episode bookkeeping (elapsed steps, accumulated reward, RL state) must *not* be
        included: it belongs to the episode, not to the MDP state.

        Any field holding a world-frame position must be stored relative to
        ``self.scene.env_origins[env_ids]``, since environment clones sit at different world
        offsets. Franka Cabinet stores joint coordinates only, so it needs no such conversion.

        Args:
            env_ids: Environment indices to capture. Must be a ``torch.long`` tensor.

        Returns:
            A pool of ``len(env_ids)`` states.
        """
        raise NotImplementedError

    def _rcg_restore_state(self, env_ids: torch.Tensor, state: StatePool) -> None:
        """Write a captured state back into the simulation. Inverse of :meth:`_rcg_capture_state`.

        Args:
            env_ids: Environment indices to restore into. Must be a ``torch.long`` tensor.
            state: A pool of exactly ``len(env_ids)`` states.
        """
        raise NotImplementedError

    def _rcg_is_solved(self) -> torch.Tensor:
        """Binary task success for every environment, evaluated on the current state.

        This is the quantity behind the paper's sparse reward ``r(s) = 1{s in S^g}`` and the
        only quantity used for the good-start criterion. It must be the same condition the
        task terminates on, so that success and termination cannot drift apart.

        Returns:
            A boolean tensor of shape ``(num_envs,)``.
        """
        raise NotImplementedError

    def _rcg_apply_state_noise(self, env_ids: torch.Tensor, std: float) -> None:
        """Inject Brownian noise directly into the task's under-actuated state.

        Only called when :attr:`~.rcg_cfg.RCGCfg.brownian_state_noise_std` is positive, which
        is a documented deviation from the paper. The default implementation does nothing.
        """
        return

    def _rcg_pool_diagnostics(self, pool: StatePool) -> dict[str, torch.Tensor | float]:
        """Task-specific scalars describing a start-state pool, for logging. Optional.

        Prefer returning 0-dim tensors over Python floats: the logger accepts either, and
        tensors avoid a host/device synchronisation on every environment step.
        """
        return {}

    ##
    # Properties.
    ##

    @property
    def rcg_active(self) -> bool:
        """Whether start states should currently be drawn from the curriculum pool."""
        return self.cfg.rcg.enabled and self._rcg_pool_ready

    @property
    def rcg_pool_size(self) -> int:
        """Number of start states in the current training pool."""
        return self._pool_size(self.rcg_pool)

    @property
    def rcg_log(self) -> dict[str, torch.Tensor | float]:
        """Diagnostics to merge into ``extras["log"]``.

        Recomputes the per-step entries on access. All reductions stay on the device, so this
        costs no host/device synchronisation.
        """
        self._rcg_refresh_step_log()
        return self._rcg_log

    ##
    # Buffer setup.
    ##

    def _rcg_init_buffers(self) -> None:
        """Allocate RCG buffers. Call at the end of the task's ``__init__``."""
        self._rcg_all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        # empty pools with the right keys/dtypes, derived from a zero-width capture so that no
        # schema has to be declared twice
        self.rcg_pool: StatePool = self._empty_state_pool()
        self.rcg_old_starts: StatePool = self._empty_state_pool()

        # which pool entry each environment's current episode was started from; -1 means "not
        # from the pool", which is the case before the pool exists and immediately after the
        # statistics are cleared
        self.rcg_env_start_id = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.rcg_start_attempts = torch.zeros(0, dtype=torch.float, device=self.device)
        self.rcg_start_successes = torch.zeros(0, dtype=torch.float, device=self.device)

        self.rcg_stage = 0
        self._rcg_pool_ready = False
        self._rcg_log: dict[str, float] = {}

    ##
    # Public API used by the RL runner.
    ##

    @torch.inference_mode()
    def initialize_rcg(self) -> None:
        """Load the goal states and build the first training pool (the paper's iteration 1).

        Idempotent: a second call is a no-op. Must run before the first rollout, because until
        it has, resets fall back to the task's default start distribution.

        .. note::
            ``inference_mode``, not ``no_grad``, and the same goes for every RCG method that
            writes to the simulator. ``Articulation.write_joint_velocity_to_sim`` updates
            ``data.joint_acc`` in place, and that buffer is an *inference tensor* whenever the RL
            runner last refreshed it inside its own ``torch.inference_mode()`` rollout. Writing
            to it from ``no_grad`` then raises "Inplace update to inference tensor outside
            InferenceMode". Isaac Lab's write APIs are only ever exercised from inside
            ``env.step()``, so they assume that context; these methods have to establish it too.
        """
        if not self.cfg.rcg.enabled or self._rcg_pool_ready:
            return

        cfg = self.cfg.rcg
        goal_states = self._load_goal_states()
        num_goals = self._pool_size(goal_states)

        self.rcg_old_starts = goal_states
        print(f"[RCG] Loaded {num_goals} goal state(s) from '{cfg.goal_state_path}'.")

        # paper, Algorithm 1 iteration 1: starts <- SampleNearby([s^g], N_new), then append
        # N_old states sampled from starts_old (which is [s^g] at this point)
        new_starts = self._brownian_expand(goal_states)
        if self._pool_size(new_starts) == 0:
            raise RuntimeError(
                "[RCG] SampleNearby produced no usable start states from the goal states. Every candidate was"
                " rejected as already-solved. Check the recorded goal states, or set"
                " 'rcg.reject_solved_candidates = False'."
            )
        old_starts = self._sample_state_pool(self.rcg_old_starts, cfg.n_old)
        self.rcg_pool = self._concat_state_pools([new_starts, old_starts])

        self._reset_rcg_statistics()
        self._rcg_pool_ready = True

        # every environment is currently sitting in whatever state the Brownian expansion left
        # it in, so put them all into the new distribution before training starts
        self._rcg_reset_all_envs()
        self._rcg_refresh_stage_log()
        print(
            f"[RCG] Stage 0 pool: {self.rcg_pool_size} start states"
            f" ({self._pool_size(new_starts)} new, {self._pool_size(old_starts)} replayed)."
        )

    def rcg_stage_ready(self, policy_steps_in_stage: int) -> bool:
        """Whether the current curriculum stage has had enough policy training.

        Args:
            policy_steps_in_stage: Environment steps collected since the last stage advance.

        Returns:
            ``True`` when the stage should end.
        """
        cfg = self.cfg.rcg
        if not self.rcg_active:
            return False
        if policy_steps_in_stage >= cfg.max_policy_steps_per_stage:
            return True
        if policy_steps_in_stage < cfg.policy_steps_per_stage:
            return False
        if cfg.require_min_attempts and self.rcg_start_attempts.numel() > 0:
            # compared as integer counts: a float mean cannot represent, say, 270/300 exactly,
            # and would fail a '>= 0.9' test on an exactly-90%-covered pool
            covered = int((self.rcg_start_attempts >= cfg.min_attempts_per_start).sum().item())
            needed = math.ceil(cfg.min_attempts_coverage * self.rcg_start_attempts.numel())
            return covered >= needed
        return True

    @torch.inference_mode()
    def advance_rcg_stage(self) -> bool:
        """Select good starts, expand outward from them and install the next training pool.

        Must only be called on a rollout boundary -- after a policy update and before the next
        rollout -- because it teleports every environment.

        Returns:
            ``True`` if a new pool was installed. ``False`` if no good starts exist yet, in
            which case nothing changed and the current stage should continue.
        """
        if not self.rcg_active:
            return False

        cfg = self.cfg.rcg

        # 1. determine the frontier: select() runs over the whole pool (new + replayed), as in
        #    the paper, not just over the newly generated states
        good_starts, num_good = self._select_good_starts()
        if num_good == 0:
            self._rcg_refresh_stage_log()
            print(
                f"[RCG] Stage {self.rcg_stage}: no start state has a success rate in"
                f" ({cfg.r_min}, {cfg.r_max}); keeping the current pool. {self._describe_stall()}"
            )
            return False

        # 2. archive them (paper: starts_old.append(starts))
        self.rcg_old_starts = self._concat_state_pools([self.rcg_old_starts, good_starts])
        self._truncate_old_starts()

        # 3. expand outward from the frontier (paper: SampleNearby)
        new_starts = self._brownian_expand(good_starts)
        if self._pool_size(new_starts) == 0:
            self._rcg_refresh_stage_log()
            print(f"[RCG] Stage {self.rcg_stage}: SampleNearby returned no usable states; keeping the current pool.")
            return False

        # 4. replay old good starts, and 5. install the new distribution
        old_starts = self._sample_state_pool(self.rcg_old_starts, cfg.n_old)
        self.rcg_pool = self._concat_state_pools([new_starts, old_starts])

        # 6. clear the statistics *before* resetting, so that the full reset below cannot
        #    record results against start ids that point into the previous (differently sized)
        #    pool
        self._reset_rcg_statistics()
        self.rcg_stage += 1

        # 7. put every environment into the new distribution
        self._rcg_reset_all_envs()
        self._rcg_refresh_stage_log()
        print(
            f"[RCG] Stage {self.rcg_stage}: {num_good} good starts ->"
            f" {self.rcg_pool_size} start states ({self._pool_size(new_starts)} new,"
            f" {self._pool_size(old_starts)} replayed); archive holds {self._pool_size(self.rcg_old_starts)}."
        )
        return True

    ##
    # Hooks for the task's `_reset_idx`.
    ##

    def _record_rcg_episode_results(self, env_ids: torch.Tensor) -> None:
        """Attribute the outcome of the finishing episodes to the start states that produced them.

        Call at the very start of ``_reset_idx``, before ``super()._reset_idx()``. Uses
        ``self.reset_terminated``, which :class:`~isaaclab.envs.DirectRLEnv` has already
        computed from ``_get_dones`` for this step, so success is read rather than recomputed.
        """
        if not self.rcg_active:
            return

        start_ids = self.rcg_env_start_id[env_ids]
        valid = start_ids >= 0
        if not bool(valid.any()):
            return

        start_ids = start_ids[valid]
        successes = self.reset_terminated[env_ids][valid].float()

        self.rcg_start_attempts.scatter_add_(0, start_ids, torch.ones_like(successes))
        self.rcg_start_successes.scatter_add_(0, start_ids, successes)

    def _rcg_record_goal_states(self, env_ids: torch.Tensor) -> None:
        """Snapshot the states of environments that have just succeeded.

        Call from ``_reset_idx`` alongside :meth:`_record_rcg_episode_results`, i.e. *before*
        ``super()._reset_idx()``. This is the only moment at which a success state is still in
        the simulator: :meth:`~isaaclab.envs.DirectRLEnv.step` resets terminated environments
        before it returns, so the state is gone by the time a script sees the step's output.

        Inactive unless :meth:`start_goal_state_recording` has been called, so it costs nothing
        during training.
        """
        if self._rcg_goal_recording_capacity <= 0:
            return
        if self._rcg_goal_recording_count >= self._rcg_goal_recording_capacity:
            return

        solved = self.reset_terminated[env_ids]
        if not bool(solved.any()):
            return

        # a recorder script drives env.step() inside torch.inference_mode(), so a state captured
        # here would be an inference tensor. Clone it out of inference mode, otherwise the
        # concatenate-and-save that happens after the loop operates on inference tensors.
        with torch.inference_mode(False):
            chunk = {key: value.clone() for key, value in self._rcg_capture_state(env_ids[solved]).items()}
        self._rcg_goal_recording.append(chunk)
        self._rcg_goal_recording_count += self._pool_size(chunk)

    def start_goal_state_recording(self, capacity: int) -> None:
        """Begin collecting success states, for use as the curriculum's goal states ``s^g``."""
        self._rcg_goal_recording = []
        self._rcg_goal_recording_count = 0
        self._rcg_goal_recording_capacity = int(capacity)

    @property
    def recorded_goal_state_count(self) -> int:
        """Number of success states collected so far."""
        return self._rcg_goal_recording_count

    def collect_recorded_goal_states(self) -> StatePool:
        """Return the collected success states, truncated to the requested capacity."""
        pool = self._concat_state_pools(self._rcg_goal_recording)
        size = self._pool_size(pool)
        if size > self._rcg_goal_recording_capacity:
            keep = torch.arange(self._rcg_goal_recording_capacity, dtype=torch.long, device=self.device)
            pool = self._index_state_pool(pool, keep)
        return pool

    def _rcg_reset_from_pool(self, env_ids: torch.Tensor) -> None:
        """Draw start states uniformly from the current pool and restore them (paper: ``rho_i = Unif(starts)``)."""
        pool_ids = torch.randint(0, self.rcg_pool_size, (len(env_ids),), dtype=torch.long, device=self.device)
        self._rcg_restore_state(env_ids, self._index_state_pool(self.rcg_pool, pool_ids))
        self.rcg_env_start_id[env_ids] = pool_ids

    ##
    # Curriculum internals.
    ##

    def _select_good_starts(self) -> tuple[StatePool, int]:
        """Return the pool entries whose success rate lies in ``(r_min, r_max)``."""
        cfg = self.cfg.rcg
        attempted = self.rcg_start_attempts > 0
        rates = torch.zeros_like(self.rcg_start_successes)
        rates[attempted] = self.rcg_start_successes[attempted] / self.rcg_start_attempts[attempted]

        good = attempted & (rates > cfg.r_min) & (rates < cfg.r_max)
        good_ids = good.nonzero(as_tuple=False).squeeze(-1)
        return self._index_state_pool(self.rcg_pool, good_ids), int(good_ids.numel())

    def _describe_stall(self) -> str:
        """Explain *why* no good starts were found, which determines what to do about it."""
        cfg = self.cfg.rcg
        attempts = self.rcg_start_attempts
        attempted = attempts > 0
        rates = torch.where(attempted, self.rcg_start_successes / attempts.clamp(min=1.0), torch.zeros_like(attempts))
        mastered = int((attempted & (rates >= cfg.r_max)).sum().item())
        too_hard = int((attempted & (rates <= cfg.r_min)).sum().item())
        unattempted = int((~attempted).sum().item())

        detail = f"({mastered} mastered, {too_hard} too hard, {unattempted} unattempted)"
        if mastered and not too_hard:
            return (
                f"{detail}. The pool is entirely mastered: the curriculum has nowhere left to expand from under the"
                " paper's criterion. Consider raising 'rcg.r_max', or a longer Brownian horizon so that new starts"
                " land further from the goal."
            )
        if too_hard and not mastered:
            return (
                f"{detail}. Every start is still too hard, so the frontier has outrun the policy. Consider a longer"
                " 'rcg.policy_steps_per_stage', a lower 'rcg.r_min', or a shorter 'rcg.brownian_horizon'."
            )
        if unattempted == attempts.numel():
            return f"{detail}. No episode finished this stage; 'rcg.policy_steps_per_stage' is likely too small."
        return f"{detail}."

    @torch.inference_mode()
    def _brownian_expand(self, seeds: StatePool) -> StatePool:
        """SampleNearby (paper, Procedure 2): random-action rollouts from the given seeds.

        Candidate states come from real rollouts of the simulator, never from noise added
        directly in state space, which is what guarantees they are physically feasible.

        Args:
            seeds: Pool of states to expand outward from.

        Returns:
            A pool of at most ``n_new`` states, subsampled uniformly from the candidates.
        """
        cfg = self.cfg.rcg
        num_seeds = self._pool_size(seeds)
        if num_seeds == 0:
            return self._empty_state_pool()

        all_ids = self._rcg_all_env_ids
        action_dim = self.actions.shape[-1]
        stride = max(1, cfg.candidate_capture_stride)

        chunks: list[StatePool] = []
        num_candidates = 0
        num_rejected = 0

        for _ in range(max(1, cfg.max_expand_batches)):
            # s_0 ~ Unif(starts), independently per environment
            seed_ids = torch.randint(0, num_seeds, (self.num_envs,), dtype=torch.long, device=self.device)
            self._rcg_restore_state(all_ids, self._index_state_pool(seeds, seed_ids))
            self.scene.write_data_to_sim()
            self.sim.forward()

            if cfg.include_seeds_in_candidates:
                chunk, rejected = self._collect_candidates(all_ids)
                num_candidates += self._pool_size(chunk)
                num_rejected += rejected
                chunks.append(chunk)

            for step in range(cfg.brownian_horizon):
                # a_t = eps_t, eps_t ~ N(0, Sigma). Passed unclipped: the environment's own
                # action pre-processing applies the same clipping it applies to policy actions.
                actions = torch.randn((self.num_envs, action_dim), device=self.device) * cfg.brownian_action_std
                self._rcg_physics_step(actions)
                if cfg.brownian_state_noise_std > 0.0:
                    self._rcg_apply_state_noise(all_ids, cfg.brownian_state_noise_std)

                if (step + 1) % stride == 0:
                    chunk, rejected = self._collect_candidates(all_ids)
                    num_candidates += self._pool_size(chunk)
                    num_rejected += rejected
                    chunks.append(chunk)

            # paper: keep generating until at least M candidates exist
            if num_candidates >= cfg.candidate_count:
                break

        candidates = self._concat_state_pools(chunks)
        total = self._pool_size(candidates)
        if total == 0:
            self._rcg_log_expansion(0, num_rejected, 0)
            return self._empty_state_pool()

        # starts_new <- sample(starts, N_new)
        take = min(cfg.n_new, total)
        ids = torch.randperm(total, device=self.device)[:take]
        if take < cfg.n_new:
            print(f"[RCG] SampleNearby produced only {total} candidate(s); using {take} instead of n_new={cfg.n_new}.")

        self._rcg_log_expansion(total, num_rejected, take)
        return self._index_state_pool(candidates, ids)

    def _collect_candidates(self, env_ids: torch.Tensor) -> tuple[StatePool, int]:
        """Capture the current state as candidates, optionally dropping already-solved ones."""
        state = self._rcg_capture_state(env_ids)
        if not self.cfg.rcg.reject_solved_candidates:
            return state, 0
        keep = ~self._rcg_is_solved()
        num_rejected = int((~keep).sum().item())
        if num_rejected == 0:
            return state, 0
        return self._index_state_pool(state, keep.nonzero(as_tuple=False).squeeze(-1)), num_rejected

    def _rcg_physics_step(self, actions: torch.Tensor) -> None:
        """Advance physics only: no rewards, no terminations, no resets, no episode bookkeeping.

        Mirrors the decimated physics portion of :meth:`~isaaclab.envs.DirectRLEnv.step`
        without any of its post-step logic, so the RL algorithm never observes these steps.
        ``episode_length_buf`` and ``common_step_counter`` are deliberately left untouched.
        """
        self._pre_physics_step(actions)
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

    def _rcg_reset_all_envs(self) -> None:
        """Reset every environment into the current pool and refresh the cached observations.

        Follows :meth:`~isaaclab.envs.DirectRLEnv.reset`: reset indices, push the writes into
        the simulator, update kinematics, then recompute observations. The runner must also
        refresh its own cached observations after this, since the states it last saw are gone.
        """
        # allocate a fresh action buffer rather than zeroing in place: the RL runner collects its
        # rollout inside torch.inference_mode(), so `_pre_physics_step` has rebound self.actions
        # to an inference tensor, and an in-place write to one from out here is an error
        self.actions = torch.zeros(self.actions.shape, dtype=self.actions.dtype, device=self.device)
        self._reset_idx(self._rcg_all_env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.obs_buf = self._get_observations()

    def _reset_rcg_statistics(self) -> None:
        """Resize and clear the per-start success statistics for a new pool."""
        size = self.rcg_pool_size
        # Allocated outside inference mode on purpose. These two buffers are updated in place by
        # _record_rcg_episode_results, which runs from _reset_idx in whatever mode the caller of
        # env.step() happens to use -- inside torch.inference_mode() for the RL runner, outside it
        # for a plain scripted loop. An inference tensor cannot be written in place from outside
        # inference mode, so allocating them as normal tensors is what makes both callers work.
        with torch.inference_mode(False):
            self.rcg_start_attempts = torch.zeros(size, dtype=torch.float, device=self.device)
            self.rcg_start_successes = torch.zeros(size, dtype=torch.float, device=self.device)
        # invalidate every in-flight episode's attribution: the ids it holds refer to the
        # previous pool and would index out of bounds
        self.rcg_env_start_id.fill_(-1)

    def _truncate_old_starts(self) -> None:
        """Keep the replay archive within :attr:`~.rcg_cfg.RCGCfg.old_starts_capacity`."""
        capacity = self.cfg.rcg.old_starts_capacity
        size = self._pool_size(self.rcg_old_starts)
        if capacity <= 0 or size <= capacity:
            return
        keep = torch.arange(size - capacity, size, dtype=torch.long, device=self.device)
        self.rcg_old_starts = self._index_state_pool(self.rcg_old_starts, keep)

    def _load_goal_states(self) -> StatePool:
        """Load and validate the recorded goal states (paper: the given ``s^g``)."""
        cfg = self.cfg.rcg
        if not cfg.goal_state_path:
            raise ValueError(
                "[RCG] 'rcg.goal_state_path' is not set. Record goal states with"
                " 'scripts/rcg/record_goal_states.py', then either use a task configuration that sets the path (such"
                " as FrankaCabinetRCGEnvCfg) or pass it explicitly via the Hydra override"
                " 'env.rcg.goal_state_path=<file>'."
            )
        path = os.path.abspath(os.path.expanduser(cfg.goal_state_path))
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[RCG] Goal-state file not found: '{path}'. Record it with"
                " 'scripts/rcg/record_goal_states.py --task <baseline task> --checkpoint <path>'."
            )

        payload = torch.load(path, map_location=self.device)
        pool = payload.get("pool", payload) if isinstance(payload, dict) else None
        if not isinstance(pool, dict) or not pool:
            raise ValueError(f"[RCG] '{path}' does not contain a start-state pool under the key 'pool'.")

        # validate against the schema the task actually captures
        expected = self._empty_state_pool()
        missing = sorted(set(expected) - set(pool))
        extra = sorted(set(pool) - set(expected))
        if missing or extra:
            raise ValueError(
                f"[RCG] Goal-state schema mismatch in '{path}'. Missing keys: {missing}. Unexpected keys: {extra}."
                " Re-record the goal states with the current environment."
            )
        pool = {key: pool[key].to(device=self.device, dtype=expected[key].dtype) for key in expected}
        for key, value in pool.items():
            if value.ndim != 2 or value.shape[1] != expected[key].shape[1]:
                raise ValueError(
                    f"[RCG] Goal-state field '{key}' has shape {tuple(value.shape)}; expected (N,"
                    f" {expected[key].shape[1]}). Re-record the goal states with the current environment."
                )

        size = self._pool_size(pool)
        if size == 0:
            raise ValueError(f"[RCG] '{path}' contains zero goal states.")
        if cfg.num_goal_states is not None and cfg.num_goal_states < size:
            pool = self._index_state_pool(pool, torch.arange(cfg.num_goal_states, dtype=torch.long, device=self.device))
        return pool

    ##
    # Pool helpers.
    ##

    @staticmethod
    def _pool_size(pool: StatePool) -> int:
        """Number of states in a pool."""
        if not pool:
            return 0
        return next(iter(pool.values())).shape[0]

    def _empty_state_pool(self) -> StatePool:
        """A pool with the task's keys and feature dimensions but zero states."""
        return self._rcg_capture_state(self._rcg_all_env_ids[:0])

    @staticmethod
    def _index_state_pool(pool: StatePool, ids: torch.Tensor) -> StatePool:
        """Gather the states at ``ids`` (advanced indexing already copies)."""
        return {key: value[ids] for key, value in pool.items()}

    def _concat_state_pools(self, pools: list[StatePool]) -> StatePool:
        """Concatenate pools along the state dimension, skipping empty ones."""
        non_empty = [pool for pool in pools if self._pool_size(pool) > 0]
        if not non_empty:
            return self._empty_state_pool()
        keys = set(non_empty[0])
        for pool in non_empty[1:]:
            if set(pool) != keys:
                raise ValueError(
                    f"[RCG] Cannot concatenate pools with different keys: {sorted(keys)} vs {sorted(pool)}"
                )
        return {key: torch.cat([pool[key] for pool in non_empty], dim=0) for key in non_empty[0]}

    def _sample_state_pool(self, pool: StatePool, count: int) -> StatePool:
        """Sample ``count`` states from a pool, without replacement where possible."""
        size = self._pool_size(pool)
        if size == 0 or count <= 0:
            return self._empty_state_pool()
        if count <= size:
            ids = torch.randperm(size, device=self.device)[:count]
        else:
            ids = torch.randint(0, size, (count,), dtype=torch.long, device=self.device)
        return self._index_state_pool(pool, ids)

    ##
    # Logging.
    ##

    def _rcg_log_expansion(self, num_candidates: int, num_rejected: int, num_kept: int) -> None:
        """Record what the last SampleNearby call produced. Stage-level, so plain floats."""
        if not self.cfg.rcg.log_diagnostics:
            return
        self._rcg_log.update(
            {
                "rcg/candidates_generated": float(num_candidates),
                "rcg/candidates_rejected_solved": float(num_rejected),
                "rcg/new_starts": float(num_kept),
            }
        )

    def _rcg_refresh_stage_log(self) -> None:
        """Record the stage-level diagnostics. Called only when a stage boundary is crossed."""
        if not self.cfg.rcg.log_diagnostics:
            return
        self._rcg_log.update(
            {
                "rcg/stage": float(self.rcg_stage),
                "rcg/pool_size": float(self.rcg_pool_size),
                "rcg/old_starts_size": float(self._pool_size(self.rcg_old_starts)),
            }
        )
        self._rcg_log.update(self._rcg_pool_diagnostics(self.rcg_pool))

    def _rcg_refresh_step_log(self) -> None:
        """Recompute the diagnostics that evolve within a stage.

        Every value stays a 0-dim device tensor so that reading these once per environment step
        does not force a host/device synchronisation.
        """
        if not self.cfg.rcg.log_diagnostics or self.rcg_start_attempts.numel() == 0:
            return

        attempts = self.rcg_start_attempts
        attempted = attempts > 0
        # success rate averaged over the starts that have been attempted at least once
        num_attempted = attempted.sum().clamp(min=1).float()
        rates = torch.where(attempted, self.rcg_start_successes / attempts.clamp(min=1.0), torch.zeros_like(attempts))

        self._rcg_log.update(
            {
                "rcg/mean_success_rate": rates.sum() / num_attempted,
                "rcg/frac_good_starts": (attempted & (rates > self.cfg.rcg.r_min) & (rates < self.cfg.rcg.r_max))
                .float()
                .mean(),
                "rcg/frac_unattempted": (~attempted).float().mean(),
                "rcg/frac_min_attempts_met": (attempts >= self.cfg.rcg.min_attempts_per_start).float().mean(),
                "rcg/min_attempts": attempts.min(),
                "rcg/mean_attempts": attempts.mean(),
            }
        )
