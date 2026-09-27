# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Reverse Curriculum Generation (RCG).

Reference:
    C. Florensa, D. Held, M. Wulfmeier, M. Zhang, P. Abbeel,
    "Reverse Curriculum Generation for Reinforcement Learning", CoRL 2017.
    https://arxiv.org/abs/1707.05300

The defaults below are the values reported in the paper (Section 5 / Appendix):
``R_min = 0.1``, ``R_max = 0.9``, ``N_new = 200``, ``N_old = 100``, ``M = 10_000``,
``T_B = 50`` and ``Sigma = I``.

.. note::
    This module intentionally has no dependency on any RL library. It is imported by the
    environment, which must stay usable without ``rsl_rl`` installed.
"""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class RCGCfg:
    """Reverse Curriculum Generation settings.

    The curriculum maintains a pool of *start states* instead of a single fixed reset
    distribution. Each RCG stage trains the policy on ``Unif(pool)`` for a fixed budget,
    measures the per-start success probability, keeps the starts whose success probability
    falls in ``(r_min, r_max)`` (the "good starts"), and expands outward from them with short
    random-action ("Brownian") rollouts.
    """

    enabled: bool = False
    """Whether the reverse curriculum is active.

    Defaults to ``False`` so that the vanilla task configuration -- and therefore the PPO
    baseline of the benchmark -- is completely unaffected by this module.
    """

    ##
    # Good-start criterion (paper: S_0^i = {s_0 : R_min < R(pi_i, s_0) < R_max}).
    ##

    r_min: float = 0.1
    """Lower bound on a start state's success probability for it to count as a good start."""

    r_max: float = 0.9
    """Upper bound on a start state's success probability for it to count as a good start."""

    ##
    # Pool composition (paper: N_new, N_old).
    ##

    n_new: int = 200
    """Number of newly generated start states per stage (paper: ``N_new = 200``)."""

    n_old: int = 100
    """Number of replayed start states per stage (paper: ``N_old = 100``)."""

    old_starts_capacity: int = 100_000
    """Capacity of the good-start replay archive.

    The paper's ``starts_old`` list is unbounded. We bound it and evict the oldest entries so
    that a long run cannot grow the archive without limit.
    """

    ##
    # SampleNearby (paper: Procedure 2).
    ##

    candidate_count: int = 10_000
    """Minimum number of candidate states to generate before subsampling (paper: ``M``).

    Brownian batches are generated until at least this many candidates exist. With many
    parallel environments a single batch already exceeds ``M``.
    """

    brownian_horizon: int = 50
    """Length of each Brownian rollout in environment steps (paper: ``T_B = 50``)."""

    brownian_action_std: float = 1.0
    """Standard deviation of the Brownian action noise (paper: ``Sigma = I``).

    Actions are drawn as ``a_t ~ N(0, brownian_action_std^2 I)`` and passed to the
    environment's own action pre-processing, which applies the usual clipping.
    """

    candidate_capture_stride: int = 1
    """Capture a candidate every ``n`` steps of the Brownian rollout.

    ``1`` (the default) treats every visited state as a candidate, which is what the paper
    does. Increase it only to reduce memory for tasks with very large state vectors.
    """

    max_expand_batches: int = 64
    """Safety cap on the number of Brownian batches per stage, so ``candidate_count`` can
    never turn into an unbounded loop."""

    include_seeds_in_candidates: bool = True
    """Whether the seed states themselves are candidates.

    Procedure 2 appends visited states to the same list that seeds are drawn from and that is
    finally subsampled, so the seeds are candidates in the paper as well.
    """

    reject_solved_candidates: bool = True
    """Drop candidate states that already satisfy the task's success condition.

    Documented deviation from the paper: such a state is terminal, so an episode started from
    it ends immediately with guaranteed success. It can only ever be filtered out as
    "mastered" while occupying a slot in the pool. Equivalent to restricting ``rho_i`` to the
    complement of the goal set.
    """

    brownian_state_noise_std: float = 0.0
    """Standard deviation of optional Brownian noise injected directly into the task's
    under-actuated state (for Franka Cabinet: the cabinet joints).

    ``0.0`` (the default) is the paper's behaviour: the random walk acts through the action
    space only. Raise it only if the diagnostics show that action-space noise does not move
    the task's progress variable away from the goal -- and then say so in the write-up.
    """

    ##
    # Stage scheduling.
    ##

    policy_steps_per_stage: int = 1_200_000
    """Number of environment steps of policy training per curriculum stage.

    Sized so that every start state is attempted enough times for its success probability to
    be estimable: ``(n_new + n_old) * attempts * episode_length``, i.e. roughly
    ``300 * 8 * 500`` for Franka Cabinet.
    """

    min_attempts_per_start: int = 5
    """Number of episodes a start state must have been attempted from for its success
    probability to count as estimable."""

    min_attempts_coverage: float = 0.9
    """Fraction of the pool that must reach :attr:`min_attempts_per_start` before a stage ends.

    Deliberately not ``1.0``. Episodes are assigned to start states at random, so the *minimum*
    attempt count over a 300-state pool lags the mean badly: at a mean of 10 attempts per start
    roughly 3% of starts still sit below 5, which would push every stage to
    :attr:`max_policy_steps_per_stage`. Requiring 90% coverage instead means a stage ends once
    the success estimates are broadly reliable, and the few under-sampled starts are simply not
    selectable as good starts that round.
    """

    require_min_attempts: bool = True
    """Whether the attempt-coverage criterion is enforced in addition to the step budget."""

    max_policy_steps_per_stage: int = 4_000_000
    """Hard cap on a stage's length, so :attr:`require_min_attempts` cannot stall training
    forever on a start state that is never sampled or never terminates."""

    ##
    # State definition.
    ##

    zero_velocities_on_restore: bool = False
    """Zero all velocities when restoring a start state.

    ``False`` (the default) restores the genuine visited state, which is what makes restore
    the exact inverse of capture. ``True`` is available as an ablation.
    """

    reset_dof_targets: bool = True
    """Reset the environment's joint-position-target buffer on every reset.

    Upstream ``FrankaCabinetEnv._reset_idx`` calls ``set_joint_position_target`` but never
    updates ``self.robot_dof_targets``, which is what ``_pre_physics_step`` integrates from,
    so the target buffer leaks across episodes. RCG requires this buffer to be part of the
    restored state; the fix is applied to the normal reset path as well so that both arms of
    the benchmark share identical reset semantics. Set to ``False`` to reproduce the upstream
    behaviour as an ablation.
    """

    ##
    # Goal states (paper: s^g).
    ##

    goal_state_path: str = ""
    """Path to the ``.pt`` file produced by ``scripts/rcg/record_goal_states.py``."""

    num_goal_states: int | None = None
    """Number of goal states to keep from the recorded file.

    ``None`` keeps all of them. The paper assumes a single goal state ``s^g``; set this to
    ``1`` for strict fidelity.
    """

    ##
    # Reward variant.
    ##

    sparse_reward: bool = False
    """Replace the task's dense reward with the paper's sparse indicator ``r = 1{s in S^g}``.

    ``False`` keeps Isaac Lab's dense reward so that RCG and the PPO baseline optimise the
    same objective. The good-start criterion always uses binary task success regardless of
    this flag -- never the dense return.
    """

    ##
    # Logging.
    ##

    log_diagnostics: bool = True
    """Whether to publish ``rcg/*`` diagnostics through ``extras["log"]``."""

    eval_env_fraction: float = 0.0
    """Fraction of environments held out of the curriculum and always reset from ``rho_0``.

    Without this, *nothing* logged during RCG training is comparable to a baseline or to the
    reset-pose curriculum: RCG teleports every environment, so every episode starts from a
    curriculum state and the success rate measures the curriculum, not the task. The held-out
    environments give a live ``dones/eval_*`` curve on the task's own start distribution.

    Defaults to ``0.0``, which is the paper's behaviour. ``0.0625`` (a sixteenth) is a
    reasonable benchmark setting.

    .. warning::
        Documented deviation from the paper when non-zero. The held-out environments are
        excluded from the curriculum's start-state statistics, but their transitions still enter
        the PPO rollout, so a small fraction of training data comes from ``rho_0`` rather than
        from ``rho_i``. Use :mod:`scripts.rcg.evaluate` for numbers that must be free of that
        caveat.
    """
