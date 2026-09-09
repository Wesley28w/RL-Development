# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vectorized Reverse Curriculum Generation (Florensa et al., CoRL 2017).

Shared across Direct and manager-based Isaac Lab tasks: this class only ever deals in
flat per-env state tensors handed to it by the caller. It has no notion of articulations,
scenes, or reset APIs -- each task is responsible for capturing/restoring its own state
and for calling the hooks below at the right point in its step/reset loop.

Expansion methodology note: candidate generation (the Brownian-motion rollout that expands
the frontier outward) is deliberately *not* implemented in this class. It has to run as a
phase that is invisible to on-policy rollout collection -- every transition the RL library
trains on must come from the current policy -- which means it needs direct access to the
task's own physics-stepping primitives (see e.g. `franka_cabinet_env.py`'s
`_run_rcg_expansion_phase`, which snapshots the full scene, drives a cohort of envs through
`brownian_horizon` raw physics ticks with noise actions, collects candidates, and restores
every env to its pre-expansion state before real rollout collection resumes). This class only
ever receives the resulting candidate states, via `rebuild_pool`.
"""

from __future__ import annotations

import torch


class ReverseCurriculum:
    """Flat-pool reverse curriculum: uniform sampling, success-rate-gated frontier selection."""

    def __init__(
        self,
        state_dim: int,
        pool_size: int,
        num_envs: int,
        device: torch.device | str,
        r_min: float = 0.15,
        r_max: float = 0.85,
        n_new: int | None = None,
        n_old: int | None = None,
        min_episodes_per_state: int = 5,
        replay_history_size: int = 2000,
    ):
        self.state_dim = state_dim
        self.pool_size = pool_size
        self.device = device
        self.r_min = r_min
        self.r_max = r_max
        # default new:old ratio of 2:1 matches the reference implementation's own arm-manipulation
        # experiments (curriculum/experiments/starts/arm3d/{arm3d_key,arm3d_disc}_brownian.py both
        # use num_new_starts=600, num_old_starts=300).
        self.n_new = n_new if n_new is not None else max(int(round(2 * pool_size / 3)), 1)
        self.n_old = n_old if n_old is not None else max(pool_size - self.n_new, 1)
        self.min_episodes = min_episodes_per_state
        self.replay_history_size = replay_history_size

        self.generation = 0
        self.active_states = torch.zeros(pool_size, state_dim, device=device)
        self.active_num_episodes = torch.zeros(pool_size, dtype=torch.long, device=device)
        self.active_num_successes = torch.zeros(pool_size, dtype=torch.long, device=device)

        self.replay_states = torch.zeros(replay_history_size, state_dim, device=device)
        self.replay_count = 0
        self._replay_write_idx = 0

        # per-env bookkeeping: which pool row (and which generation of the pool) an
        # episode started from, so a finished episode's outcome can be attributed back.
        self.env_start_row = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.env_start_generation = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    # -- seeding / pool bookkeeping ------------------------------------------------

    def seed(self, goal_states: torch.Tensor) -> None:
        """Initialize the pool from one or more known-good goal states."""
        n_goals = goal_states.shape[0]
        reps = (self.pool_size + n_goals - 1) // n_goals
        self.active_states[:] = goal_states.repeat(reps, 1)[: self.pool_size]
        self.active_num_episodes.zero_()
        self.active_num_successes.zero_()
        self._push_replay(goal_states)

    def _push_replay(self, states: torch.Tensor) -> None:
        n = states.shape[0]
        if n == 0:
            return
        if n >= self.replay_history_size:
            self.replay_states[:] = states[-self.replay_history_size :]
            self._replay_write_idx = 0
            self.replay_count = self.replay_history_size
            return
        idx = (torch.arange(n, device=self.device) + self._replay_write_idx) % self.replay_history_size
        self.replay_states[idx] = states
        self._replay_write_idx = (self._replay_write_idx + n) % self.replay_history_size
        self.replay_count = min(self.replay_count + n, self.replay_history_size)

    # -- reset-time sampling ---------------------------------------------------------

    def sample_for_resets(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Uniformly sample pool rows for the given envs and record start (row, generation)."""
        rows = torch.randint(0, self.pool_size, (len(env_ids),), device=self.device)
        self.env_start_row[env_ids] = rows
        self.env_start_generation[env_ids] = self.generation
        return self.active_states[rows]

    def record_episode_results(self, env_ids: torch.Tensor, successes: torch.Tensor) -> None:
        """Attribute finished episodes back to the pool row they started from.

        Episodes whose start row belongs to a pool generation the curriculum has since
        rebuilt past are dropped: that row index no longer means the same state, so
        folding the result in would corrupt the new generation's statistics.
        """
        stale = self.env_start_generation[env_ids] != self.generation
        rows = self.env_start_row[env_ids]
        valid = (~stale) & (rows >= 0)
        if valid.any():
            rows = rows[valid]
            succ = successes[valid].long()
            self.active_num_episodes.scatter_add_(0, rows, torch.ones_like(rows))
            self.active_num_successes.scatter_add_(0, rows, succ)

    # -- generation update -------------------------------------------------------------

    def select_good_starts(self) -> torch.Tensor:
        """Pool rows with enough episodes and an intermediate (R_min, R_max) success rate.

        Empty-frontier fallback (guide sec. 11): if nothing lands in the intermediate band
        this generation, keep whichever tested rows came closest to it -- rather than
        collapsing back to only the seed/goal state, which would silently turn this into
        a much easier "always reset to the goal" algorithm.
        """
        enough = self.active_num_episodes >= self.min_episodes
        if not enough.any():
            return self.active_states[:1]

        rate = torch.zeros(self.pool_size, device=self.device)
        rate[enough] = self.active_num_successes[enough].float() / self.active_num_episodes[enough].float()

        good = enough & (rate >= self.r_min) & (rate <= self.r_max)
        if good.any():
            return self.active_states[good]

        dist_to_band = torch.where(rate < self.r_min, self.r_min - rate, rate - self.r_max)
        dist_to_band = torch.where(enough, dist_to_band, torch.full_like(dist_to_band, float("inf")))
        k = min(8, int(enough.sum().item()))
        keep = torch.topk(-dist_to_band, k=k).indices
        return self.active_states[keep]

    def rebuild_pool(self, candidates: torch.Tensor) -> None:
        """Rebuild the active pool from this generation's expansion candidates.

        `candidates` is whatever the caller's own expansion phase collected since the last call.
        """
        good = self.select_good_starts()
        self._push_replay(good)

        n_candidates = candidates.shape[0]
        if n_candidates >= self.n_new:
            perm = torch.randperm(n_candidates, device=self.device)[: self.n_new]
            new_part = candidates[perm]
        elif n_candidates > 0:
            new_part = candidates
        else:
            new_part = good  # nothing new generated this round -- hold the frontier steady

        if self.replay_count > 0:
            n_old = min(self.n_old, self.replay_count)
            idx = torch.randint(0, self.replay_count, (n_old,), device=self.device)
            old_part = self.replay_states[idx]
        else:
            old_part = good

        pool = torch.cat([new_part, old_part, good], dim=0)
        if pool.shape[0] >= self.pool_size:
            pool = pool[: self.pool_size]
        else:
            reps = (self.pool_size + pool.shape[0] - 1) // pool.shape[0]
            pool = pool.repeat(reps, 1)[: self.pool_size]

        self.active_states[:] = pool
        self.active_num_episodes.zero_()
        self.active_num_successes.zero_()
        self.generation += 1

    # -- diagnostics --------------------------------------------------------------------

    def stats(self) -> dict[str, float]:
        enough = self.active_num_episodes >= self.min_episodes
        rate = torch.zeros(self.pool_size, device=self.device)
        rate[enough] = self.active_num_successes[enough].float() / self.active_num_episodes[enough].float()
        n_enough = int(enough.sum().item())
        return {
            "rcg/generation": float(self.generation),
            "rcg/pool_mean_success_rate": rate[enough].mean().item() if n_enough else 0.0,
            "rcg/pool_frac_too_hard": ((rate < self.r_min) & enough).float().mean().item() if n_enough else 0.0,
            "rcg/pool_frac_intermediate": (
                ((rate >= self.r_min) & (rate <= self.r_max) & enough).float().mean().item() if n_enough else 0.0
            ),
            "rcg/pool_frac_mastered": ((rate > self.r_max) & enough).float().mean().item() if n_enough else 0.0,
            "rcg/pool_frac_untested": (~enough).float().mean().item(),
            "rcg/replay_history_size": float(self.replay_count),
        }
