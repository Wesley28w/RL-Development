# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TSCL-style Window teacher for empirical reset-state curricula.

This module adapts Algorithm 3 from Matiisen et al., "Teacher-Student Curriculum Learning"
(arXiv:1707.00183), to the reset-state setting used by this repository. Each empirical reset
category is treated as one discrete teacher task. The student remains the environment's existing
PPO policy; teacher values are used only to choose reset categories and never enter the reward or
the PPO rollout buffer.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

import torch


class TSCLWindowTeacher:
    """Learning-progress teacher over a fixed set of empirical reset categories.

    Episode outcomes are processed at fixed teacher-update boundaries. A category contributes one
    success-rate measurement only after it has accumulated at least ``min_samples`` completed episodes;
    sparse samples carry forward instead of being treated as failures or discarded. Per-category slopes
    are estimated from a FIFO history using ordinary least squares over PPO-iteration indices, then
    exponentially smoothed as specified by the Window teacher algorithm.

    Sampling remains uniform over currently available state-bank categories until every available
    category has ``min_history`` measurements. Afterwards, a Boltzmann distribution over ``abs(Q)``
    gives attention to both improvement and forgetting. A uniform exploration mixture prevents a valid
    category from becoming permanently unobservable.
    """

    def __init__(
        self,
        num_tasks: int,
        device: str | torch.device,
        *,
        history_size: int,
        min_history: int,
        alpha: float,
        temperature: float,
        min_samples: int,
        update_interval_iterations: int,
        exploration_fraction: float,
    ) -> None:
        if num_tasks < 1:
            raise ValueError(f"num_tasks must be positive, got {num_tasks}.")
        if history_size < 2:
            raise ValueError(f"history_size must be at least 2, got {history_size}.")
        if not 2 <= min_history <= history_size:
            raise ValueError(
                f"min_history must be in [2, history_size], got {min_history} with history_size={history_size}."
            )
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}.")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        if min_samples < 1:
            raise ValueError(f"min_samples must be positive, got {min_samples}.")
        if update_interval_iterations < 1:
            raise ValueError(
                f"update_interval_iterations must be positive, got {update_interval_iterations}."
            )
        if not 0.0 <= exploration_fraction <= 1.0:
            raise ValueError(f"exploration_fraction must be in [0, 1], got {exploration_fraction}.")

        self.num_tasks = num_tasks
        self.device = torch.device(device)
        self.history_size = history_size
        self.min_history = min_history
        self.alpha = alpha
        self.temperature = temperature
        self.min_samples = min_samples
        self.update_interval_iterations = update_interval_iterations
        self.exploration_fraction = exploration_fraction

        self.performance_history = [deque(maxlen=history_size) for _ in range(num_tasks)]
        self.time_history = [deque(maxlen=history_size) for _ in range(num_tasks)]

        self.q_values = torch.zeros(num_tasks, dtype=torch.float32, device=self.device)
        self.raw_slopes = torch.zeros(num_tasks, dtype=torch.float32, device=self.device)
        self.distribution = torch.full((num_tasks,), 1.0 / num_tasks, device=self.device)

        self._completed = torch.zeros(num_tasks, dtype=torch.long, device=self.device)
        self._successful = torch.zeros(num_tasks, dtype=torch.long, device=self.device)
        self._pending_completed = torch.zeros(num_tasks, dtype=torch.long, device=self.device)
        self._pending_successful = torch.zeros(num_tasks, dtype=torch.long, device=self.device)
        self._reset_draws = torch.zeros(num_tasks, dtype=torch.long, device=self.device)

        self.last_interval_completed = torch.zeros_like(self._completed)
        self.last_interval_successful = torch.zeros_like(self._successful)
        self.last_interval_reset_draws = torch.zeros_like(self._reset_draws)
        self.last_measured_success = torch.full((num_tasks,), -1.0, device=self.device)

        self.teacher_update_count = 0
        self.last_update_iteration = 0
        self.last_update_step = 0

    def record_episode_outcomes(self, task_ids: torch.Tensor, successes: torch.Tensor) -> None:
        """Accumulate completed curriculum episodes and their existing task-success outcomes.

        ``task_ids == -1`` denotes ordinary/default-reset episodes and is intentionally ignored: the
        default-reset fraction stays fixed outside the teacher's category distribution.
        """
        task_ids = torch.as_tensor(task_ids, dtype=torch.long, device=self.device).reshape(-1)
        successes = torch.as_tensor(successes, dtype=torch.bool, device=self.device).reshape(-1)
        if task_ids.numel() != successes.numel():
            raise ValueError("task_ids and successes must contain the same number of elements.")

        valid = (task_ids >= 0) & (task_ids < self.num_tasks)
        if not valid.any():
            return

        valid_tasks = task_ids[valid]
        self._completed += torch.bincount(valid_tasks, minlength=self.num_tasks)
        successful_tasks = valid_tasks[successes[valid]]
        if successful_tasks.numel() > 0:
            self._successful += torch.bincount(successful_tasks, minlength=self.num_tasks)

    def record_reset_draws(self, task_ids: torch.Tensor) -> None:
        """Accumulate actual reset-category draws for later teacher diagnostics."""
        task_ids = torch.as_tensor(task_ids, dtype=torch.long, device=self.device).reshape(-1)
        valid = (task_ids >= 0) & (task_ids < self.num_tasks)
        if valid.any():
            self._reset_draws += torch.bincount(task_ids[valid], minlength=self.num_tasks)

    def update_if_due(
        self, ppo_iteration: int, global_env_steps: int, available_tasks: torch.Tensor
    ) -> bool:
        """Update histories, slopes, smoothed values, and probabilities at a fixed PPO interval.

        Args:
            ppo_iteration: Completed PPO rollout iterations, inferred from environment steps and the
                configured (unchanged) rollout horizon.
            global_env_steps: Total vectorized environment transitions consumed by the student.
            available_tasks: Boolean mask identifying categories with at least one empirical state.

        Returns:
            ``True`` if a teacher update boundary was processed, otherwise ``False``.
        """
        ppo_iteration = int(ppo_iteration)
        global_env_steps = int(global_env_steps)
        if ppo_iteration - self.last_update_iteration < self.update_interval_iterations:
            return False

        available_tasks = self._validate_available_tasks(available_tasks)
        self.teacher_update_count += 1
        self.last_update_iteration = ppo_iteration
        self.last_update_step = global_env_steps

        self.last_interval_completed.copy_(self._completed)
        self.last_interval_successful.copy_(self._successful)
        self.last_interval_reset_draws.copy_(self._reset_draws)
        self.last_measured_success.fill_(-1.0)
        self._pending_completed += self._completed
        self._pending_successful += self._successful

        for task in range(self.num_tasks):
            completed = int(self._pending_completed[task].item())
            if completed < self.min_samples:
                continue

            success_rate = float(self._pending_successful[task].item()) / completed
            self.performance_history[task].append(success_rate)
            # PPO iteration is the training-time variable. It stays meaningful when a category misses one
            # or more measurements and is comparable across different numbers of parallel environments.
            self.time_history[task].append(float(ppo_iteration))
            self.last_measured_success[task] = success_rate

            if len(self.performance_history[task]) >= self.min_history:
                slope = self._estimate_slope(self.time_history[task], self.performance_history[task])
                self.raw_slopes[task] = slope
                self.q_values[task] = self.alpha * slope + (1.0 - self.alpha) * self.q_values[task]

            self._pending_completed[task] = 0
            self._pending_successful[task] = 0

        self._completed.zero_()
        self._successful.zero_()
        self._reset_draws.zero_()
        self.distribution.copy_(self.get_distribution(available_tasks))
        return True

    def get_distribution(self, available_tasks: torch.Tensor) -> torch.Tensor:
        """Return the current distribution restricted to populated empirical state banks."""
        available_tasks = self._validate_available_tasks(available_tasks)
        num_available = int(available_tasks.sum().item())
        if num_available == 0:
            return torch.zeros(self.num_tasks, device=self.device)

        history_lengths = torch.tensor(
            [len(history) for history in self.performance_history], dtype=torch.long, device=self.device
        )
        bootstrap = bool((history_lengths[available_tasks] < self.min_history).any().item())
        if bootstrap:
            return available_tasks.float() / num_available

        logits = self.q_values.abs() / self.temperature
        logits = torch.where(available_tasks, logits, torch.full_like(logits, -torch.inf))
        distribution = torch.softmax(logits, dim=0)

        if self.exploration_fraction > 0.0:
            uniform = available_tasks.float() / num_available
            distribution = (1.0 - self.exploration_fraction) * distribution
            distribution = distribution + self.exploration_fraction * uniform

        return distribution / distribution.sum()

    def metrics(
        self, ppo_iteration: int, global_env_steps: int, available_tasks: torch.Tensor
    ) -> Mapping[str, float]:
        """Return scalar diagnostics sufficient to reconstruct teacher behavior."""
        available_tasks = self._validate_available_tasks(available_tasks)
        distribution = self.get_distribution(available_tasks)
        history_lengths = [len(history) for history in self.performance_history]
        available_lengths = [history_lengths[i] for i in range(self.num_tasks) if bool(available_tasks[i].item())]
        bootstrap = not available_lengths or any(length < self.min_history for length in available_lengths)
        total_draws = int(self.last_interval_reset_draws.sum().item())

        metrics: dict[str, float] = {
            "tscl/global_env_steps": float(global_env_steps),
            "tscl/ppo_iteration": float(ppo_iteration),
            "tscl/teacher_update": float(self.teacher_update_count),
            "tscl/last_update_ppo_iteration": float(self.last_update_iteration),
            "tscl/last_update_global_env_steps": float(self.last_update_step),
            "tscl/bootstrap": float(bootstrap),
            "tscl/temperature": float(self.temperature),
            "tscl/alpha": float(self.alpha),
            "tscl/history_size": float(self.history_size),
            "tscl/min_history": float(self.min_history),
            "tscl/min_samples": float(self.min_samples),
            "tscl/update_interval_iterations": float(self.update_interval_iterations),
            "tscl/exploration_fraction": float(self.exploration_fraction),
        }
        for task in range(self.num_tasks):
            reset_count = int(self.last_interval_reset_draws[task].item())
            metrics[f"tscl/task_{task + 1}/available"] = float(available_tasks[task].item())
            metrics[f"tscl/task_{task + 1}/completed"] = float(self.last_interval_completed[task].item())
            metrics[f"tscl/task_{task + 1}/successful"] = float(self.last_interval_successful[task].item())
            metrics[f"tscl/task_{task + 1}/pending_completed"] = float(self._pending_completed[task].item())
            metrics[f"tscl/task_{task + 1}/success_rate"] = float(self.last_measured_success[task].item())
            metrics[f"tscl/task_{task + 1}/history_length"] = float(history_lengths[task])
            metrics[f"tscl/task_{task + 1}/slope"] = float(self.raw_slopes[task].item())
            metrics[f"tscl/task_{task + 1}/q"] = float(self.q_values[task].item())
            metrics[f"tscl/task_{task + 1}/priority"] = float(self.q_values[task].abs().item())
            metrics[f"tscl/task_{task + 1}/probability"] = float(distribution[task].item())
            metrics[f"tscl/task_{task + 1}/reset_count"] = float(reset_count)
            metrics[f"tscl/task_{task + 1}/reset_fraction"] = reset_count / total_draws if total_draws else 0.0
        return metrics

    def _validate_available_tasks(self, available_tasks: torch.Tensor) -> torch.Tensor:
        available_tasks = torch.as_tensor(available_tasks, dtype=torch.bool, device=self.device).reshape(-1)
        if available_tasks.numel() != self.num_tasks:
            raise ValueError(
                f"available_tasks must have {self.num_tasks} elements, got {available_tasks.numel()}."
            )
        return available_tasks

    @staticmethod
    def _estimate_slope(times: deque[float], scores: deque[float]) -> float:
        x = torch.tensor(list(times), dtype=torch.float64)
        y = torch.tensor(list(scores), dtype=torch.float64)
        x_centered = x - x.mean()
        denominator = x_centered.square().sum()
        if denominator <= 0.0:
            return 0.0
        return float((x_centered * (y - y.mean())).sum() / denominator)
