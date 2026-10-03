# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the TSCL-style reset-category Window teacher."""

import pytest
import torch

from isaaclab_tasks.utils.tscl_window import TSCLWindowTeacher


def _make_teacher(**kwargs) -> TSCLWindowTeacher:
    defaults = {
        "num_tasks": 3,
        "device": "cpu",
        "history_size": 4,
        "min_history": 3,
        "alpha": 0.5,
        "temperature": 0.1,
        "min_samples": 2,
        "update_interval_iterations": 10,
        "exploration_fraction": 0.2,
    }
    defaults.update(kwargs)
    return TSCLWindowTeacher(**defaults)


def _record_rate(teacher: TSCLWindowTeacher, task: int, successes: int, total: int) -> None:
    task_ids = torch.full((total,), task)
    outcomes = torch.arange(total) < successes
    teacher.record_episode_outcomes(task_ids, outcomes)


def test_ignores_default_resets_and_missing_measurements() -> None:
    teacher = _make_teacher()
    teacher.record_episode_outcomes(torch.tensor([-1, -1]), torch.tensor([True, False]))
    assert teacher.update_if_due(10, 1_000, torch.tensor([True, True, True]))

    assert [len(history) for history in teacher.performance_history] == [0, 0, 0]
    assert torch.equal(teacher.q_values, torch.zeros(3))
    assert torch.allclose(teacher.get_distribution(torch.tensor([True, True, True])), torch.full((3,), 1 / 3))


def test_updates_only_after_configured_ppo_iteration_interval() -> None:
    teacher = _make_teacher(update_interval_iterations=20)
    available = torch.tensor([True, True, True])

    assert not teacher.update_if_due(19, 1_900, available)
    assert teacher.update_if_due(20, 2_000, available)
    assert not teacher.update_if_due(39, 3_900, available)
    assert teacher.update_if_due(40, 4_000, available)
    assert teacher.teacher_update_count == 2


@pytest.mark.parametrize(
    ("total_iterations", "update_interval"),
    [(3_000, 30), (2_500, 25), (200, 2)],
)
def test_normalized_benchmark_schedule_has_100_updates(total_iterations: int, update_interval: int) -> None:
    teacher = _make_teacher(update_interval_iterations=update_interval)
    available = torch.tensor([True, True, True])

    for iteration in range(1, total_iterations + 1):
        teacher.update_if_due(iteration, iteration * 100, available)

    assert teacher.teacher_update_count == 100


def test_carries_sparse_samples_to_a_later_teacher_interval() -> None:
    teacher = _make_teacher(min_samples=3)
    available = torch.tensor([True, False, False])

    _record_rate(teacher, task=0, successes=1, total=2)
    teacher.update_if_due(10, 1_000, available)
    assert len(teacher.performance_history[0]) == 0
    assert teacher.metrics(10, 1_000, available)["tscl/task_1/pending_completed"] == 2

    _record_rate(teacher, task=0, successes=1, total=1)
    teacher.update_if_due(20, 2_000, available)
    assert list(teacher.performance_history[0]) == pytest.approx([2 / 3])
    assert teacher.metrics(20, 2_000, available)["tscl/task_1/pending_completed"] == 0


def test_bootstraps_uniformly_until_every_available_task_is_ready() -> None:
    teacher = _make_teacher()
    available = torch.tensor([True, True, False])

    for update in range(1, 4):
        _record_rate(teacher, task=0, successes=update, total=4)
        _record_rate(teacher, task=1, successes=4 - update, total=4)
        assert teacher.update_if_due(update * 10, update * 1_000, available)

    distribution = teacher.get_distribution(available)
    # Equal-magnitude improvement and forgetting receive equal attention once bootstrap ends.
    assert distribution[0] == pytest.approx(distribution[1])
    assert distribution[1] >= teacher.exploration_fraction / 2
    assert distribution[2] == 0.0
    assert torch.isclose(distribution.sum(), torch.tensor(1.0))

    newly_available = torch.tensor([True, True, True])
    assert torch.allclose(teacher.get_distribution(newly_available), torch.full((3,), 1 / 3))


def test_absolute_smoothed_progress_prioritizes_improvement_and_forgetting() -> None:
    teacher = _make_teacher(alpha=0.5, temperature=0.01)
    available = torch.tensor([True, True, True])
    rates = ([0, 1, 2], [4, 3, 2], [2, 2, 2])

    for update in range(3):
        for task, task_rates in enumerate(rates):
            _record_rate(teacher, task, successes=task_rates[update], total=4)
        teacher.update_if_due((update + 1) * 10, (update + 1) * 1_000, available)

    assert teacher.raw_slopes[0] > 0
    assert teacher.raw_slopes[1] < 0
    assert teacher.raw_slopes[2] == 0
    assert torch.isclose(teacher.q_values[0].abs(), teacher.q_values[1].abs())
    assert teacher.distribution[0] > teacher.distribution[2]
    assert teacher.distribution[1] > teacher.distribution[2]


def test_boltzmann_policy_has_exact_uniform_exploration_mixture() -> None:
    teacher = _make_teacher(temperature=0.0004, exploration_fraction=0.2)
    available = torch.tensor([True, True, True])
    for history in teacher.performance_history:
        history.extend([0.0, 0.0, 0.0])
    teacher.q_values.copy_(torch.tensor([1.0, 0.0, 0.0]))

    distribution = teacher.get_distribution(available)
    assert distribution.tolist() == pytest.approx([13 / 15, 1 / 15, 1 / 15])


def test_logs_interval_outcomes_and_actual_reset_draws() -> None:
    teacher = _make_teacher()
    teacher.record_reset_draws(torch.tensor([0, 0, 1, -1]))
    _record_rate(teacher, task=0, successes=1, total=2)
    teacher.update_if_due(10, 1_000, torch.tensor([True, True, False]))

    metrics = teacher.metrics(10, 1_000, torch.tensor([True, True, False]))
    assert metrics["tscl/last_update_ppo_iteration"] == 10
    assert metrics["tscl/last_update_global_env_steps"] == 1_000
    assert metrics["tscl/task_1/completed"] == 2
    assert metrics["tscl/task_1/successful"] == 1
    assert metrics["tscl/task_1/reset_count"] == 2
    assert metrics["tscl/task_1/reset_fraction"] == pytest.approx(2 / 3)
    assert metrics["tscl/task_2/reset_count"] == 1
    assert metrics["tscl/task_2/reset_fraction"] == pytest.approx(1 / 3)
    assert metrics["tscl/task_3/available"] == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"history_size": 1},
        {"min_history": 1},
        {"alpha": 0.0},
        {"temperature": 0.0},
        {"min_samples": 0},
        {"update_interval_iterations": 0},
        {"exploration_fraction": 1.1},
    ],
)
def test_rejects_invalid_hyperparameters(overrides) -> None:
    with pytest.raises(ValueError):
        _make_teacher(**overrides)
