# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for the corrected Factory ACES distribution."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

from types import SimpleNamespace

import torch

from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskNutThreadCfg


def _make_factory_scheduler() -> FactoryEnv:
    env = object.__new__(FactoryEnv)
    env._is_closed = True
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env.cfg = SimpleNamespace(
        success_rate_alpha=1.0,
        softmax_temperature=0.25,
        greedy_margin_lo=0.5,
        greedy_margin_hi=1.5,
        min_subtask_prob=0.05,
    )
    env.is_curriculum_episode = torch.zeros(4, dtype=torch.bool)
    env.progression = torch.zeros((4, 3, 24))
    env.progression[:1, 0, 0] = 1.0
    env.progression[:2, 1, 0] = 1.0
    env.progression[:3, 2, 0] = 1.0
    env.success_rate = torch.zeros(3)
    env.distribution = torch.full((3,), 1.0 / 3.0)
    env.extras = {"log": {}}
    return env


def test_factory_aces_uses_temperature_margin_band_and_probability_floor():
    env = _make_factory_scheduler()

    env._update_distribution()

    difficulty = 1.0 - env.success_rate
    previous = torch.cat([torch.zeros(1), difficulty[:-1]])
    gaps = difficulty - previous
    gaps = gaps - gaps.min() + 1e-8
    confidence = gaps / gaps.sum().clamp(min=1e-8)
    soft = (confidence / env.cfg.softmax_temperature).softmax(dim=0)

    top2 = torch.topk(confidence, k=2)
    hard = torch.zeros_like(confidence)
    hard[top2.indices[0]] = 1.0
    margin_norm = (top2.values[0] - top2.values[1]) / (1.0 / (confidence.numel() - 1))
    blend = torch.clamp(
        (margin_norm - env.cfg.greedy_margin_lo)
        / (env.cfg.greedy_margin_hi - env.cfg.greedy_margin_lo),
        0.0,
        1.0,
    )
    expected = (1.0 - blend) * soft + blend * hard
    expected = (1.0 - env.cfg.min_subtask_prob * confidence.numel()) * expected + env.cfg.min_subtask_prob
    expected /= expected.sum()

    torch.testing.assert_close(env.distribution, expected)
    assert bool((env.distribution >= env.cfg.min_subtask_prob).all())
    assert env.extras["log"]["curriculum/margin_norm"] == margin_norm.item()


def test_factory_nut_config_exposes_corrected_aces_parameters():
    cfg = FactoryTaskNutThreadCfg()

    assert cfg.softmax_temperature == 0.25
    assert cfg.greedy_margin_lo == 0.5
    assert cfg.greedy_margin_hi == 1.5
    assert cfg.min_subtask_prob == 0.05
    assert not hasattr(cfg, "prob_exp")
    assert not hasattr(cfg, "greedy_margin")
