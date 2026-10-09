# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim before importing the Cabinet environment."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.franka_cabinet.franka_cabinet_env import FrankaCabinetEnv
from isaaclab_tasks.utils import load_cfg_from_registry


def test_uniform_distribution_uses_only_populated_banks():
    available = torch.tensor([True, False, True, True])

    distribution = FrankaCabinetEnv._uniform_available_distribution(available)

    torch.testing.assert_close(distribution, torch.tensor([1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0]))


def test_uniform_distribution_is_empty_before_any_bank_is_populated():
    available = torch.zeros(4, dtype=torch.bool)

    distribution = FrankaCabinetEnv._uniform_available_distribution(available)

    torch.testing.assert_close(distribution, torch.zeros(4))


def test_uniform_task_changes_only_curriculum_selection_config():
    base_spec = gym.spec("Isaac-Franka-Cabinet-Direct-v0")
    uniform_spec = gym.spec("Isaac-Franka-Cabinet-Uniform-Direct-v0")

    assert base_spec.kwargs["rsl_rl_cfg_entry_point"] == uniform_spec.kwargs["rsl_rl_cfg_entry_point"]
    assert uniform_spec.kwargs["env_cfg_entry_point"].endswith(":FrankaCabinetUniformRandomEnvCfg")

    base_cfg = load_cfg_from_registry("Isaac-Franka-Cabinet-Direct-v0", "env_cfg_entry_point").to_dict()
    uniform_cfg = load_cfg_from_registry("Isaac-Franka-Cabinet-Uniform-Direct-v0", "env_cfg_entry_point").to_dict()

    assert uniform_cfg.pop("reset_state_curriculum_enabled") is True
    assert uniform_cfg.pop("reset_curriculum_mode") == "uniform_random"
    base_cfg.pop("reset_state_curriculum_enabled")
    base_cfg.pop("reset_curriculum_mode")
    assert uniform_cfg == base_cfg
