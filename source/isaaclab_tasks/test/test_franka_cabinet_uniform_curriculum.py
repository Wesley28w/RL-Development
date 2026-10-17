# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim before importing the Cabinet environment."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

from types import SimpleNamespace

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


def test_absolute_difficulty_is_linear_failure_rate_distribution():
    success_rates = torch.tensor([0.10, 0.40, 0.80, 0.90])
    available = torch.ones(4, dtype=torch.bool)

    distribution = FrankaCabinetEnv._absolute_difficulty_distribution(success_rates, available)

    expected = torch.tensor([0.90, 0.60, 0.20, 0.10])
    expected /= expected.sum()
    torch.testing.assert_close(distribution, expected)


def test_absolute_difficulty_masks_empty_banks_and_falls_back_at_full_success():
    available = torch.tensor([True, False, True, True])

    distribution = FrankaCabinetEnv._absolute_difficulty_distribution(
        torch.tensor([0.25, 0.0, 0.75, 1.0]), available
    )
    torch.testing.assert_close(distribution, torch.tensor([0.75, 0.0, 0.25, 0.0]))

    mastered_distribution = FrankaCabinetEnv._absolute_difficulty_distribution(torch.ones(4), available)
    torch.testing.assert_close(mastered_distribution, torch.tensor([1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0]))


def _make_aces_scheduler_test_env(mode: str) -> FrankaCabinetEnv:
    env = object.__new__(FrankaCabinetEnv)
    env._is_closed = True
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env.cfg = SimpleNamespace(
        success_rate_alpha=1.0,
        softmax_temperature=0.25,
        greedy_margin_lo=0.5,
        greedy_margin_hi=1.5,
        min_subtask_prob=0.05,
        reset_curriculum_mode=mode,
    )
    env.is_curriculum_episode = torch.zeros(4, dtype=torch.bool)
    env.progression = torch.zeros((4, 4, 14))
    env.progression[:1, 0, 0] = 1.0
    env.progression[:2, 1, 0] = 1.0
    env.progression[:3, 2, 0] = 1.0
    env.progression[:, 3, 0] = 1.0
    env.success_rate = torch.zeros(4)
    env.extras = {"log": {}}
    return env


def _scheduler_confidence_and_hard(env: FrankaCabinetEnv) -> tuple[torch.Tensor, torch.Tensor]:
    difficulty = 1.0 - env.success_rate
    previous = torch.cat([torch.zeros(1), difficulty[:-1]])
    gaps = difficulty - previous
    gaps = gaps - gaps.min() + 1e-8
    confidence = gaps / gaps.sum().clamp(min=1e-8)
    hard = torch.zeros_like(confidence)
    hard[confidence.argmax()] = 1.0
    return confidence, hard


def test_fixed_50_50_mode_uses_equal_soft_and_hard_weights():
    env = _make_aces_scheduler_test_env("fixed_50_50_blend")

    env._update_distribution()

    confidence, hard = _scheduler_confidence_and_hard(env)
    soft = (confidence / env.cfg.softmax_temperature).softmax(dim=0)
    expected = 0.5 * soft + 0.5 * hard
    expected = (1.0 - env.cfg.min_subtask_prob * 4) * expected + env.cfg.min_subtask_prob
    expected /= expected.sum()

    assert env.extras["log"]["fixed_50_50_blend/blend"] == 0.5
    torch.testing.assert_close(env.distribution, expected)


def test_argmax_only_mode_uses_hard_branch_plus_unchanged_floor():
    env = _make_aces_scheduler_test_env("argmax_only")

    env._update_distribution()

    _, hard = _scheduler_confidence_and_hard(env)
    expected = (1.0 - env.cfg.min_subtask_prob * 4) * hard + env.cfg.min_subtask_prob
    expected /= expected.sum()

    assert env.extras["log"]["argmax_only/blend"] == 1.0
    torch.testing.assert_close(env.distribution, expected)
    torch.testing.assert_close(env.distribution.max(), torch.tensor(0.85))


def test_softmax_only_mode_uses_soft_branch_plus_unchanged_floor():
    env = _make_aces_scheduler_test_env("softmax_only")

    env._update_distribution()

    confidence, _ = _scheduler_confidence_and_hard(env)
    soft = (confidence / env.cfg.softmax_temperature).softmax(dim=0)
    expected = (1.0 - env.cfg.min_subtask_prob * 4) * soft + env.cfg.min_subtask_prob
    expected /= expected.sum()

    assert env.extras["log"]["softmax_only/blend"] == 0.0
    torch.testing.assert_close(env.distribution, expected)


def test_no_ema_mode_uses_current_batch_success_directly():
    env = _make_aces_scheduler_test_env("no_ema")
    env.success_rate = torch.tensor([0.95, 0.85, 0.75, 0.65])

    env._update_distribution()

    expected_batch_success = env.progression[:, :, 0].float().mean(dim=0)
    torch.testing.assert_close(env.success_rate, expected_batch_success)
    assert env.extras["log"]["no_ema/ema_alpha_effective"] == 1.0
    for i in range(4):
        assert env.extras["log"][f"no_ema/batch_success_rate_{i+1}"] == expected_batch_success[i].item()


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


def test_absolute_difficulty_task_changes_only_curriculum_selection_config():
    base_spec = gym.spec("Isaac-Franka-Cabinet-Direct-v0")
    absolute_spec = gym.spec("Isaac-Franka-Cabinet-Absolute-Difficulty-Direct-v0")

    assert base_spec.kwargs["rsl_rl_cfg_entry_point"] == absolute_spec.kwargs["rsl_rl_cfg_entry_point"]
    assert absolute_spec.kwargs["env_cfg_entry_point"].endswith(":FrankaCabinetAbsoluteDifficultyEnvCfg")

    base_cfg = load_cfg_from_registry("Isaac-Franka-Cabinet-Direct-v0", "env_cfg_entry_point").to_dict()
    absolute_cfg = load_cfg_from_registry(
        "Isaac-Franka-Cabinet-Absolute-Difficulty-Direct-v0", "env_cfg_entry_point"
    ).to_dict()

    assert absolute_cfg.pop("reset_state_curriculum_enabled") is True
    assert absolute_cfg.pop("reset_curriculum_mode") == "absolute_difficulty"
    base_cfg.pop("reset_state_curriculum_enabled")
    base_cfg.pop("reset_curriculum_mode")
    assert absolute_cfg == base_cfg


def test_fixed_50_50_blend_task_changes_only_curriculum_selection_config():
    base_spec = gym.spec("Isaac-Franka-Cabinet-Direct-v0")
    fixed_spec = gym.spec("Isaac-Franka-Cabinet-Fixed-50-50-Blend-Direct-v0")

    assert base_spec.kwargs["rsl_rl_cfg_entry_point"] == fixed_spec.kwargs["rsl_rl_cfg_entry_point"]
    assert fixed_spec.kwargs["env_cfg_entry_point"].endswith(":FrankaCabinetFixed5050BlendEnvCfg")

    base_cfg = load_cfg_from_registry("Isaac-Franka-Cabinet-Direct-v0", "env_cfg_entry_point").to_dict()
    fixed_cfg = load_cfg_from_registry(
        "Isaac-Franka-Cabinet-Fixed-50-50-Blend-Direct-v0", "env_cfg_entry_point"
    ).to_dict()

    assert fixed_cfg.pop("reset_state_curriculum_enabled") is True
    assert fixed_cfg.pop("reset_curriculum_mode") == "fixed_50_50_blend"
    base_cfg.pop("reset_state_curriculum_enabled")
    base_cfg.pop("reset_curriculum_mode")
    assert fixed_cfg == base_cfg


def test_argmax_only_task_changes_only_curriculum_selection_config():
    base_spec = gym.spec("Isaac-Franka-Cabinet-Direct-v0")
    argmax_spec = gym.spec("Isaac-Franka-Cabinet-Argmax-Only-Direct-v0")

    assert base_spec.kwargs["rsl_rl_cfg_entry_point"] == argmax_spec.kwargs["rsl_rl_cfg_entry_point"]
    assert argmax_spec.kwargs["env_cfg_entry_point"].endswith(":FrankaCabinetArgmaxOnlyEnvCfg")

    base_cfg = load_cfg_from_registry("Isaac-Franka-Cabinet-Direct-v0", "env_cfg_entry_point").to_dict()
    argmax_cfg = load_cfg_from_registry(
        "Isaac-Franka-Cabinet-Argmax-Only-Direct-v0", "env_cfg_entry_point"
    ).to_dict()

    assert argmax_cfg.pop("reset_state_curriculum_enabled") is True
    assert argmax_cfg.pop("reset_curriculum_mode") == "argmax_only"
    base_cfg.pop("reset_state_curriculum_enabled")
    base_cfg.pop("reset_curriculum_mode")
    assert argmax_cfg == base_cfg


def test_softmax_only_task_changes_only_curriculum_selection_config():
    base_spec = gym.spec("Isaac-Franka-Cabinet-Direct-v0")
    softmax_spec = gym.spec("Isaac-Franka-Cabinet-Softmax-Only-Direct-v0")

    assert base_spec.kwargs["rsl_rl_cfg_entry_point"] == softmax_spec.kwargs["rsl_rl_cfg_entry_point"]
    assert softmax_spec.kwargs["env_cfg_entry_point"].endswith(":FrankaCabinetSoftmaxOnlyEnvCfg")

    base_cfg = load_cfg_from_registry("Isaac-Franka-Cabinet-Direct-v0", "env_cfg_entry_point").to_dict()
    softmax_cfg = load_cfg_from_registry(
        "Isaac-Franka-Cabinet-Softmax-Only-Direct-v0", "env_cfg_entry_point"
    ).to_dict()

    assert softmax_cfg.pop("reset_state_curriculum_enabled") is True
    assert softmax_cfg.pop("reset_curriculum_mode") == "softmax_only"
    base_cfg.pop("reset_state_curriculum_enabled")
    base_cfg.pop("reset_curriculum_mode")
    assert softmax_cfg == base_cfg


def test_no_ema_task_changes_only_curriculum_selection_config():
    base_spec = gym.spec("Isaac-Franka-Cabinet-Direct-v0")
    no_ema_spec = gym.spec("Isaac-Franka-Cabinet-No-EMA-Direct-v0")

    assert base_spec.kwargs["rsl_rl_cfg_entry_point"] == no_ema_spec.kwargs["rsl_rl_cfg_entry_point"]
    assert no_ema_spec.kwargs["env_cfg_entry_point"].endswith(":FrankaCabinetNoEMAEnvCfg")

    base_cfg = load_cfg_from_registry("Isaac-Franka-Cabinet-Direct-v0", "env_cfg_entry_point").to_dict()
    no_ema_cfg = load_cfg_from_registry(
        "Isaac-Franka-Cabinet-No-EMA-Direct-v0", "env_cfg_entry_point"
    ).to_dict()

    assert no_ema_cfg.pop("reset_state_curriculum_enabled") is True
    assert no_ema_cfg.pop("reset_curriculum_mode") == "no_ema"
    assert no_ema_cfg.pop("success_rate_alpha") == 1.0
    base_cfg.pop("reset_state_curriculum_enabled")
    base_cfg.pop("reset_curriculum_mode")
    assert base_cfg.pop("success_rate_alpha") == 0.05
    assert no_ema_cfg == base_cfg


def test_sampling_ratio_0_5_task_changes_only_curriculum_fraction():
    base_spec = gym.spec("Isaac-Franka-Cabinet-Direct-v0")
    ratio_spec = gym.spec("Isaac-Franka-Cabinet-Sampling-Ratio-0-5-Direct-v0")

    assert base_spec.kwargs["rsl_rl_cfg_entry_point"] == ratio_spec.kwargs["rsl_rl_cfg_entry_point"]
    assert ratio_spec.kwargs["env_cfg_entry_point"].endswith(":FrankaCabinetSamplingRatio05EnvCfg")

    base_cfg = load_cfg_from_registry("Isaac-Franka-Cabinet-Direct-v0", "env_cfg_entry_point").to_dict()
    ratio_cfg = load_cfg_from_registry(
        "Isaac-Franka-Cabinet-Sampling-Ratio-0-5-Direct-v0", "env_cfg_entry_point"
    ).to_dict()

    assert ratio_cfg.pop("reset_state_curriculum_enabled") is True
    assert ratio_cfg.pop("sampling_ratio") == 0.5
    base_cfg.pop("reset_state_curriculum_enabled")
    assert base_cfg.pop("sampling_ratio") == 0.3
    assert ratio_cfg == base_cfg


def test_delay_15_task_changes_only_curriculum_start_time():
    base_spec = gym.spec("Isaac-Franka-Cabinet-Direct-v0")
    delay_spec = gym.spec("Isaac-Franka-Cabinet-Delay-15-Direct-v0")

    assert base_spec.kwargs["rsl_rl_cfg_entry_point"] == delay_spec.kwargs["rsl_rl_cfg_entry_point"]
    assert delay_spec.kwargs["env_cfg_entry_point"].endswith(":FrankaCabinetDelay15EnvCfg")

    base_cfg = load_cfg_from_registry("Isaac-Franka-Cabinet-Direct-v0", "env_cfg_entry_point").to_dict()
    delay_cfg = load_cfg_from_registry("Isaac-Franka-Cabinet-Delay-15-Direct-v0", "env_cfg_entry_point").to_dict()

    assert delay_cfg.pop("reset_state_curriculum_enabled") is True
    assert delay_cfg.pop("curriculum_delay_fraction") == 0.15
    base_cfg.pop("reset_state_curriculum_enabled")
    assert base_cfg.pop("curriculum_delay_fraction") == 0.0
    assert delay_cfg == base_cfg


def test_delay_15_enables_reset_replay_at_boundary_only():
    assert not FrankaCabinetEnv._curriculum_reset_is_active(True, 0.149999, 0.15)
    assert FrankaCabinetEnv._curriculum_reset_is_active(True, 0.15, 0.15)
    assert FrankaCabinetEnv._curriculum_reset_is_active(True, 1.0, 0.15)
    assert not FrankaCabinetEnv._curriculum_reset_is_active(False, 1.0, 0.15)


def test_buffer_size_1_task_changes_only_empirical_bank_capacity():
    base_spec = gym.spec("Isaac-Franka-Cabinet-Direct-v0")
    buffer_spec = gym.spec("Isaac-Franka-Cabinet-Buffer-Size-1-Direct-v0")

    assert base_spec.kwargs["rsl_rl_cfg_entry_point"] == buffer_spec.kwargs["rsl_rl_cfg_entry_point"]
    assert buffer_spec.kwargs["env_cfg_entry_point"].endswith(":FrankaCabinetBufferSize1EnvCfg")

    base_cfg = load_cfg_from_registry("Isaac-Franka-Cabinet-Direct-v0", "env_cfg_entry_point").to_dict()
    buffer_cfg = load_cfg_from_registry(
        "Isaac-Franka-Cabinet-Buffer-Size-1-Direct-v0", "env_cfg_entry_point"
    ).to_dict()

    assert buffer_cfg.pop("reset_state_curriculum_enabled") is True
    assert buffer_cfg.pop("success_buffer_size") == 1
    base_cfg.pop("reset_state_curriculum_enabled")
    assert base_cfg.pop("success_buffer_size") == 64
    assert buffer_cfg == base_cfg


def test_velocity_world_task_changes_only_empirical_state_representation():
    base_spec = gym.spec("Isaac-Franka-Cabinet-Direct-v0")
    velocity_spec = gym.spec("Isaac-Franka-Cabinet-Velocity-World-Direct-v0")

    assert base_spec.kwargs["rsl_rl_cfg_entry_point"] == velocity_spec.kwargs["rsl_rl_cfg_entry_point"]
    assert velocity_spec.kwargs["env_cfg_entry_point"].endswith(":FrankaCabinetVelocityWorldEnvCfg")

    base_cfg = load_cfg_from_registry("Isaac-Franka-Cabinet-Direct-v0", "env_cfg_entry_point").to_dict()
    velocity_cfg = load_cfg_from_registry(
        "Isaac-Franka-Cabinet-Velocity-World-Direct-v0", "env_cfg_entry_point"
    ).to_dict()

    assert velocity_cfg.pop("reset_state_curriculum_enabled") is True
    assert velocity_cfg.pop("curriculum_store_velocities") is True
    base_cfg.pop("reset_state_curriculum_enabled")
    assert base_cfg.pop("curriculum_store_velocities") is False
    assert velocity_cfg == base_cfg


def test_velocity_world_appends_robot_and_cabinet_joint_velocities():
    env = object.__new__(FrankaCabinetEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(curriculum_store_velocities=True)
    env._robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.arange(18, dtype=torch.float32).reshape(2, 9),
            joint_vel=torch.arange(18, 36, dtype=torch.float32).reshape(2, 9),
        )
    )
    env._cabinet = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.arange(8, dtype=torch.float32).reshape(2, 4),
            joint_vel=torch.arange(8, 16, dtype=torch.float32).reshape(2, 4),
        )
    )

    world = env._get_world()

    assert world.shape == (2, 26)
    torch.testing.assert_close(
        world,
        torch.cat(
            [
                env._robot.data.joint_pos,
                env._cabinet.data.joint_pos,
                env._robot.data.joint_vel,
                env._cabinet.data.joint_vel,
            ],
            dim=1,
        ),
    )

    env.cfg.curriculum_store_velocities = False
    control_world = env._get_world()
    assert control_world.shape == (2, 13)
    torch.testing.assert_close(
        control_world,
        torch.cat([env._robot.data.joint_pos, env._cabinet.data.joint_pos], dim=1),
    )


def test_simplified_subtasks_task_changes_only_curriculum_decomposition():
    base_spec = gym.spec("Isaac-Franka-Cabinet-Direct-v0")
    simplified_spec = gym.spec("Isaac-Franka-Cabinet-Simplified-Subtasks-Direct-v0")

    assert base_spec.kwargs["rsl_rl_cfg_entry_point"] == simplified_spec.kwargs["rsl_rl_cfg_entry_point"]
    assert simplified_spec.kwargs["env_cfg_entry_point"].endswith(":FrankaCabinetSimplifiedSubtasksEnvCfg")

    base_cfg = load_cfg_from_registry("Isaac-Franka-Cabinet-Direct-v0", "env_cfg_entry_point").to_dict()
    simplified_cfg = load_cfg_from_registry(
        "Isaac-Franka-Cabinet-Simplified-Subtasks-Direct-v0", "env_cfg_entry_point"
    ).to_dict()

    assert simplified_cfg.pop("reset_state_curriculum_enabled") is True
    assert simplified_cfg.pop("curriculum_subtask_decomposition") == "two_stage"
    assert base_cfg.pop("reset_state_curriculum_enabled") is False
    assert base_cfg.pop("curriculum_subtask_decomposition") == "four_stage"
    assert simplified_cfg == base_cfg


def test_simplified_subtasks_are_handle_reach_and_half_open():
    env = object.__new__(FrankaCabinetEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(curriculum_subtask_decomposition="two_stage")
    env.robot_grasp_pos = torch.tensor([[0.05, 0.0, 0.0], [0.15, 0.0, 0.0], [0.05, 0.0, 0.0]])
    env.drawer_grasp_pos = torch.zeros((3, 3))
    env.drawer_joint_idx = 0
    env._cabinet = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.tensor([[0.0], [0.21], [0.21]])))

    completions = env._get_subtasks()

    assert completions.shape == (3, 2)
    torch.testing.assert_close(
        completions,
        torch.tensor([[True, False], [False, True], [True, True]]),
    )

    env.cfg.curriculum_subtask_decomposition = "four_stage"
    assert env._get_subtasks().shape == (3, 4)
