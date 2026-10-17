# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small simulator reset checks, isolated per process; no training or result logging."""

import subprocess
import sys
from pathlib import Path


def _check_lift():
    cfg = parse_env_cfg("Isaac-Lift-Cube-Franka-v0", device="cuda:0", num_envs=4)
    cfg.reset_state_curriculum_enabled = False
    env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=cfg)
    try:
        env.reset()
        raw = env.unwrapped
        tracker = raw.reward_manager.cfg.subtask_progression_tracker.func
        recorded = tracker._get_world(
            raw, torch.tensor([0], device=raw.device), SceneEntityCfg("robot"), SceneEntityCfg("object")
        )
        tracker.success_buffer[0, 0] = recorded[0]
        tracker.pose_buffer_count.zero_()
        tracker.pose_buffer_count[0] = 1
        tracker.distribution = torch.tensor([0.0, 1.0, 0.0, 0.0], device=raw.device)
        raw.cfg.reset_state_curriculum_enabled = True
        raw.cfg.sampling_ratio = 1.0
        env.reset()
        robot = raw.scene["robot"]
        torch.testing.assert_close(
            robot.data.joint_pos, recorded[:, : robot.num_joints].expand_as(robot.data.joint_pos)
        )
        object_pos = raw.scene["object"].data.root_pos_w - raw.scene.env_origins
        torch.testing.assert_close(
            object_pos, recorded[:, robot.num_joints : robot.num_joints + 3].expand_as(object_pos)
        )
        assert bool(tracker.is_curriculum_episode.all())
        assert bool((tracker.curriculum_subtask == 1).all())
    finally:
        env.close()


def _check_direct_cabinet():
    cfg = parse_env_cfg("Isaac-Franka-Cabinet-Direct-v0", device="cuda:0", num_envs=4)
    cfg.reset_state_curriculum_enabled = False
    env = gym.make("Isaac-Franka-Cabinet-Direct-v0", cfg=cfg)
    try:
        env.reset()
        raw = env.unwrapped
        recorded = raw._get_world()[0].clone()
        raw.success_buffer[0, 0] = recorded
        raw.pose_buffer_count.zero_()
        raw.pose_buffer_count[0] = 1
        raw.distribution = torch.tensor([0.0, 1.0, 0.0, 0.0], device=raw.device)
        raw.cfg.reset_state_curriculum_enabled = True
        raw.curriculum_enabled = True
        raw.cfg.sampling_ratio = 1.0
        env.reset()
        expected = recorded[: raw._robot.num_joints].expand_as(raw._robot.data.joint_pos)
        torch.testing.assert_close(raw._robot.data.joint_pos, expected)
        assert bool(raw.is_curriculum_episode.all())
        assert bool((raw.curriculum_subtask == 1).all())
    finally:
        env.close()


def _check_factory():
    cfg = parse_env_cfg("Isaac-Factory-NutThread-Direct-v0", device="cuda:0", num_envs=4)
    cfg.reset_state_curriculum_enabled = False
    env = gym.make("Isaac-Factory-NutThread-Direct-v0", cfg=cfg)
    try:
        env.reset()
        raw = env.unwrapped
        recorded = raw._get_world()[0].clone()
        raw.success_buffer[0, 0] = recorded
        raw.pose_buffer_count.zero_()
        raw.pose_buffer_count[0] = 1
        raw.distribution = torch.tensor([0.0, 1.0, 0.0], device=raw.device)
        raw.cfg.reset_state_curriculum_enabled = True
        raw.curriculum_enabled = True
        raw.cfg.sampling_ratio = 0.5
        env.reset()
        replay = raw.is_curriculum_episode
        assert int(replay.sum()) == 2
        assert bool((raw.curriculum_subtask[replay] == 1).all())
        torch.testing.assert_close(raw.ctrl_target_joint_pos[replay], recorded[:9].expand(2, -1))
        torch.testing.assert_close(raw.init_fixed_pos_obs_noise[replay], torch.zeros((2, 3), device=raw.device))
        assert bool(torch.isfinite(raw._get_observations()["policy"]).all())
    finally:
        env.close()


def _run_isolated(task):
    # Recreating different task types in one Isaac Sim process can deadlock on stage teardown.
    subprocess.run([sys.executable, str(Path(__file__).resolve()), task], check=True, timeout=120)


def test_lift_predecessor_replay_is_not_overwritten_by_baseline_reset_events():
    _run_isolated("lift")


def test_direct_cabinet_replays_predecessor_without_joint_perturbation():
    _run_isolated("direct_cabinet")


def test_factory_mixed_batch_randomizes_only_natural_episodes():
    _run_isolated("factory")


if __name__ == "__main__":
    task = sys.argv[1]
    # Do not pass our positional selector to Isaac Sim's application arguments.
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    import gymnasium as gym
    import torch

    from isaaclab.managers import SceneEntityCfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    try:
        {"lift": _check_lift, "direct_cabinet": _check_direct_cabinet, "factory": _check_factory}[task]()
        print(f"Reset simulation check passed: {task}", flush=True)
    finally:
        simulation_app.close()
