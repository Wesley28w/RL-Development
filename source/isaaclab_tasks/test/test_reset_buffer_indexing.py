# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU regression tests for predecessor reset banks; no simulator or training required.

The adapter tests execute the actual reset functions isolated from simulator
imports, with small recording assets. Physics is not simulated by these tests.
"""

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

TASKS = Path(__file__).resolve().parents[1] / "isaaclab_tasks"
spec = importlib.util.spec_from_file_location("reset_buffer", TASKS / "utils/reset_buffer.py")
reset_buffer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reset_buffer)
available_reset_targets = reset_buffer.available_reset_targets
natural_reset_ids = reset_buffer.natural_reset_ids
sample_reset_worlds = reset_buffer.sample_reset_worlds


class RecordingAsset:
    def __init__(self, n: int, joints: int):
        self.num_joints = joints
        self.data = SimpleNamespace(
            default_joint_pos=torch.zeros(n, joints),
            default_root_state=torch.zeros(n, 13),
            soft_joint_pos_limits=torch.tensor([-100.0, 100.0]).repeat(n, joints, 1),
        )
        self.writes = []

    def write_joint_state_to_sim(self, pos, vel, *, env_ids):
        self.writes.append((env_ids.clone(), pos.clone(), vel.clone()))

    def set_joint_position_target(self, pos, *, env_ids):
        pass

    def write_root_pose_to_sim(self, pose, *, env_ids):
        self.writes.append((env_ids.clone(), pose.clone()))

    def write_root_velocity_to_sim(self, vel, *, env_ids):
        pass


class ResetBase:
    def _reset_idx(self, env_ids):
        pass


def load_reset_adapter(
    relative_path: str, class_name: str | None = None, function_name="sample_curriculum_reset_state", extra_globals=None
):
    """Load real reset-function bodies without starting Isaac Sim during a tensor unit test."""
    tree = ast.parse((TASKS / relative_path).read_text(encoding="utf-8"))
    if class_name:
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        methods = [
            n
            for n in original.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("_reset_idx", "_available_curriculum_targets", "_uniform_available_distribution")
        ]
        nodes = [
            ast.ClassDef(
                name=class_name,
                bases=[ast.Name(id="ResetBase", ctx=ast.Load())],
                keywords=[],
                body=methods,
                decorator_list=[],
            )
        ]
    else:
        nodes = [next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name)]
    namespace = {
        "torch": torch,
        "ResetBase": ResetBase,
        "NUM_SUBTASKS": 4,
        "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
        "available_reset_targets": available_reset_targets,
        "sample_reset_worlds": sample_reset_worlds,
        "natural_reset_ids": natural_reset_ids,
        "sample_uniform": lambda low, high, shape, device: torch.zeros(shape, device=device),
    }
    namespace.update(extra_globals or {})
    module = ast.fix_missing_locations(
        ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
            type_ignores=[],
        )
    )
    exec(compile(module, str(TASKS / relative_path), "exec"), namespace)
    return namespace[class_name or function_name]


def banks(n: int, width: int):
    # Bank A=10, B=20, C=30, etc.; unwritten slots are deliberately conspicuous.
    worlds = torch.full((n, 3, width), 99.0)
    worlds[:, 0] = 10.0 * torch.arange(1, n + 1)[:, None]
    return worlds


class TestPredecessorBankMapping(unittest.TestCase):
    def test_bottleneck_b_replays_a_even_if_b_has_never_completed(self):
        counts = torch.tensor([1, 0, 0])
        torch.testing.assert_close(available_reset_targets(counts), torch.tensor([True, True, False]))
        replay, worlds = sample_reset_worlds(banks(3, 2), counts, torch.tensor([1] * 64))
        self.assertTrue(bool(replay.all()))
        torch.testing.assert_close(worlds, torch.full((64, 2), 10.0))

    def test_each_target_maps_to_its_immediate_predecessor(self):
        for n in (2, 3, 4):
            with self.subTest(milestones=n):
                replay, worlds = sample_reset_worlds(banks(n, 2), torch.ones(n, dtype=torch.long), torch.arange(n))
                torch.testing.assert_close(replay, torch.arange(n) > 0)
                torch.testing.assert_close(worlds[:, 0], torch.arange(1, n).float() * 10)

    def test_first_target_never_wraps_to_last_bank_or_draws_a_world(self):
        torch.manual_seed(42)
        before = torch.random.get_rng_state()
        replay, worlds = sample_reset_worlds(banks(4, 2), torch.tensor([0, 0, 0, 1]), torch.zeros(8, dtype=torch.long))
        self.assertFalse(bool(replay.any()))
        self.assertEqual(worlds.shape, (0, 2))
        torch.testing.assert_close(before, torch.random.get_rng_state())

    def test_empty_banks_leave_only_the_normal_reset_available(self):
        torch.testing.assert_close(
            available_reset_targets(torch.zeros(4, dtype=torch.long)), torch.tensor([True, False, False, False])
        )

    def test_empty_predecessor_is_rejected_instead_of_replaying_filler(self):
        with self.assertRaisesRegex(ValueError, "unpopulated"):
            sample_reset_worlds(banks(3, 2), torch.zeros(3, dtype=torch.long), torch.tensor([1]))

    def test_tscl_retains_direct_bank_categories(self):
        counts = torch.tensor([0, 1, 1])
        torch.testing.assert_close(
            available_reset_targets(counts, use_predecessor=False), torch.tensor([False, True, True])
        )
        replay, worlds = sample_reset_worlds(banks(3, 2), counts, torch.tensor([1, 2]), use_predecessor=False)
        self.assertTrue(bool(replay.all()))
        torch.testing.assert_close(worlds[:, 0], torch.tensor([20.0, 30.0]))


class TestManagerResetAdapters(unittest.TestCase):
    def test_baseline_reset_events_exclude_replays_and_preserve_disabled_path(self):
        for task, randomized_term in (("lift", "reset_root_state_uniform"), ("cabinet", "reset_joints_by_offset")):
            for original in ("reset_scene_to_default", randomized_term):
                with self.subTest(task=task, event=original):
                    calls = []
                    wrapper = load_reset_adapter(
                        f"manager_based/manipulation/{task}/mdp/events.py",
                        function_name=original + "_natural",
                        extra_globals={"_baseline_" + original: lambda env, ids, *a, **k: calls.append(ids.clone())},
                    )
                    tracker = SimpleNamespace(is_curriculum_episode=torch.tensor([False, True, False, True]))
                    env = SimpleNamespace(
                        cfg=SimpleNamespace(reset_state_curriculum_enabled=True),
                        reward_manager=SimpleNamespace(
                            cfg=SimpleNamespace(subtask_progression_tracker=SimpleNamespace(func=tracker))
                        ),
                    )
                    args = () if original == "reset_scene_to_default" else ((0.0, 0.1), (0.0, 0.0))
                    wrapper(env, torch.arange(4), *args)
                    torch.testing.assert_close(calls[-1], torch.tensor([0, 2]))
                    env.cfg.reset_state_curriculum_enabled = False
                    wrapper(env, torch.arange(4), *args)
                    torch.testing.assert_close(calls[-1], torch.arange(4))

    def test_lift_and_manager_cabinet_restore_a_for_target_b_without_noise(self):
        for task in ("lift", "cabinet"):
            for target in (0, 1, 3):
                with self.subTest(task=task, target=target):
                    reset = load_reset_adapter(f"manager_based/manipulation/{task}/mdp/events.py")
                    n, joints = 4, 2
                    robot, other = RecordingAsset(n, joints), RecordingAsset(n, 2)
                    tracker = SimpleNamespace(
                        pose_buffer_count=torch.tensor([1, 1, 1, 0]),
                        success_buffer=banks(4, joints + (7 if task == "lift" else 2)),
                        is_curriculum_episode=torch.zeros(n, dtype=torch.bool),
                        curriculum_subtask=torch.full((n,), -1),
                        tscl_teacher=None,
                        _log={},
                        distribution=torch.nn.functional.one_hot(torch.tensor(target), 4).float(),
                    )
                    env = SimpleNamespace(
                        cfg=SimpleNamespace(
                            reset_state_curriculum_enabled=True, sampling_ratio=1.0, reset_curriculum_mode="aces"
                        ),
                        device="cpu",
                        scene={"robot": robot, "object": other, "cabinet": other},
                        reward_manager=SimpleNamespace(
                            cfg=SimpleNamespace(subtask_progression_tracker=SimpleNamespace(func=tracker))
                        ),
                    )

                    # Dict plus attribute access, matching InteractiveScene.
                    class Scene(dict):
                        env_origins = torch.zeros(n, 3)

                    env.scene = Scene(env.scene)
                    reset(env, torch.arange(n))
                    if target == 0:
                        self.assertFalse(robot.writes)
                        self.assertFalse(bool(tracker.is_curriculum_episode.any()))
                        self.assertEqual(tracker._log["curriculum/sample_rate"], 0.0)
                    else:
                        self.assertTrue(bool(tracker.is_curriculum_episode.all()))
                        torch.testing.assert_close(robot.writes[0][1], torch.full((n, joints), 10.0 * target))
                        torch.testing.assert_close(robot.writes[0][2], torch.zeros(n, joints))


class TestDirectResetAdapters(unittest.TestCase):
    def test_cabinet_and_factory_use_predecessors_and_preserve_first_target(self):
        for task, class_name in (("franka_cabinet", "FrankaCabinetEnv"), ("factory", "FactoryEnv")):
            adapter = load_reset_adapter(f"direct/{task}/{task}_env.py", class_name)
            for target in (0, 1, 2):
                with self.subTest(task=task, target=target):
                    n, milestones, joints = 4, 3, 9
                    env = adapter()
                    env.device = "cpu"
                    env.cfg = SimpleNamespace(
                        reset_state_curriculum_enabled=True,
                        sampling_ratio=1.0,
                        reset_curriculum_mode="aces",
                        curriculum_store_velocities=False,
                        ctrl=SimpleNamespace(reset_joints=[0.0] * 7),
                    )
                    env.curriculum_enabled = True
                    env.tscl_teacher = None
                    env.pose_buffer_count = torch.tensor([1, 1, 0])
                    env.success_buffer = banks(milestones, 23 if task == "factory" else joints + 2)
                    env.distribution = torch.nn.functional.one_hot(torch.tensor(target), milestones).float()
                    env.curriculum_subtask = torch.full((n,), -1)
                    env.is_curriculum_episode = torch.zeros(n, dtype=torch.bool)
                    env.progression = torch.zeros(n, milestones, 1)
                    env._robot = RecordingAsset(n, joints)
                    env._cabinet = RecordingAsset(n, 2)
                    env.robot_dof_targets = torch.zeros(n, joints)
                    env.robot_dof_lower_limits = torch.full((joints,), -100.0)
                    env.robot_dof_upper_limits = torch.full((joints,), 100.0)
                    env._compute_intermediate_values = lambda ids: None
                    restored = []
                    randomized = []
                    if task == "factory":
                        env._held_asset = RecordingAsset(n, 0)
                        env._fixed_asset = RecordingAsset(n, 0)
                        env.cfg_task = SimpleNamespace(held_asset_cfg=SimpleNamespace(diameter=0.02))
                        env._set_assets_to_default_pose = lambda **kwargs: None
                        env._set_franka_to_default_pose = lambda joints, env_ids: restored.append(joints.clone())
                        env.step_sim_no_action = lambda: None
                        env.randomize_initial_state = lambda ids: randomized.append(ids.clone())
                        env.init_fixed_pos_obs_noise = torch.ones(n, 3)
                        env._refresh_fixed_asset_observation_frame = lambda: None
                        env.joint_pos = torch.zeros(n, joints)
                        env.prev_joint_pos = torch.zeros(n, 7)
                        env.fingertip_midpoint_pos = torch.zeros(n, 3)
                        env.prev_fingertip_pos = torch.zeros(n, 3)
                        env.fingertip_midpoint_quat = torch.zeros(n, 4)
                        env.prev_fingertip_quat = torch.zeros(n, 4)
                        env.ee_angvel_fd = torch.ones(n, 3)
                        env.ee_linvel_fd = torch.ones(n, 3)
                        env.actions = torch.ones(n, 6)
                        env.prev_actions = torch.ones(n, 6)
                    env._reset_idx(torch.arange(n))
                    self.assertEqual(bool(env.is_curriculum_episode.any()), target != 0)
                    if task == "factory":
                        self.assertEqual(len(restored), 1 + int(target != 0))
                        if target:
                            torch.testing.assert_close(restored[-1], torch.full((n, joints), target * 10.0))
                            self.assertFalse(randomized)
                            torch.testing.assert_close(env.init_fixed_pos_obs_noise, torch.zeros(n, 3))
                            torch.testing.assert_close(env.ee_linvel_fd, torch.zeros(n, 3))
                        else:
                            torch.testing.assert_close(randomized[0], torch.arange(n))
                            torch.testing.assert_close(env.init_fixed_pos_obs_noise, torch.ones(n, 3))
                    else:
                        torch.testing.assert_close(env._robot.writes[0][1], torch.full((n, joints), target * 10.0))


class TestRuntimeConfigurations(unittest.TestCase):
    def test_requested_budgets_and_rollout_consistency(self):
        def scalar(path, name):
            tree = ast.parse((TASKS / path).read_text(encoding="utf-8"))
            return next(
                ast.literal_eval(n.value)
                for n in ast.walk(tree)
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
            )

        self.assertEqual(scalar("direct/franka_cabinet/agents/rsl_rl_ppo_cfg.py", "max_iterations"), 1250)
        self.assertEqual(scalar("direct/franka_cabinet/franka_cabinet_env.py", "curriculum_total_iterations"), 1250)
        self.assertEqual(
            scalar("manager_based/manipulation/lift/config/franka/agents/rsl_rl_ppo_cfg.py", "max_iterations"), 1500
        )
        self.assertEqual(scalar("direct/factory/factory_env_cfg.py", "curriculum_total_iterations"), 150)
        for path, expected in (
            ("direct/franka_cabinet/franka_cabinet_env.py", 8192),
            ("direct/factory/factory_env_cfg.py", 256),
            ("manager_based/manipulation/lift/lift_env_cfg.py", 4096),
        ):
            tree = ast.parse((TASKS / path).read_text(encoding="utf-8"))
            self.assertIn(
                expected,
                [
                    ast.literal_eval(k.value)
                    for n in ast.walk(tree)
                    if isinstance(n, ast.Call)
                    for k in n.keywords
                    if k.arg == "num_envs"
                ],
            )

    def test_manager_replays_are_registered_before_natural_reset_terms(self):
        for path in (
            "manager_based/manipulation/lift/lift_env_cfg.py",
            "manager_based/manipulation/cabinet/cabinet_env_cfg.py",
            "manager_based/manipulation/cabinet/config/openarm/cabinet_openarm_env_cfg.py",
        ):
            tree = ast.parse((TASKS / path).read_text(encoding="utf-8"))
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EventCfg")
            fields = [n.targets[0].id for n in cls.body if isinstance(n, ast.Assign)]
            self.assertLess(fields.index("sample_curriculum_reset_state"), fields.index("reset_all"))

    def test_no_curriculum_noise_fields_or_noise_injection_remain(self):
        for relative in (
            "direct/franka_cabinet/franka_cabinet_env.py",
            "direct/factory/factory_env.py",
            "direct/factory/factory_env_cfg.py",
            "manager_based/manipulation/lift/mdp/events.py",
            "manager_based/manipulation/lift/lift_env_cfg.py",
            "manager_based/manipulation/cabinet/mdp/events.py",
            "manager_based/manipulation/cabinet/cabinet_env_cfg.py",
        ):
            source = (TASKS / relative).read_text(encoding="utf-8")
            self.assertNotIn("curriculum_dr", source)
            self.assertNotIn("cfg.action_std", source)
            self.assertNotIn("cfg.observation_std", source)


if __name__ == "__main__":
    unittest.main()
