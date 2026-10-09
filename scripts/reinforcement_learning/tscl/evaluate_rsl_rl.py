# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate final TSCL Lift or Cabinet checkpoints on curriculum-free resets."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from eval_utils import output_paths, refuse_overwrite, write_results

from isaaclab.app import AppLauncher

# local imports
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "rsl_rl"))
import cli_args  # noqa: E402, isort: skip


original_argv = list(sys.argv)
parser = argparse.ArgumentParser(description="Fixed-budget TSCL evaluation for RSL-RL Franka tasks.")
parser.add_argument("--task", required=True, choices=("Isaac-Lift-Cube-Franka-v0", "Isaac-Franka-Cabinet-Direct-v0"))
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point", help="Agent configuration registry entry point.")
parser.add_argument("--train_seed", required=True, type=int, help="Training seed represented by the checkpoint.")
parser.add_argument("--seed", type=int, default=12345, help="Common evaluation-environment seed.")
parser.add_argument("--num_envs", type=int, default=512, help="Parallel evaluation environments.")
parser.add_argument("--episodes", type=int, default=10240, help="Exact number of completed episodes to record.")
parser.add_argument("--output_dir", required=True, help="Directory for JSON summary and episode CSV.")
parser.add_argument("--overwrite", action="store_true", help="Replace an existing result for this training seed.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.checkpoint is None:
    parser.error("--checkpoint is required and must name the exact final model_N.pt checkpoint")
summary_path, episodes_path = output_paths(args_cli.output_dir, args_cli.train_seed)
refuse_overwrite(summary_path, episodes_path, args_cli.overwrite)
if args_cli.episodes <= 0 or args_cli.num_envs <= 0:
    parser.error("--episodes and --num_envs must both be positive")

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim starts."""

import importlib.metadata as metadata
import types
from pathlib import Path

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import combine_frame_transforms

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


class _LiftMetricCapture:
    """Capture Lift's original success decision and read-only episode diagnostics."""

    def __init__(self, original, env):
        self.original = original
        self.env = env
        tracker_cfg = env.reward_manager.cfg.subtask_progression_tracker
        self.tracker = tracker_cfg.func
        self.params = tracker_cfg.params
        robot = env.scene[self.params["robot_cfg"].name]
        self.gripper_joint_ids, _ = robot.find_joints(list(self.params["gripper_joint_names"]))

        self.terminal_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.terminal_subtasks = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device)
        self.final_ee_object_distance = torch.zeros(env.num_envs, device=env.device)
        self.final_goal_distance = torch.zeros(env.num_envs, device=env.device)
        self.final_object_height = torch.zeros(env.num_envs, device=env.device)
        self.final_gripper_opening = torch.zeros(env.num_envs, device=env.device)
        self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        self.ever_subtasks = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device)
        self.min_ee_object_distance = torch.full((env.num_envs,), torch.inf, device=env.device)
        self.min_goal_distance = torch.full((env.num_envs,), torch.inf, device=env.device)
        self.max_object_height = torch.full((env.num_envs,), -torch.inf, device=env.device)

    def _ids(self, env_ids):
        if isinstance(env_ids, slice):
            return torch.arange(self.env.num_envs, device=self.env.device)[env_ids]
        return torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)

    def _current(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        robot = self.env.scene[self.params["robot_cfg"].name]
        obj = self.env.scene[self.params["object_cfg"].name]
        ee_frame = self.env.scene[self.params["ee_frame_cfg"].name]
        ee_pos = ee_frame.data.target_pos_w[env_ids, 0, :]
        object_pos = obj.data.root_pos_w[env_ids]
        ee_distance = torch.norm(ee_pos - object_pos, dim=-1)
        gripper_opening = robot.data.joint_pos[env_ids][:, self.gripper_joint_ids].mean(dim=-1)
        command = self.env.command_manager.get_command(self.params["command_name"])
        desired_pos, _ = combine_frame_transforms(
            robot.data.root_pos_w[env_ids], robot.data.root_quat_w[env_ids], command[env_ids, :3]
        )
        goal_distance = torch.norm(desired_pos - object_pos, dim=-1)
        gripper_holding = (gripper_opening < self.params["gripper_closed_threshold"]) & (
            gripper_opening > self.params["gripper_min_opening"]
        )
        subtasks = torch.stack(
            [
                ee_distance < self.params["reach_threshold"],
                (ee_distance < self.params["grasp_distance_threshold"]) & gripper_holding,
                object_pos[:, 2] > self.params["lift_height_threshold"],
                goal_distance < self.params["near_goal_threshold"],
            ],
            dim=1,
        )
        return {
            "subtasks": subtasks,
            "ee_distance": ee_distance,
            "goal_distance": goal_distance,
            "object_height": object_pos[:, 2],
            "gripper_opening": gripper_opening,
        }

    def observe_current(self, env_ids=None) -> None:
        ids = self._ids(slice(None) if env_ids is None else env_ids)
        current = self._current(ids)
        self.ever_subtasks[ids] |= current["subtasks"]
        self.min_ee_object_distance[ids] = torch.minimum(self.min_ee_object_distance[ids], current["ee_distance"])
        self.min_goal_distance[ids] = torch.minimum(self.min_goal_distance[ids], current["goal_distance"])
        self.max_object_height[ids] = torch.maximum(self.max_object_height[ids], current["object_height"])

    def __call__(self, env, env_ids, **kwargs):
        env_ids = self._ids(env_ids)
        current = self._current(env_ids)
        self.ever_subtasks[env_ids] |= current["subtasks"]
        self.min_ee_object_distance[env_ids] = torch.minimum(
            self.min_ee_object_distance[env_ids], current["ee_distance"]
        )
        self.min_goal_distance[env_ids] = torch.minimum(self.min_goal_distance[env_ids], current["goal_distance"])
        self.max_object_height[env_ids] = torch.maximum(self.max_object_height[env_ids], current["object_height"])
        command_name = kwargs.get("command_name", "object_pose")
        threshold = float(kwargs.get("threshold", 0.02))
        robot = env.scene["robot"]
        obj = env.scene["object"]
        command = env.command_manager.get_command(command_name)
        desired_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w[env_ids], robot.data.root_quat_w[env_ids], command[env_ids, :3]
        )
        self.terminal_success[env_ids] = torch.norm(desired_pos_w - obj.data.root_pos_w[env_ids], dim=1) < threshold
        self.terminal_subtasks[env_ids] = current["subtasks"]
        self.final_ee_object_distance[env_ids] = current["ee_distance"]
        self.final_goal_distance[env_ids] = current["goal_distance"]
        self.final_object_height[env_ids] = current["object_height"]
        self.final_gripper_opening[env_ids] = current["gripper_opening"]
        self.valid[env_ids] = True
        return self.original(env, env_ids, **kwargs)

    def reset(self, env_ids=None):
        return self.original.reset(env_ids=env_ids)

    def take(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if not bool(torch.all(self.valid[env_ids]).item()):
            raise RuntimeError("Lift success capture did not run for every completed episode.")
        values = {
            "success": self.terminal_success[env_ids].clone(),
            "final_subtasks": self.terminal_subtasks[env_ids].clone(),
            "ever_subtasks": self.ever_subtasks[env_ids].clone(),
            "final_ee_object_distance_m": self.final_ee_object_distance[env_ids].clone(),
            "min_ee_object_distance_m": self.min_ee_object_distance[env_ids].clone(),
            "final_goal_distance_m": self.final_goal_distance[env_ids].clone(),
            "min_goal_distance_m": self.min_goal_distance[env_ids].clone(),
            "final_object_height_m": self.final_object_height[env_ids].clone(),
            "max_object_height_m": self.max_object_height[env_ids].clone(),
            "final_gripper_opening_m": self.final_gripper_opening[env_ids].clone(),
        }
        self.valid[env_ids] = False
        self.ever_subtasks[env_ids] = False
        self.min_ee_object_distance[env_ids] = torch.inf
        self.min_goal_distance[env_ids] = torch.inf
        self.max_object_height[env_ids] = -torch.inf
        return values

    def clear(self):
        self.valid[:] = False
        self.ever_subtasks[:] = False
        self.min_ee_object_distance[:] = torch.inf
        self.min_goal_distance[:] = torch.inf
        self.max_object_height[:] = -torch.inf

    def __getattr__(self, name):
        return getattr(self.original, name)


def _install_lift_capture(raw_env) -> _LiftMetricCapture:
    for name, term_cfg in zip(raw_env.curriculum_manager._term_names, raw_env.curriculum_manager._term_cfgs):
        if name == "lift_success_rate":
            proxy = _LiftMetricCapture(term_cfg.func, raw_env)
            term_cfg.func = proxy
            return proxy
    raise RuntimeError("The Lift task has no lift_success_rate term; refusing to substitute another metric.")


class _CabinetMetricCapture:
    """Capture Cabinet's continuous margin, milestone ladder, and physical diagnostics."""

    def __init__(self, env):
        self.env = env
        device = env.device
        self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        self.ever_subtasks = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=device)
        self.max_margin = torch.zeros(env.num_envs, device=device)
        self.max_drawer_position = torch.full((env.num_envs,), -torch.inf, device=device)
        self.min_gripper_handle_distance = torch.full((env.num_envs,), torch.inf, device=device)
        self.final_subtasks = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=device)
        self.final_margin = torch.zeros(env.num_envs, device=device)
        self.final_drawer_position = torch.zeros(env.num_envs, device=device)
        self.final_gripper_handle_distance = torch.zeros(env.num_envs, device=device)
        self.terminal_success = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

    def _current(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        drawer = self.env._cabinet.data.joint_pos[env_ids, self.env.drawer_joint_idx]
        distance = torch.norm(self.env.robot_grasp_pos[env_ids] - self.env.drawer_grasp_pos[env_ids], p=2, dim=-1)
        margin = torch.clamp(drawer / 0.39, 0.0, 1.0)
        subtasks = torch.stack([distance < 0.20, distance < 0.10, drawer > 0.20, drawer > 0.38], dim=1)
        return {"drawer": drawer, "distance": distance, "margin": margin, "subtasks": subtasks}

    def observe_current(self, env_ids=None) -> None:
        ids = (
            torch.arange(self.env.num_envs, device=self.env.device)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)
        )
        current = self._current(ids)
        self.ever_subtasks[ids] |= current["subtasks"]
        self.max_margin[ids] = torch.maximum(self.max_margin[ids], current["margin"])
        self.max_drawer_position[ids] = torch.maximum(self.max_drawer_position[ids], current["drawer"])
        self.min_gripper_handle_distance[ids] = torch.minimum(
            self.min_gripper_handle_distance[ids], current["distance"]
        )

    def capture_terminal(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.observe_current(env_ids)
        current = self._current(env_ids)
        self.final_subtasks[env_ids] = current["subtasks"]
        self.final_margin[env_ids] = current["margin"]
        self.final_drawer_position[env_ids] = current["drawer"]
        self.final_gripper_handle_distance[env_ids] = current["distance"]
        self.terminal_success[env_ids] = self.env.reset_terminated[env_ids].bool()
        self.valid[env_ids] = True

    def take(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if not bool(torch.all(self.valid[env_ids]).item()):
            raise RuntimeError("Cabinet metric capture did not run for every completed episode.")
        values = {
            "success": self.terminal_success[env_ids].clone(),
            "final_subtasks": self.final_subtasks[env_ids].clone(),
            "ever_subtasks": self.ever_subtasks[env_ids].clone(),
            "final_success_margin": self.final_margin[env_ids].clone(),
            "max_success_margin": self.max_margin[env_ids].clone(),
            "final_drawer_position_m": self.final_drawer_position[env_ids].clone(),
            "max_drawer_position_m": self.max_drawer_position[env_ids].clone(),
            "final_gripper_handle_distance_m": self.final_gripper_handle_distance[env_ids].clone(),
            "min_gripper_handle_distance_m": self.min_gripper_handle_distance[env_ids].clone(),
        }
        self.valid[env_ids] = False
        self.ever_subtasks[env_ids] = False
        self.max_margin[env_ids] = 0.0
        self.max_drawer_position[env_ids] = -torch.inf
        self.min_gripper_handle_distance[env_ids] = torch.inf
        return values


def _install_cabinet_capture(raw_env) -> _CabinetMetricCapture:
    capture = _CabinetMetricCapture(raw_env)
    original_rewards = raw_env._get_rewards

    def wrapped_rewards(_env):
        rewards = original_rewards()
        capture.observe_current()
        capture.capture_terminal(torch.nonzero(_env.reset_buf, as_tuple=False).squeeze(-1))
        return rewards

    raw_env._get_rewards = types.MethodType(wrapped_rewards, raw_env)
    return capture


def _assert_curriculum_off(raw_env) -> None:
    if bool(raw_env.cfg.reset_state_curriculum_enabled):
        raise RuntimeError("Evaluation requires reset_state_curriculum_enabled=False.")
    if hasattr(raw_env, "is_curriculum_episode") and bool(raw_env.is_curriculum_episode.any().item()):
        raise RuntimeError("A curriculum reset was observed during evaluation.")
    if hasattr(raw_env, "reward_manager"):
        tracker_cfg = getattr(raw_env.reward_manager.cfg, "subtask_progression_tracker", None)
        if tracker_cfg is not None and bool(tracker_cfg.func.is_curriculum_episode.any().item()):
            raise RuntimeError("A Lift curriculum reset was observed during evaluation.")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Evaluation changes only the reset scheduler: every episode starts from the task's original distribution.
    env_cfg.reset_state_curriculum_enabled = False
    env_cfg.reset_curriculum_mode = "aces"
    env_cfg.log_dir = str(Path(args_cli.output_dir).resolve())

    checkpoint = retrieve_file_path(args_cli.checkpoint)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    raw_env = env.unwrapped
    is_lift = "Lift" in args_cli.task
    if not is_lift:
        # These legacy controller fields are diagnostics only. Some Cabinet revisions still log them after
        # removing their initialization; provide the original neutral values so evaluation remains read-only.
        if not hasattr(raw_env, "controller_snapshot"):
            raw_env.controller_snapshot = None
        if not hasattr(raw_env, "controller_snapshot_two"):
            raw_env.controller_snapshot_two = None
    lift_capture = _install_lift_capture(raw_env) if is_lift else None
    cabinet_capture = None if is_lift else _install_cabinet_capture(raw_env)
    metric_capture = lift_capture if is_lift else cabinet_capture
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=raw_env.device)

    obs, _ = env.reset()
    if lift_capture is not None:
        lift_capture.clear()
        lift_capture.observe_current()
    else:
        raw_env._compute_intermediate_values()
        cabinet_capture.observe_current()
    _assert_curriculum_off(raw_env)

    episode_returns = torch.zeros(raw_env.num_envs, device=raw_env.device)
    episode_lengths = torch.zeros(raw_env.num_envs, dtype=torch.long, device=raw_env.device)
    action_l2_sum = torch.zeros(raw_env.num_envs, device=raw_env.device)
    max_abs_action = torch.zeros(raw_env.num_envs, device=raw_env.device)
    records: list[dict[str, int | float | bool]] = []

    while simulation_app.is_running() and len(records) < args_cli.episodes:
        with torch.inference_mode():
            actions = policy(obs)
            applied_actions = (
                torch.clamp(actions, -agent_cfg.clip_actions, agent_cfg.clip_actions)
                if agent_cfg.clip_actions is not None
                else actions
            )
            action_l2_sum += torch.linalg.vector_norm(applied_actions, dim=-1)
            max_abs_action = torch.maximum(max_abs_action, applied_actions.abs().amax(dim=-1))
            obs, rewards, dones, _ = env.step(actions)
            episode_returns += rewards
            episode_lengths += 1

            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                diagnostics = metric_capture.take(done_ids)
                success = diagnostics["success"]
                count = min(done_ids.numel(), args_cli.episodes - len(records))
                selected = done_ids[:count]
                selected_success = success[:count]
                selected_returns = episode_returns[selected].detach().cpu().tolist()
                selected_lengths = episode_lengths[selected].detach().cpu().tolist()
                selected_ids = selected.detach().cpu().tolist()
                selected_successes = selected_success.detach().cpu().tolist()
                selected_timeouts = raw_env.reset_time_outs[selected].bool().detach().cpu().tolist()
                selected_terminated = raw_env.reset_terminated[selected].bool().detach().cpu().tolist()
                selected_action_l2 = (
                    (action_l2_sum[selected] / episode_lengths[selected].clamp(min=1)).detach().cpu().tolist()
                )
                selected_max_action = max_abs_action[selected].detach().cpu().tolist()
                diagnostic_cpu = {key: value[:count].detach().cpu() for key, value in diagnostics.items()}
                for row in range(count):
                    record = {
                        "episode": len(records) + 1,
                        "env_id": int(selected_ids[row]),
                        "return": float(selected_returns[row]),
                        "length": int(selected_lengths[row]),
                        "success": bool(selected_successes[row]),
                        "time_out": bool(selected_timeouts[row]),
                        "terminated": bool(selected_terminated[row]),
                        "mean_action_l2": float(selected_action_l2[row]),
                        "max_abs_action": float(selected_max_action[row]),
                    }
                    for subtask in range(diagnostic_cpu["ever_subtasks"].shape[1]):
                        record[f"subtask_{subtask + 1}_ever_success"] = bool(
                            diagnostic_cpu["ever_subtasks"][row, subtask].item()
                        )
                        record[f"subtask_{subtask + 1}_final_success"] = bool(
                            diagnostic_cpu["final_subtasks"][row, subtask].item()
                        )
                    for key, values in diagnostic_cpu.items():
                        if key not in {"success", "ever_subtasks", "final_subtasks"}:
                            record[key] = float(values[row].item())
                    records.append(record)
                episode_returns[done_ids] = 0.0
                episode_lengths[done_ids] = 0
                action_l2_sum[done_ids] = 0.0
                max_abs_action[done_ids] = 0.0

            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn = (
                    runner.alg.policy
                    if version.parse(installed_version) >= version.parse("2.3.0")
                    else runner.alg.actor_critic
                )
                policy_nn.reset(dones)
            if lift_capture is not None:
                lift_capture.observe_current()

    if len(records) != args_cli.episodes:
        raise RuntimeError(f"Simulator stopped after {len(records)} of {args_cli.episodes} requested episodes.")
    _assert_curriculum_off(raw_env)

    def mean(field: str) -> float:
        return sum(float(record[field]) for record in records) / len(records)

    common_summary = {
        "timeout_rate": mean("time_out"),
        "terminated_rate": mean("terminated"),
        "mean_action_l2": mean("mean_action_l2"),
        "mean_episode_max_abs_action": mean("max_abs_action"),
    }
    if is_lift:
        subtask_names = ("reach_10cm", "grasp", "lift_10cm", "near_goal_5cm")
        additional_summary = {
            **common_summary,
            "lift_success_rate": mean("success"),
            "mean_final_ee_object_distance_m": mean("final_ee_object_distance_m"),
            "mean_min_ee_object_distance_m": mean("min_ee_object_distance_m"),
            "mean_final_goal_distance_m": mean("final_goal_distance_m"),
            "mean_min_goal_distance_m": mean("min_goal_distance_m"),
            "mean_final_object_height_m": mean("final_object_height_m"),
            "mean_max_object_height_m": mean("max_object_height_m"),
            "mean_final_gripper_opening_m": mean("final_gripper_opening_m"),
        }
        metric_definitions = {
            "lift_success_rate": "Fraction ending within 0.02 m of the commanded object goal.",
            "subtask_rates": (
                "Ever rates mean the milestone occurred at least once; final rates mean it held at episode end."
            ),
            "action_metrics": "Policy actions after any RSL-RL wrapper clipping and before task-level processing.",
            "subtask_1": "End effector comes within 0.10 m of the cube.",
            "subtask_2": "End effector is within 0.03 m while gripper opening is in (0.005, 0.035) m.",
            "subtask_3": "Cube world-frame height exceeds 0.10 m.",
            "subtask_4": "Cube comes within 0.05 m of the commanded goal.",
        }
    else:
        subtask_names = ("reach_20cm", "reach_10cm", "open_20cm", "open_38cm")
        additional_summary = {
            **common_summary,
            "success_rate_margin": mean("final_success_margin"),
            "mean_max_success_margin": mean("max_success_margin"),
            "mean_final_drawer_position_m": mean("final_drawer_position_m"),
            "mean_max_drawer_position_m": mean("max_drawer_position_m"),
            "mean_final_gripper_handle_distance_m": mean("final_gripper_handle_distance_m"),
            "mean_min_gripper_handle_distance_m": mean("min_gripper_handle_distance_m"),
        }
        metric_definitions = {
            "success_rate_margin": (
                "Mean terminal clamp(drawer_position / 0.39, 0, 1), matching dones/success_rate_margin."
            ),
            "success_rate": "Secondary binary rate: drawer position exceeds 0.39 m.",
            "subtask_rates": (
                "Ever rates mean the milestone occurred at least once; final rates mean it held at episode end."
            ),
            "action_metrics": "Policy actions after any RSL-RL wrapper clipping and before task-level processing.",
            "subtask_1": "Gripper-handle distance is below 0.20 m.",
            "subtask_2": "Gripper-handle distance is below 0.10 m.",
            "subtask_3": "Drawer position exceeds 0.20 m.",
            "subtask_4": "Drawer position exceeds 0.38 m.",
        }
    for index, name in enumerate(subtask_names, start=1):
        additional_summary[f"subtask_{index}_{name}_ever_success_rate"] = mean(f"subtask_{index}_ever_success")
        additional_summary[f"subtask_{index}_{name}_final_success_rate"] = mean(f"subtask_{index}_final_success")

    result = write_results(
        summary_path=summary_path,
        episodes_path=episodes_path,
        checkpoint=checkpoint,
        task=args_cli.task,
        framework="rsl_rl",
        train_seed=args_cli.train_seed,
        evaluation_seed=args_cli.seed,
        num_envs=args_cli.num_envs,
        requested_episodes=args_cli.episodes,
        deterministic=True,
        success_definition=(
            "Object center is within 0.02 m of the commanded goal at the task-defined episode reset."
            if is_lift
            else "Drawer joint position exceeds 0.39 m (the task's original successful termination)."
        ),
        records=records,
        invocation=original_argv,
        additional_summary=additional_summary,
        metric_definitions=metric_definitions,
    )
    print(f"[INFO] Wrote {summary_path}")
    primary_name = "lift_success_rate" if is_lift else "success_rate_margin"
    print(f"[RESULT] {primary_name}={result[primary_name]:.6f} episodes={result['completed_episodes']}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
