# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a final TSCL Factory Nut-Thread checkpoint on curriculum-free resets."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from eval_utils import output_paths, refuse_overwrite, write_results

from isaaclab.app import AppLauncher

original_argv = list(sys.argv)
parser = argparse.ArgumentParser(description="Fixed-budget TSCL evaluation for Factory Nut-Thread.")
parser.add_argument(
    "--task", default="Isaac-Factory-NutThread-Direct-v0", choices=("Isaac-Factory-NutThread-Direct-v0",)
)
parser.add_argument("--agent", default="rl_games_cfg_entry_point")
parser.add_argument("--checkpoint", required=True, help="Exact final epoch-200 checkpoint to evaluate.")
parser.add_argument("--train_seed", required=True, type=int, help="Training seed represented by the checkpoint.")
parser.add_argument("--seed", type=int, default=12345, help="Common evaluation-environment seed.")
parser.add_argument("--num_envs", type=int, default=512, help="Parallel evaluation environments.")
parser.add_argument("--episodes", type=int, default=10240, help="Exact number of completed episodes to record.")
parser.add_argument("--output_dir", required=True, help="Directory for JSON summary and episode CSV.")
parser.add_argument("--overwrite", action="store_true", help="Replace an existing result for this training seed.")
parser.add_argument(
    "--deterministic", action="store_true", help="Use mean actions instead of the saved player setting."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

summary_path, episodes_path = output_paths(args_cli.output_dir, args_cli.train_seed)
refuse_overwrite(summary_path, episodes_path, args_cli.overwrite)
windows_path = summary_path.with_name(f"seed_{args_cli.train_seed}_dones_success_windows.csv")
if windows_path.exists() and not args_cli.overwrite:
    raise FileExistsError(f"Evaluation output already exists; pass --overwrite to replace it:\n  {windows_path}")
if args_cli.episodes <= 0 or args_cli.num_envs <= 0:
    parser.error("--episodes and --num_envs must both be positive")

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim starts."""

import csv
import math
import statistics
import types
from pathlib import Path

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

import isaacsim.core.utils.torch as torch_utils

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.utils.hydra import hydra_task_config


class _FactoryMetricCapture:
    """Capture Factory's original per-step success signal and terminal diagnostics."""

    def __init__(self, env):
        self.env = env
        device = env.device
        self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        self.ever_subtasks = torch.zeros(env.num_envs, 3, dtype=torch.bool, device=device)
        self.min_xy_error = torch.full((env.num_envs,), torch.inf, device=device)
        self.min_z_displacement = torch.full((env.num_envs,), torch.inf, device=device)
        self.min_wrapped_yaw = torch.full((env.num_envs,), torch.inf, device=device)
        self.final_subtasks = torch.zeros(env.num_envs, 3, dtype=torch.bool, device=device)
        self.final_xy_error = torch.zeros(env.num_envs, device=device)
        self.final_z_displacement = torch.zeros(env.num_envs, device=device)
        self.final_wrapped_yaw = torch.zeros(env.num_envs, device=device)
        self.terminal_success = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        self.episode_success_steps = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        self.step_success_rates: list[float] = []

    def observe_success_step(self) -> None:
        """Record the exact signal exported by Factory as ``dones/success_rate``."""
        if bool(self.env.cfg.reset_state_curriculum_enabled):
            natural_mask = ~self.env.is_curriculum_episode
            success_for_metric = (
                self.env.overall_success[natural_mask] if bool(natural_mask.any().item()) else self.env.overall_success
            )
        else:
            success_for_metric = self.env.overall_success
        self.step_success_rates.append(float(success_for_metric.float().mean().item()))
        self.episode_success_steps += self.env.overall_success.long()

    def _current(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        held_pos, _ = factory_utils.get_held_base_pose(
            self.env.held_pos,
            self.env.held_quat,
            self.env.cfg_task.name,
            self.env.cfg_task.fixed_asset_cfg,
            self.env.num_envs,
            self.env.device,
        )
        target_pos, _ = factory_utils.get_target_held_base_pose(
            self.env.fixed_pos,
            self.env.fixed_quat,
            self.env.cfg_task.name,
            self.env.cfg_task.fixed_asset_cfg,
            self.env.num_envs,
            self.env.device,
        )
        xy_error = torch.linalg.vector_norm(target_pos[env_ids, :2] - held_pos[env_ids, :2], dim=1)
        z_displacement = held_pos[env_ids, 2] - target_pos[env_ids, 2]
        _, _, yaw = torch_utils.get_euler_xyz(self.env.fingertip_midpoint_quat[env_ids])
        yaw = factory_utils.wrap_yaw(yaw)
        subtasks = self.env._get_subtasks()[env_ids]
        return {
            "subtasks": subtasks,
            "xy_error": xy_error,
            "z_displacement": z_displacement,
            "wrapped_yaw": yaw,
        }

    def observe_current(self, env_ids=None) -> None:
        ids = (
            torch.arange(self.env.num_envs, device=self.env.device)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)
        )
        current = self._current(ids)
        self.ever_subtasks[ids] |= current["subtasks"]
        self.min_xy_error[ids] = torch.minimum(self.min_xy_error[ids], current["xy_error"])
        self.min_z_displacement[ids] = torch.minimum(self.min_z_displacement[ids], current["z_displacement"])
        self.min_wrapped_yaw[ids] = torch.minimum(self.min_wrapped_yaw[ids], current["wrapped_yaw"])

    def capture_terminal(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.observe_current(env_ids)
        current = self._current(env_ids)
        self.final_subtasks[env_ids] = current["subtasks"]
        self.final_xy_error[env_ids] = current["xy_error"]
        self.final_z_displacement[env_ids] = current["z_displacement"]
        self.final_wrapped_yaw[env_ids] = current["wrapped_yaw"]
        self.terminal_success[env_ids] = self.env.overall_success[env_ids].bool()
        self.valid[env_ids] = True

    def take(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if not bool(torch.all(self.valid[env_ids]).item()):
            raise RuntimeError("Factory metric capture did not run for every completed episode.")
        values = {
            "success": self.terminal_success[env_ids].clone(),
            "final_subtasks": self.final_subtasks[env_ids].clone(),
            "ever_subtasks": self.ever_subtasks[env_ids].clone(),
            "final_xy_error_m": self.final_xy_error[env_ids].clone(),
            "min_xy_error_m": self.min_xy_error[env_ids].clone(),
            "final_z_displacement_m": self.final_z_displacement[env_ids].clone(),
            "min_z_displacement_m": self.min_z_displacement[env_ids].clone(),
            "final_wrapped_yaw_rad": self.final_wrapped_yaw[env_ids].clone(),
            "min_wrapped_yaw_rad": self.min_wrapped_yaw[env_ids].clone(),
            "success_steps": self.episode_success_steps[env_ids].clone(),
        }
        self.valid[env_ids] = False
        self.ever_subtasks[env_ids] = False
        self.min_xy_error[env_ids] = torch.inf
        self.min_z_displacement[env_ids] = torch.inf
        self.min_wrapped_yaw[env_ids] = torch.inf
        self.episode_success_steps[env_ids] = 0
        return values


def _install_factory_capture(raw_env) -> _FactoryMetricCapture:
    capture = _FactoryMetricCapture(raw_env)
    original_rewards = raw_env._get_rewards

    def wrapped_rewards(_env):
        rewards = original_rewards()
        # _get_dones has already updated overall_success and the TensorBoard
        # dones/success_rate value for this simulation step.
        capture.observe_success_step()
        capture.observe_current()
        capture.capture_terminal(torch.nonzero(_env.reset_buf, as_tuple=False).squeeze(-1))
        return rewards

    raw_env._get_rewards = types.MethodType(wrapped_rewards, raw_env)
    return capture


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.reset_state_curriculum_enabled = False
    env_cfg.reset_curriculum_mode = "aces"
    env_cfg.log_dir = str(Path(args_cli.output_dir).resolve())
    agent_cfg["params"]["seed"] = args_cli.seed

    checkpoint = retrieve_file_path(args_cli.checkpoint)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concatenate = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    raw_env = env.unwrapped
    metric_capture = _install_factory_capture(raw_env)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concatenate)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoint
    agent_cfg["params"]["config"]["num_actors"] = raw_env.num_envs
    if args_cli.deterministic:
        agent_cfg["params"]["config"].setdefault("player", {})["deterministic"] = True

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(checkpoint)
    agent.reset()

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()
    metric_capture.observe_current()
    if bool(raw_env.cfg.reset_state_curriculum_enabled):
        raise RuntimeError("Evaluation requires reset_state_curriculum_enabled=False.")
    if bool(raw_env.is_curriculum_episode.any().item()):
        raise RuntimeError("A curriculum reset was observed during evaluation.")

    episode_returns = torch.zeros(raw_env.num_envs, device=rl_device)
    episode_lengths = torch.zeros(raw_env.num_envs, dtype=torch.long, device=rl_device)
    action_l2_sum = torch.zeros(raw_env.num_envs, device=rl_device)
    max_abs_action = torch.zeros(raw_env.num_envs, device=rl_device)
    records: list[dict[str, int | float | bool]] = []

    while simulation_app.is_running() and len(records) < args_cli.episodes:
        with torch.inference_mode():
            agent_obs = agent.obs_to_torch(obs)
            actions = agent.get_action(agent_obs, is_deterministic=agent.is_deterministic)
            applied_actions = torch.clamp(actions, -clip_actions, clip_actions)
            action_l2_sum += torch.linalg.vector_norm(applied_actions, dim=-1)
            max_abs_action = torch.maximum(max_abs_action, applied_actions.abs().amax(dim=-1))
            obs, rewards, dones, _ = env.step(actions)
            episode_returns += rewards
            episode_lengths += 1

            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                diagnostics = metric_capture.take(done_ids.to(device=raw_env.device))
                diagnostics = {key: value.to(device=rl_device) for key, value in diagnostics.items()}
                success_all = diagnostics["success"]
                count = min(done_ids.numel(), args_cli.episodes - len(records))
                selected = done_ids[:count]
                selected_returns = episode_returns[selected].detach().cpu().tolist()
                selected_lengths = episode_lengths[selected].detach().cpu().tolist()
                selected_ids = selected.detach().cpu().tolist()
                selected_successes = success_all[:count].detach().cpu().tolist()
                selected_action_l2 = (
                    (action_l2_sum[selected] / episode_lengths[selected].clamp(min=1)).detach().cpu().tolist()
                )
                selected_max_action = max_abs_action[selected].detach().cpu().tolist()
                diagnostic_cpu = {key: value[:count].detach().cpu() for key, value in diagnostics.items()}
                for row in range(count):
                    episode_length = int(selected_lengths[row])
                    record = {
                        "episode": len(records) + 1,
                        "env_id": int(selected_ids[row]),
                        "return": float(selected_returns[row]),
                        "length": episode_length,
                        "success": bool(selected_successes[row]),
                        "dones_success_fraction": float(diagnostic_cpu["success_steps"][row].item()) / episode_length,
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
                        if key not in {"success", "success_steps", "ever_subtasks", "final_subtasks"}:
                            record[key] = float(values[row].item())
                    records.append(record)
                episode_returns[done_ids] = 0.0
                episode_lengths[done_ids] = 0
                action_l2_sum[done_ids] = 0.0
                max_abs_action[done_ids] = 0.0

                if agent.is_rnn and agent.states is not None:
                    for state in agent.states:
                        state[:, dones, :] = 0.0

    if len(records) != args_cli.episodes:
        raise RuntimeError(f"Simulator stopped after {len(records)} of {args_cli.episodes} requested episodes.")
    if bool(raw_env.is_curriculum_episode.any().item()):
        raise RuntimeError("A curriculum reset was observed during evaluation.")

    def mean(field: str) -> float:
        return sum(float(record[field]) for record in records) / len(records)

    rollout_window_steps = int(agent_cfg["params"]["config"]["horizon_length"])
    step_success_rates = metric_capture.step_success_rates
    if len(step_success_rates) == 0:
        raise RuntimeError("Factory per-step success capture produced no samples.")
    window_records = []
    for start in range(0, len(step_success_rates), rollout_window_steps):
        values = step_success_rates[start : start + rollout_window_steps]
        window_records.append(
            {
                "window": len(window_records) + 1,
                "start_step": start + 1,
                "end_step": start + len(values),
                "num_steps": len(values),
                "complete_window": len(values) == rollout_window_steps,
                "dones_success_rate": statistics.fmean(values),
            }
        )
    complete_windows = [row for row in window_records if row["complete_window"]]
    if len(complete_windows) == 0:
        raise RuntimeError(
            f"Evaluation ended before one complete {rollout_window_steps}-step RL-Games-equivalent window."
        )

    windows_path.parent.mkdir(parents=True, exist_ok=True)
    with windows_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(window_records[0]))
        writer.writeheader()
        writer.writerows(window_records)

    additional_summary = {
        # Primary Factory metric: the mean of the same per-step boolean success
        # predicate exported during training. Averaging complete, equal-length
        # episodes makes this independent of the 128-step rollout phase.
        "dones_success_rate": mean("dones_success_fraction"),
        "dones_success_rate_all_env_steps": statistics.fmean(step_success_rates),
        "dones_success_rate_complete_128_step_windows": statistics.fmean(
            float(row["dones_success_rate"]) for row in complete_windows
        ),
        "dones_success_rate_last_complete_128_step_window": float(complete_windows[-1]["dones_success_rate"]),
        "dones_success_rate_min_complete_128_step_window": min(
            float(row["dones_success_rate"]) for row in complete_windows
        ),
        "dones_success_rate_max_complete_128_step_window": max(
            float(row["dones_success_rate"]) for row in complete_windows
        ),
        "terminal_success_rate": mean("success"),
        "dones_success_window_steps": rollout_window_steps,
        "dones_success_complete_windows": len(complete_windows),
        "dones_success_total_simulation_steps": len(step_success_rates),
        "dones_success_windows_file": str(windows_path.resolve()),
        "mean_action_l2": mean("mean_action_l2"),
        "mean_episode_max_abs_action": mean("max_abs_action"),
        "mean_final_xy_error_m": mean("final_xy_error_m"),
        "mean_min_xy_error_m": mean("min_xy_error_m"),
        "mean_final_z_displacement_m": mean("final_z_displacement_m"),
        "mean_min_z_displacement_m": mean("min_z_displacement_m"),
        "mean_final_wrapped_yaw_rad": mean("final_wrapped_yaw_rad"),
        "mean_min_wrapped_yaw_rad": mean("min_wrapped_yaw_rad"),
    }
    subtask_names = ("aligned", "engaged", "threaded")
    for index, name in enumerate(subtask_names, start=1):
        additional_summary[f"subtask_{index}_{name}_ever_success_rate"] = mean(f"subtask_{index}_ever_success")
        additional_summary[f"subtask_{index}_{name}_final_success_rate"] = mean(f"subtask_{index}_final_success")
    metric_definitions = {
        "dones_success_rate": (
            "Mean fraction of episode steps satisfying Factory's original overall_success predicate. This is the "
            "phase-independent aggregate of the same per-step signal logged during training as dones/success_rate."
        ),
        "dones_success_rate_all_env_steps": (
            "Mean of the environment-wide dones/success_rate captured on every evaluation simulation step."
        ),
        "dones_success_rate_complete_128_step_windows": (
            "Mean across complete RL-Games-equivalent 128-step windows; see dones_success_windows_file for the "
            "spiking time series."
        ),
        "terminal_success_rate": "Fraction of recorded episodes satisfying overall_success at timeout.",
        "subtask_rates": "Ever rates mean the milestone occurred at least once; final rates mean it held at timeout.",
        "action_metrics": "Policy actions after RL-Games wrapper clipping and before task-level processing.",
        "subtask_1": "Nut is aligned: XY error < 0.001 m and wrapped yaw is below ee_success_yaw.",
        "subtask_2": "Nut satisfies the configured engage_threshold without the rotation requirement.",
        "subtask_3": "Nut satisfies the configured success_threshold including rotation.",
        "z_displacement_m": "Held-base Z minus target held-base Z; smaller values represent deeper threading.",
    }

    result = write_results(
        summary_path=summary_path,
        episodes_path=episodes_path,
        checkpoint=checkpoint,
        task=args_cli.task,
        framework="rl_games",
        train_seed=args_cli.train_seed,
        evaluation_seed=args_cli.seed,
        num_envs=args_cli.num_envs,
        requested_episodes=args_cli.episodes,
        deterministic=bool(agent.is_deterministic),
        success_definition=(
            "Factory's original _get_curr_successes criterion at episode timeout, using the configured "
            "nut-thread position and rotation thresholds."
        ),
        records=records,
        invocation=original_argv,
        additional_summary=additional_summary,
        metric_definitions=metric_definitions,
    )
    print(f"[INFO] Wrote {summary_path}")
    print(
        f"[RESULT] dones_success_rate={result['dones_success_rate']:.6f} "
        f"terminal_success_rate={result['terminal_success_rate']:.6f} "
        f"episodes={result['completed_episodes']}"
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
