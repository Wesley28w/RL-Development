# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline evaluator for the RCG benchmark.

A policy trained with a reverse curriculum is trained on curriculum start states, so its
training-time success rate is not comparable to a PPO baseline that always starts from the
task's own distribution ``rho_0``. This script produces the comparable number: it walks the
checkpoints of a run and measures, for each one, the success rate over complete episodes started
from ``rho_0`` with the curriculum switched off.

Point it at both arms of the benchmark and the two CSVs are directly comparable.

Usage:

.. code-block:: powershell

    # note: ` is PowerShell's line continuation, and nothing may follow it on the line
    isaaclab.bat -p scripts/rcg/evaluate.py --task Isaac-Franka-Cabinet-RCG-Direct-v0 `
        --run_dir logs/rsl_rl/franka_cabinet_rcg/<run> --episodes 512 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import glob
import os
import re
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate checkpoints on the task's own start distribution.")
parser.add_argument("--task", type=str, required=True, help="Name of the task the run was trained on.")
parser.add_argument("--run_dir", type=str, required=True, help="Run directory containing model_*.pt checkpoints.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to simulate.")
parser.add_argument("--episodes", type=int, default=512, help="Complete episodes to evaluate per checkpoint.")
parser.add_argument("--every", type=int, default=1, help="Evaluate every n-th checkpoint.")
parser.add_argument("--output", type=str, default=None, help="Output CSV path. Defaults to <run_dir>/rho0_eval.csv.")
parser.add_argument("--seed", type=int, default=12345, help="Seed for the evaluation environment.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg

from isaaclab_rl.rsl_rl import RCGOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def _iteration_of(path: str) -> int:
    match = re.search(r"model_(\d+)\.pt$", os.path.basename(path))
    return int(match.group(1)) if match else -1


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Measure the rho_0 success rate of every checkpoint in a run."""
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    # the whole point of this script: evaluate on the task's own start distribution
    if hasattr(env_cfg, "rcg"):
        env_cfg.rcg.enabled = False

    run_dir = os.path.abspath(args_cli.run_dir)
    checkpoints = sorted(glob.glob(os.path.join(run_dir, "model_*.pt")), key=_iteration_of)
    checkpoints = [path for path in checkpoints if _iteration_of(path) >= 0][:: max(1, args_cli.every)]
    if not checkpoints:
        raise FileNotFoundError(f"No 'model_*.pt' checkpoints found in '{run_dir}'.")
    output_path = args_cli.output or os.path.join(run_dir, "rho0_eval.csv")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped

    runner_cls = RCGOnPolicyRunner if agent_cfg.class_name == "RCGOnPolicyRunner" else OnPolicyRunner
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    print(f"[INFO] Evaluating {len(checkpoints)} checkpoint(s) over >= {args_cli.episodes} episodes each.")
    rows = []
    for path in checkpoints:
        runner.load(path)
        policy = runner.get_inference_policy(device=base_env.device)
        result = _evaluate(env, base_env, policy, agent_cfg.device)
        result["iteration"] = _iteration_of(path)
        result["checkpoint"] = os.path.basename(path)
        rows.append(result)
        print(
            f"[INFO] {result['checkpoint']:<20} success_rate {result['success_rate']:.4f}"
            f"  episodes {result['episodes']}  mean_len {result['mean_episode_length']:.1f}"
        )

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["iteration", "checkpoint", "episodes", "success_rate", "mean_episode_length"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[INFO] Wrote {len(rows)} row(s) to {output_path}")

    env.close()


def _evaluate(env, base_env, policy, device) -> dict:
    """Run complete episodes from rho_0 and return the fraction that reached the goal."""
    env.reset()
    obs = env.get_observations().to(device)

    episodes = 0
    successes = 0
    total_length = 0
    # cap the wall time: even a policy that never terminates early finishes an episode every
    # max_episode_length steps, so this bound is always reachable
    max_steps = int(base_env.max_episode_length * (args_cli.episodes / base_env.num_envs + 2))

    with torch.inference_mode():
        for _ in range(max_steps):
            if episodes >= args_cli.episodes:
                break
            actions = policy(obs)
            # episode_length_buf is read before the step, since terminated environments are
            # reset inside env.step() and their counter is back to zero afterwards
            lengths = base_env.episode_length_buf.clone()
            obs, _, dones, _ = env.step(actions.to(env.device))
            obs = obs.to(device)

            finished = dones.bool()
            num_finished = int(finished.sum().item())
            if num_finished:
                episodes += num_finished
                successes += int(base_env.reset_terminated[finished].sum().item())
                total_length += int(lengths[finished].sum().item()) + num_finished

    return {
        "episodes": episodes,
        "success_rate": successes / episodes if episodes else float("nan"),
        "mean_episode_length": total_length / episodes if episodes else float("nan"),
    }


if __name__ == "__main__":
    main()
    simulation_app.close()
