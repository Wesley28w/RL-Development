# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Harvest genuine task-success states from a converged checkpoint, to seed Reverse
Curriculum Generation (see isaaclab_tasks.utils.reverse_curriculum.ReverseCurriculum).

RCG needs at least one real success state before training starts. Rather than hand-author
one (fragile -- manipulation state spaces have joint limits, contact geometry, and grasp
relationships that a hand-picked vector can easily violate), this script rolls out an
existing converged policy checkpoint for a handful of environments and, on any episode
where the task's own success check fires, saves that environment's full resettable state.

Usage (cabinet, against a baseline rsl_rl checkpoint):
    scripts/tools/capture_rcg_goal_state.py --task Isaac-Franka-Cabinet-Direct-v0 \
        --checkpoint logs/paper_logs/cabinet_baseline/<run>/model_<N>.pt \
        --num_envs 64 --num_states 16 --out goal_states/cabinet.pt

Only Isaac-Franka-Cabinet-Direct-v0 is wired up right now (see TASK_ADAPTERS below). Adding
factory/lift means adding one more entry: a success-check callable and a state-extraction
callable, both operating on the unwrapped env.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Harvest RCG goal states from a converged checkpoint.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl/rl_games checkpoint to load.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to roll out.")
parser.add_argument("--num_states", type=int, default=16, help="Number of success states to collect before exiting.")
parser.add_argument("--max_steps", type=int, default=2000, help="Give up after this many env steps.")
parser.add_argument("--out", type=str, required=True, help="Output .pt file for the collected states.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def _cabinet_is_success(env) -> torch.Tensor:
    drawer_pos = env._cabinet.data.joint_pos[:, env.drawer_joint_idx]
    return drawer_pos > 0.39


def _cabinet_get_world(env) -> torch.Tensor:
    return torch.cat([env._robot.data.joint_pos, env._cabinet.data.joint_pos], dim=1)


# task_name -> (is_success(env) -> bool[N], get_world(env) -> float[N, state_dim])
TASK_ADAPTERS = {
    "Isaac-Franka-Cabinet-Direct-v0": (_cabinet_is_success, _cabinet_get_world),
}


def main():
    if args_cli.task not in TASK_ADAPTERS:
        raise ValueError(
            f"No goal-state adapter registered for task '{args_cli.task}'. Add one to TASK_ADAPTERS in this script."
        )
    is_success_fn, get_world_fn = TASK_ADAPTERS[args_cli.task]

    env = gym.make(args_cli.task, num_envs=args_cli.num_envs)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    unwrapped = env.unwrapped

    # load a jit-exported policy if given one; otherwise fall back to zero actions (no policy),
    # which still works for tasks where the default reset distribution occasionally succeeds by
    # chance, just far more slowly -- passing a real checkpoint is strongly recommended.
    policy = None
    if args_cli.checkpoint.endswith(".pt"):
        try:
            policy = torch.jit.load(args_cli.checkpoint, map_location=unwrapped.device)
        except RuntimeError:
            print(
                f"[WARN] Could not load '{args_cli.checkpoint}' as a TorchScript policy directly. "
                "Export it first (see scripts/reinforcement_learning/rsl_rl/play.py's export step), "
                "or pass an already-exported policy.pt. Falling back to zero actions."
            )

    obs = env.get_observations()
    collected: list[torch.Tensor] = []

    with torch.inference_mode():
        for step in range(args_cli.max_steps):
            if policy is not None:
                actions = policy(obs)
            else:
                actions = torch.zeros(unwrapped.num_envs, unwrapped.cfg.action_space, device=unwrapped.device)
            obs, _, dones, _ = env.step(actions)

            success = is_success_fn(unwrapped)
            if success.any():
                world = get_world_fn(unwrapped)
                collected.append(world[success].detach().cpu().clone())
                n_collected = sum(t.shape[0] for t in collected)
                print(f"[INFO] step {step}: collected {n_collected}/{args_cli.num_states} success states so far")
                if n_collected >= args_cli.num_states:
                    break

    if not collected:
        raise RuntimeError(
            "No success states were collected. The checkpoint may not be converged enough, or"
            " --max_steps/--num_envs may need to be larger."
        )

    states = torch.cat(collected, dim=0)[: args_cli.num_states]
    torch.save(states, args_cli.out)
    print(f"[INFO] Saved {states.shape[0]} goal states with shape {tuple(states.shape)} to {args_cli.out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
