# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record goal states for Reverse Curriculum Generation.

RCG assumes that a single state in which the task is achieved, ``s^g``, is given as prior
knowledge (Florensa et al., CoRL 2017). This script produces that prior by rolling out a
trained policy on the *baseline* task and snapshotting the simulator state at the exact instant
each environment succeeds.

Why a trained policy rather than a hand-built state: the informative curriculum dimension for
Franka Cabinet is the arm configuration, not the drawer opening. A state with the drawer open
but the gripper parked at its default pose is physically valid yet useless as a goal, because
expanding backwards from it never produces starts in which the gripper is near the handle. A
partially trained checkpoint is enough -- only a few hundred success states are needed.

Usage:

.. code-block:: powershell

    # note: ` is PowerShell's line continuation, and nothing may follow it on the line
    isaaclab.bat -p scripts/rcg/record_goal_states.py --task Isaac-Franka-Cabinet-Direct-v0 `
        --checkpoint logs/rsl_rl/franka_cabinet_direct/<run>/model_1499.pt --num_states 1000 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record RCG goal states from a trained policy.")
parser.add_argument("--task", type=str, default="Isaac-Franka-Cabinet-Direct-v0", help="Name of the baseline task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint.")
parser.add_argument("--num_states", type=int, default=1000, help="Number of goal states to record.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to simulate.")
parser.add_argument("--output", type=str, default=None, help="Output .pt path. Defaults to the task's data directory.")
parser.add_argument("--max_steps", type=int, default=5000, help="Give up after this many environment steps.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
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
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RCGOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Roll out a trained policy and snapshot every state in which the task is achieved."""
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    # record from the task's own start distribution, never from a curriculum pool
    if not hasattr(env_cfg, "rcg"):
        raise ValueError(f"Task '{args_cli.task}' does not support RCG: its configuration has no 'rcg' field.")
    env_cfg.rcg.enabled = False

    resume_path = retrieve_file_path(args_cli.checkpoint)
    output_path = args_cli.output
    if output_path is None:
        from isaaclab_tasks.direct.franka_cabinet.franka_cabinet_env import DEFAULT_GOAL_STATE_PATH

        output_path = DEFAULT_GOAL_STATE_PATH
    output_path = os.path.abspath(output_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped

    runner_cls = RCGOnPolicyRunner if agent_cfg.class_name == "RCGOnPolicyRunner" else OnPolicyRunner
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=base_env.device)

    # the environment snapshots success states from inside _reset_idx, which is the only moment
    # at which they still exist -- env.step() resets terminated environments before returning
    base_env.start_goal_state_recording(args_cli.num_states)

    obs = env.get_observations().to(agent_cfg.device)
    steps = 0
    with torch.inference_mode():
        while base_env.recorded_goal_state_count < args_cli.num_states and steps < args_cli.max_steps:
            actions = policy(obs)
            obs, _, _, _ = env.step(actions.to(env.device))
            obs = obs.to(agent_cfg.device)
            steps += 1
            if steps % 100 == 0:
                print(f"[INFO] step {steps}: {base_env.recorded_goal_state_count}/{args_cli.num_states} goal states")

    pool = base_env.collect_recorded_goal_states()
    num_recorded = base_env._pool_size(pool)
    if num_recorded == 0:
        raise RuntimeError(
            f"No success states recorded in {steps} steps. The checkpoint at '{resume_path}' never solves the task;"
            " train the baseline for longer before recording goal states."
        )

    payload = {
        "pool": {key: value.cpu() for key, value in pool.items()},
        "meta": {
            "task": args_cli.task,
            "checkpoint": resume_path,
            "num_states": int(num_recorded),
            "num_envs": int(args_cli.num_envs),
            "seed": int(args_cli.seed),
            "steps": int(steps),
        },
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)

    print(f"\n[INFO] Recorded {num_recorded} goal state(s) in {steps} steps -> {output_path}")
    if num_recorded < args_cli.num_states:
        # the success rate implied by the shortfall is worth knowing before reading any benchmark
        # number, so report it rather than quietly writing a smaller file than was asked for
        episodes = steps * args_cli.num_envs / max(1, base_env.max_episode_length)
        print(
            f"[WARNING] Asked for {args_cli.num_states} goal states but only reached {num_recorded} before the"
            f" --max_steps limit of {args_cli.max_steps}. That implies this checkpoint solves roughly"
            f" {100.0 * num_recorded / max(1.0, episodes):.1f}% of episodes from the task's own start"
            " distribution. RCG only needs a handful of goal states, so this file is still usable; raise"
            " --max_steps or --num_envs to collect more."
        )
    _describe(base_env, pool)

    env.close()


def _describe(base_env, pool) -> None:
    """Print the spread of the recorded set, so a degenerate recording is obvious immediately."""
    drawer = pool["cabinet_joint_pos"][:, base_env.drawer_joint_idx]
    print("[INFO] Recorded goal states:")
    print(f"       drawer_top_joint : mean {drawer.mean():.4f}  min {drawer.min():.4f}  max {drawer.max():.4f}")
    print(f"       success threshold: {base_env.cfg.drawer_open_threshold}")
    joint_std = pool["robot_joint_pos"].std(dim=0)
    print(f"       arm joint std    : {[f'{v:.3f}' for v in joint_std.tolist()]}")
    if float(joint_std.max()) < 1e-3:
        print(
            "[WARNING] All recorded goal states share essentially the same arm configuration. The curriculum will"
            " expand from a single pose; consider recording with more environments or a noisier policy."
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
