# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lift report for Franka Lift checkpoints, evaluated from rho_0.

The counterpart of ``quick_drawer_eval.py``. Any curriculum is switched off, so every checkpoint is
measured from the task's own start distribution -- which is the only way a curriculum arm's number
means the same thing as a baseline arm's.

One episode per environment: accumulation for an environment stops the moment its episode ends, so
an early success cannot leak into the next episode's statistics.

Metrics, per episode then averaged:

* ``mean_dist``     -- timestep-averaged object-to-goal distance, in metres.
* ``min_dist``      -- closest the object ever got to the goal. The single most informative number
                       here: it separates "never went near the goal" from "got there and lost it".
* ``terminal_dist`` -- object-to-goal distance at the episode's last observed step.
* ``success``       -- fraction of episodes that *ended* with the object at the goal. The primary
                       metric, and the same quantity as the ``dones/success_rate`` logged during
                       training.
* ``success_ever``  -- fraction that reached the goal at any point. Saturates for any competent
                       policy, so it is a yes/no that the task is being solved, not a comparison axis.
* ``solved_frac``   -- fraction of steps spent within the success threshold, i.e. how long the
                       policy holds the object there once it arrives.
* ``mean_height``   -- timestep-averaged object height; separates "cannot grasp" from "cannot carry".
* ``lifted``        -- fraction of episodes in which the object ever rose above ``--lift_height``.
* ``reach_p``       -- fraction of episodes that ever came within ``--progress_threshold`` of the goal.
* ``steps_to_p``    -- steps to first come that close, over the episodes that did.
* ``dropped``       -- fraction of episodes that ended early, i.e. by dropping the object.
* ``ep_len``        -- mean episode length; a sanity check (250 = nothing terminated early).

``success`` is read from ``rcg_last_episode_success``, which the environment latches from
``_reset_idx`` while the final step's state is still readable. Everything else is read straight off
the scene, pre-step, because ``env.step()`` resets finished environments before it returns and a
post-step read would report the *next* episode's state for exactly those environments.

Usage:

.. code-block:: powershell

    # every seed of both arms, with a CSV and per-group mean +/- sd
    isaaclab.bat -p scripts/rcg/quick_lift_eval.py `
        --run_glob "logs/rsl_rl/franka_lift_baseline/*" "logs/rsl_rl/franka_lift_rcg/*" --output lift_eval.csv

    # or name checkpoints explicitly
    isaaclab.bat -p scripts/rcg/quick_lift_eval.py --checkpoints <ck1> <ck2> --labels baseline rcg
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Franka Lift report from rho_0.")
parser.add_argument("--run_glob", type=str, nargs="*", default=None, help="Glob(s) matching run directories.")
parser.add_argument("--checkpoint_name", type=str, default="model_1499.pt", help="Checkpoint file within each run.")
parser.add_argument("--checkpoints", type=str, nargs="*", default=None, help="Explicit checkpoint paths instead.")
parser.add_argument("--labels", type=str, nargs="*", default=None, help="Optional label per explicit checkpoint.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-RCG-v0", help="Task to instantiate.")
parser.add_argument("--num_envs", type=int, default=256, help="Environments = episodes per checkpoint.")
parser.add_argument(
    "--success_threshold",
    type=float,
    default=None,
    help="Object-to-goal distance counted as success. Defaults to the task's own 'success_threshold'.",
)
parser.add_argument("--progress_threshold", type=float, default=0.10, help="Distance counted as progress, in metres.")
parser.add_argument("--lift_height", type=float, default=0.10, help="Object height counted as lifted, in metres.")
parser.add_argument("--seed", type=int, default=12345, help="Seed for the evaluation environment.")
parser.add_argument("--output", type=str, default=None, help="Optional CSV output path.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import glob
import importlib.metadata as metadata
import os
import re
import statistics

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.math import combine_frame_transforms

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

NAN = float("nan")
FIELDS = [
    "mean_dist",
    "min_dist",
    "terminal_dist",
    "success",
    "success_ever",
    "solved_frac",
    "mean_height",
    "lifted",
    "reach_p",
    "steps_to_p",
    "dropped",
    "ep_len",
]
HEADLINE = ("success", "min_dist", "success_ever", "steps_to_p")


def _seed_of(run_dir: str) -> str:
    """Read the seed out of a run's dumped agent configuration."""
    path = os.path.join(run_dir, "params", "agent.yaml")
    if os.path.isfile(path):
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                m = re.match(r"^seed:\s*(\S+)", line)
                if m:
                    return m.group(1)
    return "?"


def _collect() -> list[dict]:
    """Resolve the checkpoints to evaluate from either --run_glob or --checkpoints."""
    items = []
    for pattern in args_cli.run_glob or []:
        for d in sorted(glob.glob(pattern)):
            ck = os.path.join(d, args_cli.checkpoint_name)
            if os.path.isfile(ck):
                norm = os.path.normpath(d)
                items.append(
                    {
                        "group": os.path.basename(os.path.dirname(norm)),
                        "run": os.path.basename(norm),
                        "seed": _seed_of(d),
                        "ckpt": ck,
                    }
                )
    labels = list(args_cli.labels or [])
    for i, ck in enumerate(args_cli.checkpoints or []):
        label = labels[i] if i < len(labels) else f"ck{i}"
        items.append({"group": label, "run": os.path.basename(os.path.dirname(ck)), "seed": "?", "ckpt": ck})
    return items


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate every resolved checkpoint in one simulator session."""
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    # every checkpoint is measured from the task's own start distribution
    if hasattr(env_cfg, "rcg"):
        env_cfg.rcg.enabled = False
    if hasattr(env_cfg, "reset_state_curriculum_enabled"):
        env_cfg.reset_state_curriculum_enabled = False
    if args_cli.success_threshold is not None:
        env_cfg.success_threshold = args_cli.success_threshold

    items = _collect()
    if not items:
        raise FileNotFoundError("Nothing to evaluate. Pass --run_glob or --checkpoints.")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base = env.unwrapped
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    threshold = float(base.cfg.success_threshold)
    print(
        f"\n[INFO] {len(items)} checkpoint(s), {args_cli.num_envs} episodes each,"
        f" success = object within {threshold} m of the commanded goal"
    )

    for it in items:
        runner.load(it["ckpt"])
        policy = runner.get_inference_policy(device=base.device)
        it.update(_evaluate(env, base, policy, agent_cfg.device, threshold))
        print(
            f"  {it['group']:<22} {it['run']:<22} seed={it['seed']:<4}"
            f" success={it['success']:.4f} min_dist={it['min_dist']:.4f}"
            f" ever={it['success_ever']:.4f} steps_to_p={it['steps_to_p']:7.1f}"
        )

    _report(items, threshold)

    if args_cli.output:
        keys = ["group", "run", "seed"] + FIELDS
        with open(args_cli.output, "w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=keys)
            w.writeheader()
            for it in items:
                w.writerow({k: it[k] for k in keys})
        print(f"\n[INFO] wrote {len(items)} rows to {os.path.abspath(args_cli.output)}")

    env.close()


def _goal_distance(base) -> torch.Tensor:
    """Object-to-goal distance for every environment, in metres."""
    robot = base.scene["robot"]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, base.command_manager.get_command("object_pose")[:, :3]
    )
    return torch.norm(des_pos_w - base.scene["object"].data.root_pos_w, dim=-1)


def _evaluate(env, base, policy, device, threshold: float) -> dict:
    """Run exactly one episode per environment, freezing each when its episode ends."""
    n = base.num_envs
    dev = base.device
    pthr = args_cli.progress_threshold
    horizon = int(base.max_episode_length)

    active = torch.ones(n, dtype=torch.bool, device=dev)
    dist_sum = torch.zeros(n, device=dev)
    dist_min = torch.full((n,), float("inf"), device=dev)
    height_sum = torch.zeros(n, device=dev)
    solved_steps = torch.zeros(n, device=dev)
    ep_len = torch.zeros(n, device=dev)
    last_dist = torch.zeros(n, device=dev)
    first_p = torch.full((n,), NAN, device=dev)
    ever_lifted = torch.zeros(n, dtype=torch.bool, device=dev)
    ever_solved = torch.zeros(n, dtype=torch.bool, device=dev)
    terminal_solved = torch.zeros(n, dtype=torch.bool, device=dev)
    timed_out = torch.zeros(n, dtype=torch.bool, device=dev)

    with torch.inference_mode():
        env.reset()
        obs = env.get_observations().to(device)
        for t in range(horizon):
            if not bool(active.any()):
                break
            # read before stepping: env.step() resets finished environments before returning
            dist = _goal_distance(base)
            height = base.scene["object"].data.root_pos_w[:, 2] - base.scene.env_origins[:, 2]
            upd = active.float()
            dist_sum += dist * upd
            height_sum += height * upd
            dist_min = torch.where(active, torch.minimum(dist_min, dist), dist_min)
            solved_steps += (dist < threshold).float() * upd
            ep_len += upd
            last_dist = torch.where(active, dist, last_dist)
            ever_lifted |= active & (height > args_cli.lift_height)
            ever_solved |= active & (dist < threshold)
            newly = active & (dist < pthr) & torch.isnan(first_p)
            first_p = torch.where(newly, torch.full_like(first_p, float(t)), first_p)

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions.to(env.device))
            obs = obs.to(device)

            finished = active & dones.bool()
            # the environment latched this from _reset_idx, i.e. from the final step's state, which
            # is the only moment at which it existed
            terminal_solved |= finished & base.rcg_last_episode_success
            timed_out |= finished & base.reset_time_outs
            # freeze this environment: its episode is over and the sim has already reset it
            active &= ~dones.bool()

    # an episode that ended at the goal reached it by definition, even if the pre-step reads never
    # caught the final step
    ever_solved |= terminal_solved
    reached = ~torch.isnan(first_p)
    denom = ep_len.clamp(min=1)
    # every finished episode either timed out or ended early, and ending early on this task means
    # the object fell below the table
    dropped = (~timed_out).float()

    return {
        "mean_dist": (dist_sum / denom).mean().item(),
        "min_dist": dist_min.mean().item(),
        "terminal_dist": last_dist.mean().item(),
        "success": terminal_solved.float().mean().item(),
        "success_ever": ever_solved.float().mean().item(),
        "solved_frac": (solved_steps / denom).mean().item(),
        "mean_height": (height_sum / denom).mean().item(),
        "lifted": ever_lifted.float().mean().item(),
        "reach_p": reached.float().mean().item(),
        "steps_to_p": first_p[reached].mean().item() if bool(reached.any()) else NAN,
        "dropped": dropped.mean().item(),
        "ep_len": ep_len.mean().item(),
    }


def _report(items: list[dict], threshold: float) -> None:
    """Print per-group mean +/- sd across seeds."""
    groups: dict[str, list[dict]] = {}
    for it in items:
        groups.setdefault(it["group"], []).append(it)

    print("\n" + "=" * 104)
    print(f"SUMMARY  (success = {threshold} m to goal, progress = {args_cli.progress_threshold} m)")
    print("=" * 104)
    print(f"{'group':<22} {'n':>3} " + "".join(f"{f:>19}" for f in HEADLINE))
    for g, rs in sorted(groups.items()):
        cells = []
        for key in HEADLINE:
            vals = [r[key] for r in rs if r[key] == r[key]]
            if not vals:
                cells.append("n/a".rjust(19))
                continue
            m = statistics.fmean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{m:.4f}+-{sd:.4f}".rjust(19))
        print(f"{g:<22} {len(rs):>3} " + "".join(cells))


if __name__ == "__main__":
    main()
    simulation_app.close()
