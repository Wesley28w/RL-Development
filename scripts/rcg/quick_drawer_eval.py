# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drawer-opening report for Franka Cabinet checkpoints, evaluated from rho_0.

Self-contained: reads the drawer joint straight off the articulation and uses only
``reset_terminated``, a standard :class:`~isaaclab.envs.DirectRLEnv` buffer, so it runs on any
branch regardless of which diagnostic buffers that branch's environment defines.

Any curriculum is switched off, so every checkpoint is measured from the task's own start
distribution. ``terminate_on_success`` is left alone -- the task is evaluated as configured.

One episode per environment: accumulation for an environment stops the moment its episode ends,
so an early success cannot leak into the next episode's statistics.

Metrics, per episode then averaged:

* ``mean_open``     -- timestep-averaged drawer position within the episode.
* ``margin``        -- ``mean_open / success_threshold``; the same quantity as the
                       ``dones/success_rate_margin`` logged during training.
* ``mean_max``      -- mean of the furthest the drawer ever got.
* ``terminal_open`` -- drawer position at the episode's last observed step.
* ``solved_frac``   -- fraction of steps spent past the success threshold.
* ``success``       -- fraction of episodes that ended by reaching the goal.
* ``reach_p``       -- fraction of episodes that ever passed ``--progress_threshold``.
* ``steps_to_p``    -- steps to first pass it, over the episodes that did.
* ``ep_len``        -- mean episode length; a sanity check (499 = nothing terminated early).

Usage:

.. code-block:: powershell

    # every seed of all three arms, with a CSV and per-group mean +/- sd
    isaaclab.bat -p scripts/rcg/quick_drawer_eval.py `
        --run_glob "logs/rsl_rl/franka_cabinet_rcg/series_one/*" --output rcg_eval.csv

    # or name checkpoints explicitly
    isaaclab.bat -p scripts/rcg/quick_drawer_eval.py --checkpoints <ck1> <ck2> --labels A B
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Drawer-opening report from rho_0.")
parser.add_argument("--run_glob", type=str, nargs="*", default=None, help="Glob(s) matching run directories.")
parser.add_argument("--checkpoint_name", type=str, default="model_2499.pt", help="Checkpoint file within each run.")
parser.add_argument("--checkpoints", type=str, nargs="*", default=None, help="Explicit checkpoint paths instead.")
parser.add_argument("--labels", type=str, nargs="*", default=None, help="Optional label per explicit checkpoint.")
parser.add_argument("--task", type=str, default="Isaac-Franka-Cabinet-Direct-v0", help="Task to instantiate.")
parser.add_argument("--num_envs", type=int, default=256, help="Environments = episodes per checkpoint.")
parser.add_argument("--success_threshold", type=float, default=0.39, help="Drawer opening counted as success.")
parser.add_argument("--progress_threshold", type=float, default=0.35, help="Drawer opening counted as progress.")
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

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

NAN = float("nan")
FIELDS = [
    "mean_open", "margin", "mean_max", "terminal_open", "solved_frac",
    "success", "reach_p", "steps_to_p", "ep_len",
]


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
                items.append({
                    "group": os.path.basename(os.path.dirname(norm)),
                    "run": os.path.basename(norm),
                    "seed": _seed_of(d),
                    "ckpt": ck,
                })
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

    items = _collect()
    if not items:
        raise FileNotFoundError("Nothing to evaluate. Pass --run_glob or --checkpoints.")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base = env.unwrapped
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    term_on = getattr(base.cfg, "terminate_on_success", True)
    print(f"\n[INFO] {len(items)} checkpoint(s), {args_cli.num_envs} episodes each,"
          f" terminate_on_success={term_on}")

    for it in items:
        runner.load(it["ckpt"])
        policy = runner.get_inference_policy(device=base.device)
        it.update(_evaluate(env, base, policy, agent_cfg.device))
        print(f"  {it['group']:<22} {it['run']:<22} seed={it['seed']:<4}"
              f" margin={it['margin']:.4f} mean_max={it['mean_max']:.4f}"
              f" success={it['success']:.4f} steps_to_p={it['steps_to_p']:7.1f}")

    _report(items)

    if args_cli.output:
        keys = ["group", "run", "seed"] + FIELDS
        with open(args_cli.output, "w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=keys)
            w.writeheader()
            for it in items:
                w.writerow({k: it[k] for k in keys})
        print(f"\n[INFO] wrote {len(items)} rows to {os.path.abspath(args_cli.output)}")

    env.close()


def _evaluate(env, base, policy, device) -> dict:
    """Run exactly one episode per environment, freezing each when its episode ends."""
    n = base.num_envs
    dev = base.device
    thr, pthr = args_cli.success_threshold, args_cli.progress_threshold
    horizon = int(base.max_episode_length)

    active = torch.ones(n, dtype=torch.bool, device=dev)
    run_max = torch.zeros(n, device=dev)
    run_sum = torch.zeros(n, device=dev)
    solved_steps = torch.zeros(n, device=dev)
    ep_len = torch.zeros(n, device=dev)
    last_q = torch.zeros(n, device=dev)
    first_p = torch.full((n,), NAN, device=dev)
    ever_term = torch.zeros(n, dtype=torch.bool, device=dev)

    with torch.inference_mode():
        env.reset()
        obs = env.get_observations().to(device)
        for t in range(horizon):
            if not bool(active.any()):
                break
            # read before stepping: env.step() resets finished environments before returning, so a
            # post-step read would give the post-reset drawer position for those environments
            q = base._cabinet.data.joint_pos[:, base.drawer_joint_idx]
            upd = active.float()
            run_max = torch.where(active, torch.maximum(run_max, q), run_max)
            run_sum += q * upd
            solved_steps += (q > thr).float() * upd
            ep_len += upd
            last_q = torch.where(active, q, last_q)
            newly = active & (q > pthr) & torch.isnan(first_p)
            first_p = torch.where(newly, torch.full_like(first_p, float(t)), first_p)

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions.to(env.device))
            obs = obs.to(device)

            # reset_terminated is a standard DirectRLEnv buffer and is not cleared by _reset_idx
            ever_term |= active & base.reset_terminated
            # freeze this environment: its episode is over and the sim has already reset it
            active &= ~dones.bool()

    # an episode that terminated crossed the threshold by definition, even if the pre-step reads
    # never caught the crossing step itself
    run_max = torch.where(ever_term, torch.maximum(run_max, torch.full_like(run_max, thr)), run_max)
    reached = ~torch.isnan(first_p)
    denom = ep_len.clamp(min=1)

    return {
        "mean_open": (run_sum / denom).mean().item(),
        "margin": ((run_sum / denom).mean() / thr).item(),
        "mean_max": run_max.mean().item(),
        "terminal_open": last_q.mean().item(),
        "solved_frac": (solved_steps / denom).mean().item(),
        "success": ever_term.float().mean().item(),
        "reach_p": reached.float().mean().item(),
        "steps_to_p": first_p[reached].mean().item() if bool(reached.any()) else NAN,
        "ep_len": ep_len.mean().item(),
    }


def _report(items: list[dict]) -> None:
    """Print per-group mean +/- sd across seeds."""
    groups: dict[str, list[dict]] = {}
    for it in items:
        groups.setdefault(it["group"], []).append(it)

    print("\n" + "=" * 104)
    print(f"SUMMARY  (success = {args_cli.success_threshold} m, progress = {args_cli.progress_threshold} m)")
    print("=" * 104)
    print(f"{'group':<22} {'n':>3} " + "".join(f"{f:>19}" for f in ("margin", "mean_max", "success", "steps_to_p")))
    for g, rs in sorted(groups.items()):
        cells = []
        for key in ("margin", "mean_max", "success", "steps_to_p"):
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
