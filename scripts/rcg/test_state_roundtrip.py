# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gate test for the RCG state capture/restore contract.

Everything in Reverse Curriculum Generation rests on one property: restoring a captured state
and continuing the simulation must be equivalent to having arrived at that state naturally. If
it is not, the curriculum is generated from states the policy can never actually be in, and no
amount of correct curriculum logic can rescue it.

Three measurements:

1. **Round trip.** Capture, scramble the simulation, restore, capture again. The two captures
   must be bit-identical.
2. **Dynamics equivalence.** Record an action sequence played from the captured state, then
   restore and replay the same sequence. This is the check that catches state the capture dict
   *forgets* -- a round trip alone would happily pass while silently dropping, say, the
   action-target buffer.
3. **A determinism control.** Restore the same state a second time and replay the same actions
   again. This separates "restore is not reproducible" from "restore is reproducible but not
   bit-identical to the natural trajectory" -- two very different problems that check 2 alone
   cannot tell apart.

On Franka Cabinet the measured result is: round trip exactly ``0`` on all five fields, control
exactly ``0``, and a replay residual of ~7e-4 of the distance the trajectory travelled. So
restore is fully deterministic, but the replayed trajectory still drifts slightly from the
natural one. The remainder lives in PhysX internals -- articulation solver caches and contact
impulses -- that Isaac Lab's ``Articulation`` API does not expose, so it cannot be captured by
adding more fields to the state dict. It is bounded and far below the +/-0.125 rad reset
randomisation the task already applies to every episode, which is why it is acceptable.

The gate is therefore relative: the replay residual must be small compared with how far the
trajectory actually moved. A fixed absolute tolerance would be either so loose it catches
nothing or so tight it fails on this irreducible drift.

All of it runs through ``_rcg_physics_step``, which is the exact code path the Brownian
expansion uses.

``--dry_run_expand`` additionally runs one SampleNearby from recorded goal states and reports
where the resulting start states sit along the task. That is the direct test of whether
action-space Brownian motion actually moves the task away from the goal.

Usage:

.. code-block:: powershell

    isaaclab.bat -p scripts/rcg/test_state_roundtrip.py --num_envs 64 --headless
    isaaclab.bat -p scripts/rcg/test_state_roundtrip.py --num_envs 256 --dry_run_expand --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify RCG state capture/restore equivalence.")
parser.add_argument("--task", type=str, default="Isaac-Franka-Cabinet-Direct-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--warmup_steps", type=int, default=30, help="Steps to run before capturing.")
parser.add_argument("--replay_steps", type=int, default=20, help="Steps in the dynamics-equivalence replay.")
parser.add_argument("--scramble_steps", type=int, default=50, help="Steps to run between capture and restore.")
parser.add_argument("--atol_roundtrip", type=float, default=1e-6, help="Tolerance for the immediate round trip.")
parser.add_argument(
    "--rtol_replay",
    type=float,
    default=0.05,
    help="Replay residual allowed, as a fraction of how far the trajectory moved.",
)
parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
parser.add_argument(
    "--dry_run_expand", action="store_true", default=False, help="Also run one SampleNearby from the goal states."
)
parser.add_argument("--goal_states", type=str, default=None, help="Goal-state file for --dry_run_expand.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _max_abs_diff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float((a[key] - b[key]).abs().max().item()) for key in a}


def _report_absolute(label: str, diffs: dict[str, float], atol: float) -> bool:
    ok = all(value <= atol for value in diffs.values())
    print(f"\n[{'PASS' if ok else 'FAIL'}] {label} (atol={atol:g})")
    for key, value in diffs.items():
        flag = "" if value <= atol else "  <-- exceeds tolerance"
        print(f"       {key:<20} max|diff| = {value:.3e}{flag}")
    return ok


def _report_relative(
    label: str,
    residual: dict[str, float],
    control: dict[str, float],
    motion: dict[str, float],
    rtol: float,
) -> bool:
    """Judge the replay residual against the trajectory's own motion, with the control for scale."""
    print(f"\n[ .. ] {label}")
    print(f"       {'field':<20} {'residual':>10} {'control':>10} {'motion':>10} {'resid/motion':>14}")
    ok = True
    for key in residual:
        # a field that barely moved cannot support a relative test, so fall back to the control
        if motion[key] > 1e-4:
            budget = rtol * motion[key]
            ratio = f"{residual[key] / motion[key]:14.2e}"
        else:
            budget = max(4.0 * control[key], 1e-5)
            ratio = f"{'n/a':>14}"
        field_ok = residual[key] <= budget
        ok = ok and field_ok
        print(
            f"       {key:<20} {residual[key]:10.3e} {control[key]:10.3e} {motion[key]:10.3e} {ratio}"
            + ("" if field_ok else "  <-- too large")
        )
    print(f"[{'PASS' if ok else 'FAIL'}] {label} (residual <= {rtol:g} * motion)")
    print(
        "       'control' is the same replay run twice from the same restored state, so it measures whether\n"
        "       restore is reproducible. A zero control with a small non-zero residual is the expected\n"
        "       outcome: restore is deterministic, and the drift from the natural trajectory comes from\n"
        "       PhysX solver state that the Articulation API does not expose. A non-zero control would\n"
        "       instead mean the simulation itself is not reproducible, which is a different problem."
    )
    return ok


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    if not hasattr(env_cfg, "rcg"):
        raise ValueError(f"Task '{args_cli.task}' does not support RCG: its configuration has no 'rcg' field.")
    # the curriculum stays off: this script exercises the capture/restore primitives directly
    env_cfg.rcg.enabled = False

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    action_dim = env.actions.shape[-1]
    all_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    def random_actions():
        return torch.randn((env.num_envs, action_dim), device=env.device)

    def restore(state):
        env._rcg_restore_state(all_ids, state)
        env.scene.write_data_to_sim()
        env.sim.forward()

    with torch.no_grad():
        # reach a non-trivial, contact-relevant state
        for _ in range(args_cli.warmup_steps):
            env._rcg_physics_step(random_actions())

        captured = env._rcg_capture_state(all_ids)

        # play a fixed action sequence forward from the captured state and remember where it ends
        replay_actions = [random_actions() for _ in range(args_cli.replay_steps)]
        for action in replay_actions:
            env._rcg_physics_step(action)
        expected_after_replay = env._rcg_capture_state(all_ids)

        # scramble the simulation, so a restore that silently does nothing cannot pass
        for _ in range(args_cli.scramble_steps):
            env._rcg_physics_step(random_actions())

        # 1. round trip
        restore(captured)
        ok_roundtrip = _report_absolute(
            "round trip: capture -> scramble -> restore -> capture",
            _max_abs_diff(captured, env._rcg_capture_state(all_ids)),
            args_cli.atol_roundtrip,
        )

        # 2. replay the recorded actions from the restored state
        for action in replay_actions:
            env._rcg_physics_step(action)
        replayed_once = env._rcg_capture_state(all_ids)

        # 3. control: restore again and replay again. Any difference from `replayed_once` is pure
        #    solver nondeterminism, since both runs start from an identical restored state.
        restore(captured)
        for action in replay_actions:
            env._rcg_physics_step(action)
        replayed_twice = env._rcg_capture_state(all_ids)

        ok_replay = _report_relative(
            f"dynamics equivalence: replay of {args_cli.replay_steps} identical actions",
            residual=_max_abs_diff(expected_after_replay, replayed_once),
            control=_max_abs_diff(replayed_once, replayed_twice),
            motion=_max_abs_diff(captured, expected_after_replay),
            rtol=args_cli.rtol_replay,
        )

    passed = ok_roundtrip and ok_replay
    print(
        "\n[RESULT] State capture/restore is"
        f" {'CONSISTENT' if passed else 'INCONSISTENT -- do not run RCG until this is fixed'}."
    )

    if args_cli.dry_run_expand:
        passed = _dry_run_expand(env) and passed

    env.close()
    if not passed:
        raise SystemExit(1)


def _dry_run_expand(env) -> bool:
    """Run one SampleNearby from the goal states and report where the new starts sit."""
    print("\n" + "=" * 78)
    print("SampleNearby dry run")
    print("=" * 78)

    env.cfg.rcg.enabled = True
    # --task defaults to the baseline id, whose RCGCfg leaves goal_state_path empty (only the RCG
    # task configuration sets it). Resolve it here so the dry run works against the baseline env,
    # which is the one whose capture/restore this script has just verified.
    if args_cli.goal_states is not None:
        env.cfg.rcg.goal_state_path = args_cli.goal_states
    elif not env.cfg.rcg.goal_state_path:
        from isaaclab_tasks.direct.franka_cabinet.franka_cabinet_env import DEFAULT_GOAL_STATE_PATH

        env.cfg.rcg.goal_state_path = DEFAULT_GOAL_STATE_PATH
    print(f"\nGoal-state file: {env.cfg.rcg.goal_state_path}")

    goal_states = env._load_goal_states()
    drawer_idx = env.drawer_joint_idx
    threshold = env.cfg.drawer_open_threshold

    goal_drawer = goal_states["cabinet_joint_pos"][:, drawer_idx]
    print(
        f"\nGoal states ({env._pool_size(goal_states)}): drawer mean {goal_drawer.mean():.4f},"
        f" min {goal_drawer.min():.4f}, max {goal_drawer.max():.4f}"
    )

    new_starts = env._brownian_expand(goal_states)
    size = env._pool_size(new_starts)
    if size == 0:
        print("\n[FAIL] SampleNearby produced no usable start states.")
        return False

    drawer = new_starts["cabinet_joint_pos"][:, drawer_idx]
    print(f"\nNew start states ({size}):")
    print(f"       drawer_top_joint  mean {drawer.mean():.4f}  min {drawer.min():.4f}  max {drawer.max():.4f}")
    edges = torch.linspace(0.0, max(float(drawer.max()), threshold) + 1e-6, 11, device=drawer.device)
    # .contiguous(): `drawer` is a column slice, and bucketize warns on non-contiguous input
    counts = torch.bucketize(drawer.contiguous(), edges).bincount(minlength=12)[1:11]
    print("       histogram of drawer opening:")
    for i in range(10):
        bar = "#" * int(40 * counts[i].item() / max(1, int(counts.max().item())))
        print(f"         [{edges[i]:.3f}, {edges[i + 1]:.3f})  {int(counts[i].item()):5d}  {bar}")
    for key, value in env.rcg_log.items():
        if key.startswith("rcg/candidates") or key == "rcg/new_starts":
            print(f"       {key} = {float(value):.0f}")

    if bool((drawer > threshold).any()):
        print("\n[FAIL] A start state past the success threshold entered the pool.")
        return False

    # the curriculum can only expand backwards if the generated starts are genuinely less far
    # along the task than the goal states
    regressed = float((drawer < threshold * 0.9).float().mean())
    print(f"\n       fraction of new starts with drawer < 0.9 * threshold: {regressed:.3f}")
    if regressed < 0.1:
        print(
            "[WARNING] Action-space Brownian motion is barely moving the drawer away from the goal, so the"
            " curriculum will struggle to expand. Consider raising 'rcg.brownian_state_noise_std' (a documented"
            " deviation from the paper) or lengthening 'rcg.brownian_horizon'."
        )
        return False
    print("[PASS] SampleNearby regresses the task away from the goal, as the reverse curriculum requires.")
    return True


if __name__ == "__main__":
    main()
    simulation_app.close()
