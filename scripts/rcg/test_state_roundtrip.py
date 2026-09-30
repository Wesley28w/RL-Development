# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gate test for the RCG state capture/restore contract.

Everything in Reverse Curriculum Generation rests on the start states being states the policy can
actually be in. This script measures that, and it measures exactly as much of it as the task's state
schema claims.

Task-agnostic: it reads the schema from whatever the task's ``_rcg_capture_state`` returns, so it
tests Franka Cabinet and Franka Lift -- and anything added later -- without knowing what is in their
states.

Three measurements:

1. **Round trip.** Capture, scramble the simulation, restore, capture again. The two captures
   must be bit-identical. **Always a gate.**
2. **A determinism control.** Restore the same state twice and replay the same actions from each.
   Any difference is the simulator failing to be reproducible, which nothing here can fix.
   **Always a gate.**
3. **Dynamics equivalence.** Record an action sequence played from the captured state, then restore
   and replay the same sequence. This is the check that catches state the capture dict *forgets* --
   a round trip alone would happily pass while silently dropping, say, an action-target buffer.
   **A gate only when ``rcg.capture_full_state`` is set.**

Why (3) is conditional. With ``capture_full_state = False`` -- the benchmark default on both tasks,
chosen so that RCG's start states carry exactly the information the reset-pose curriculum's do -- the
state is **positions only**. A restore deliberately zeroes velocities and re-anchors any
action-target buffer. So a replay starts from rest where the natural trajectory had momentum, and it
*cannot* reproduce that trajectory. Measured on 32 environments over 20 steps, the residual is 1.19x
the motion on Franka Cabinet and 0.14x on Franka Lift: not a bug, but the direct cost of the
positions-only decision, which the run-time restore pays on every reset.

With ``capture_full_state = True`` the claim "restore and continue is equivalent to having arrived
here naturally" is a real claim, and then (3) gates on it. What remains at that setting lives in
PhysX internals -- articulation solver caches and contact impulses -- that Isaac Lab's asset APIs do
not expose, so no addition to the state dict can capture it. Expect more of that remainder on a task
with a free-floating object in a gripper, such as Franka Lift, than on Franka Cabinet.

The gate for (3) is relative: the residual must be small compared with how far the trajectory
actually moved. A fixed absolute tolerance would be either so loose it catches nothing or so tight it
fails on irreducible drift.

Run it **both ways** on a new task. At the default it tells you what the positions-only start states
cost; with ``env.rcg.capture_full_state=true`` it tells you whether the schema is complete, which is
the thing a bug would show up in.

All of it runs through ``_rcg_physics_step``, which is the exact code path the Brownian expansion
uses.

``--dry_run_expand`` additionally runs one SampleNearby from recorded goal states and reports where
the resulting start states sit along the task. That is the direct test of whether action-space
Brownian motion actually moves the task *away* from the goal -- the one property a reverse
curriculum cannot work without, and the one that is not obvious a priori for an under-actuated
progress variable.

Usage:

.. code-block:: powershell

    isaaclab.bat -p scripts/rcg/test_state_roundtrip.py --task Isaac-Franka-Cabinet-Direct-v0 --num_envs 64 --headless
    isaaclab.bat -p scripts/rcg/test_state_roundtrip.py --task Isaac-Lift-Cube-Franka-RCG-v0 --num_envs 64 --headless

    # is the state schema complete? (gates on dynamics equivalence)
    # note: ` is PowerShell's line continuation, and nothing may follow it on the line
    isaaclab.bat -p scripts/rcg/test_state_roundtrip.py --task Isaac-Lift-Cube-Franka-RCG-v0 `
        --num_envs 64 --headless env.rcg.capture_full_state=true

    # does SampleNearby move the task away from the goal?
    isaaclab.bat -p scripts/rcg/test_state_roundtrip.py --task Isaac-Lift-Cube-Franka-RCG-v0 `
        --num_envs 256 --dry_run_expand --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify RCG state capture/restore equivalence.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
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

# Hydra-style overrides for `rcg` fields, e.g. `env.rcg.capture_full_state=true`. This script builds
# its configuration with `parse_env_cfg` rather than the `hydra_task_config` decorator the training
# scripts use -- it needs no agent configuration -- so the handful of overrides that matter here are
# applied by hand rather than by pulling in the whole Hydra stack.
_OVERRIDES = [arg for arg in hydra_args if arg.startswith("env.rcg.")]


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
    is_gate: bool,
) -> bool:
    """Judge the replay residual against the trajectory's own motion, with the control for scale.

    ``is_gate`` is False when the state schema is positions-only, in which case the residual is
    reported as a measurement and the return value is not allowed to fail the script -- see the
    module docstring for why the check is not applicable at that setting.
    """
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
            + ("" if field_ok else "  <-- larger than the budget")
        )
    if is_gate:
        print(f"[{'PASS' if ok else 'FAIL'}] {label} (residual <= {rtol:g} * motion)")
    else:
        print(
            f"[INFO] {label}: not a gate at 'rcg.capture_full_state = False'.\n"
            "       The state is positions only, so a restore starts from rest where the natural trajectory\n"
            "       had momentum and cannot reproduce it. The number above is the cost of that choice, which\n"
            "       the run-time restore pays on every reset -- not a defect. Re-run with\n"
            "       'env.rcg.capture_full_state=true' to turn this back into a gate on schema completeness."
        )
    print(
        "       'control' is the same replay run twice from the same restored state, so it measures whether\n"
        "       restore is reproducible. It must be zero (or solver noise): a non-zero control means the\n"
        "       simulation itself is not reproducible, which is a different problem and nothing here can\n"
        "       fix it. Any residual above the control comes either from state the schema omits or from\n"
        "       PhysX internals that the asset APIs do not expose."
    )
    return ok


def _apply_overrides(env_cfg) -> None:
    """Apply any ``env.rcg.<field>=<value>`` arguments onto the configuration."""
    for override in _OVERRIDES:
        key, _, raw = override.partition("=")
        field = key[len("env.rcg.") :]
        if not hasattr(env_cfg.rcg, field):
            raise ValueError(f"'RCGCfg' has no field '{field}' (from '{override}').")
        current = getattr(env_cfg.rcg, field)
        if isinstance(current, bool):
            value = raw.strip().lower() in ("1", "true", "yes")
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        else:
            value = raw
        setattr(env_cfg.rcg, field, value)
        print(f"[INFO] override: rcg.{field} = {value!r}")


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    if not hasattr(env_cfg, "rcg"):
        raise ValueError(f"Task '{args_cli.task}' does not support RCG: its configuration has no 'rcg' field.")
    # the curriculum stays off: this script exercises the capture/restore primitives directly
    env_cfg.rcg.enabled = False
    _apply_overrides(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    action_dim = env._rcg_action_dim
    all_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    def random_actions():
        return torch.randn((env.num_envs, action_dim), device=env.device)

    def restore(state):
        env._rcg_restore_state(all_ids, state)
        env.scene.write_data_to_sim()
        env.sim.forward()

    # inference_mode, not no_grad: the simulator write APIs update buffers such as `data.joint_acc`
    # in place, and those are inference tensors whenever anything has refreshed them from inside an
    # inference-mode block. The mixin's own methods establish this context for the same reason.
    with torch.inference_mode():
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

        control = _max_abs_diff(replayed_once, replayed_twice)
        ok_control = _report_absolute(
            "determinism control: the same restore replayed twice", control, args_cli.atol_roundtrip
        )
        ok_replay = _report_relative(
            f"dynamics equivalence: replay of {args_cli.replay_steps} identical actions",
            residual=_max_abs_diff(expected_after_replay, replayed_once),
            control=control,
            motion=_max_abs_diff(captured, expected_after_replay),
            rtol=args_cli.rtol_replay,
            is_gate=env.cfg.rcg.capture_full_state,
        )

    # dynamics equivalence only gates when the schema claims to be a complete state
    passed = ok_roundtrip and ok_control and (ok_replay or not env.cfg.rcg.capture_full_state)
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
    """Run one SampleNearby from the goal states and report where the new starts sit.

    Two checks, both of which a reverse curriculum needs and neither of which follows from the
    capture/restore contract:

    1. No generated start is already in the goal set. Verified by *restoring* the states and asking
       the task's own ``_rcg_is_solved``, rather than by inspecting the state dict, so it holds for
       any task's notion of success.
    2. The generated starts are genuinely *less* far along the task than the goal states. If
       action-space Brownian motion cannot regress the task's progress variable -- which is not
       obvious when that variable is under-actuated from the arm's point of view -- every candidate
       is either already solved or trivially solvable, and the curriculum has nowhere to expand to.
    """
    print("\n" + "=" * 78)
    print("SampleNearby dry run")
    print("=" * 78)

    env.cfg.rcg.enabled = True
    if args_cli.goal_states is not None:
        env.cfg.rcg.goal_state_path = args_cli.goal_states
    if not env.cfg.rcg.goal_state_path:
        print("\n[FAIL] No goal-state file: the task sets no 'rcg.goal_state_path' and --goal_states was not given.")
        return False
    print(f"\nGoal-state file: {env.cfg.rcg.goal_state_path}")

    goal_states = env._load_goal_states()
    print(f"Goal states:     {env._pool_size(goal_states)}")

    try:
        goal_progress = env._rcg_pool_progress(goal_states)
    except NotImplementedError:
        goal_progress = None
    else:
        print(
            f"Goal progress:   mean {goal_progress.mean():.4f}  min {goal_progress.min():.4f} "
            f" max {goal_progress.max():.4f}   (1.0 = in the goal set)"
        )

    new_starts = env._brownian_expand(goal_states)
    size = env._pool_size(new_starts)
    if size == 0:
        print("\n[FAIL] SampleNearby produced no usable start states.")
        return False
    print(f"\nNew start states: {size}")
    for key, value in env.rcg_log.items():
        if key.startswith("rcg/candidates") or key == "rcg/new_starts":
            print(f"       {key} = {float(value):.0f}")

    # 1. nothing in the pool may already satisfy the success condition
    with torch.inference_mode():
        check = min(env.num_envs, size)
        ids = torch.arange(check, dtype=torch.long, device=env.device)
        env._rcg_restore_state(ids, env._index_state_pool(new_starts, ids))
        env.scene.write_data_to_sim()
        env.sim.forward()
        already_solved = int(env._rcg_is_solved()[ids].sum().item())
    print(f"\n       already-solved start states among the first {check}: {already_solved}")
    if already_solved:
        print("\n[FAIL] A start state that already satisfies the success condition entered the pool.")
        return False

    if goal_progress is None:
        print("\n[ ok ] No '_rcg_pool_progress' on this task, so the regression check is skipped.")
        return True

    # 2. the pool has to sit further from the goal than the goal states do -- but not *much*
    #    further. "SampleNearby" is two words, and both of them are load-bearing.
    progress = env._rcg_pool_progress(new_starts)
    print(f"       new-start progress  mean {progress.mean():.4f}  min {progress.min():.4f}  max {progress.max():.4f}")
    edges = torch.linspace(0.0, 1.0, 11, device=progress.device)
    # clamp inside the first and last bucket rather than onto their edges: `bucketize` returns 0 for
    # a value sitting exactly on edges[0], and that bucket is not part of the printed range, so
    # states at progress 0 -- the ones that matter most here -- would vanish from the histogram
    bucketed = progress.contiguous().clamp(1e-6, 1.0 - 1e-6)
    counts = torch.bucketize(bucketed, edges).bincount(minlength=12)[1:11]
    print("       histogram of progress toward the goal:")
    for i in range(10):
        bar = "#" * int(40 * counts[i].item() / max(1, int(counts.max().item())))
        print(f"         [{edges[i]:.2f}, {edges[i + 1]:.2f})  {int(counts[i].item()):6d}  {bar}")

    # relative to the goal states' own progress, which is 1.0 for a task whose progress variable
    # saturates at the goal but need not be for one measured as a distance
    goal = float(goal_progress.mean())
    regressed = float((progress < 0.9 * goal).float().mean())
    nearby = float((progress >= 0.5 * goal).float().mean())
    print(f"\n       progress < 0.9 * goal progress ({0.9 * goal:.4f}):  {regressed:.3f}   (want > 0.10)")
    print(f"       progress >= 0.5 * goal progress ({0.5 * goal:.4f}):  {nearby:.3f}   (want > 0.30)")

    if regressed < 0.1:
        print(
            "\n[FAIL] Brownian motion is barely moving the task away from the goal, so the curriculum has nowhere to"
            " expand to: every generated start is already solved or trivially solvable. Lengthen"
            " 'rcg.brownian_horizon', or raise 'rcg.brownian_state_noise_std' (a documented deviation from the paper)."
        )
        return False

    if nearby < 0.3:
        print(
            "\n[FAIL] Brownian motion is moving the task *too far* from the goal. SampleNearby is producing starts"
            " that are effectively random rather than nearby, and a random start is one the policy cannot solve, so"
            " every new start scores 0 while the replayed archive scores 1. `select()` then finds nothing in"
            " (r_min, r_max) and the curriculum never advances on its own merits -- watch for 'rcg/frac_good_starts'"
            " pinned at 0 and 'rcg/mean_success_rate' pinned at n_old / (n_new + n_old).\n"
            "       Shorten 'rcg.brownian_horizon'. The paper's T_B = 50 assumes a task whose progress variable has"
            " a restoring force; a free object in a gripper does not, and is gone within a few steps."
        )
        return False

    print("\n[PASS] SampleNearby generates starts that are near the goal but not at it, as the curriculum requires.")
    return True


if __name__ == "__main__":
    main()
    simulation_app.close()
