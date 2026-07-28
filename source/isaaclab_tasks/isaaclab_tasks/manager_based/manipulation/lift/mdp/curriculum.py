# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success-replay auto-curriculum for the lift task.

Tracks per-episode completion of 5 subtasks (move close, grasp, lift, orient in air, place
upright), stashes the full physics state of every env the moment it newly completes a subtask
into a per-subtask ring buffer, and periodically recomputes a sampling distribution over "which
subtask most needs practice" from an EMA of each subtask's success rate. At reset time, a
configurable fraction of resetting envs are teleported into a buffer-sampled state for a
distribution-sampled subtask (plus small joint-space domain randomization) instead of a normal
reset, so training focuses on whichever subtask is currently the bottleneck.

This mirrors the curriculum built for the Direct RL ``FrankaCabinetEnv``
(``isaaclab_tasks/direct/franka_cabinet/franka_cabinet_env.py``), adapted to the manager-based
term system:

- buffer allocation (``__init__`` there) -> ``init_success_curriculum``, an ``EventTermCfg(mode="startup")``.
- per-step bookkeeping (``_get_rewards`` there) -> ``update_success_progression``, a ``RewardTermCfg``
  (runs every step for all envs; always returns zero reward, it exists purely for the side effects).
- reset-time replay (``_reset_idx`` there) -> ``reset_success_curriculum``, an ``EventTermCfg(mode="reset")``.

All curriculum state is stored on a single object (``env.lift_curriculum``) so the reward term and
the event term - which are independently-instantiated manager terms and would otherwise not share
any state - can read and write the same buffers. This mirrors the existing precedent in
``isaaclab_tasks/manager_based/manipulation/deploy/mdp/events.py``'s ``randomize_gear_type``,
which stashes itself onto the env the same way (``env._gear_type_manager = self``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

NUM_SUBTASKS = 5
# subtask indices, for readability:
#   0 = move close to the cube
#   1 = grasp the cube
#   2 = lift it up
#   3 = orient it upright in the air
#   4 = place it upright at the target


def _identity_quat(n: int, device) -> torch.Tensor:
    return torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1)


def _get_subtasks(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns an ``(num_envs, 5)`` bool tensor of which subtasks are currently satisfied."""
    C = env.lift_curriculum
    robot = env.scene["robot"]
    obj = env.scene["object"]
    ee_frame = env.scene["ee_frame"]

    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    obj_pos = obj.data.root_pos_w
    obj_quat = obj.data.root_quat_w
    ee_dist = torch.norm(obj_pos - ee_pos, dim=-1)

    gripper_pos = robot.data.joint_pos[:, C.gripper_joint_ids]
    gripper_closed = (gripper_pos < env.cfg.curriculum_gripper_closed_thresh).all(dim=-1)
    gripper_open = (gripper_pos > env.cfg.curriculum_gripper_open_thresh).all(dim=-1)

    orient_err = quat_error_magnitude(obj_quat, _identity_quat(env.num_envs, env.device))

    command = env.command_manager.get_command("object_pose")
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    goal_dist = torch.norm(des_pos_w - obj_pos, dim=-1)

    # 1. move close to the cube
    sub_task_1 = ee_dist < env.cfg.curriculum_reach_dist
    # 2. grasp the cube: close to it AND fingers closed around it
    sub_task_2 = (ee_dist < env.cfg.curriculum_grasp_dist) & gripper_closed
    # 3. lift it up: clearly above the table/grasp-jitter range
    sub_task_3 = obj_pos[:, 2] > env.cfg.curriculum_lift_height
    # 4. orient it upright in the air: still lifted AND close to its upright (spawn) orientation
    sub_task_4 = (obj_pos[:, 2] > env.cfg.curriculum_lift_height) & (orient_err < env.cfg.curriculum_orient_tol)
    # 5. place it upright at the target: near the commanded goal, upright, and released
    sub_task_5 = (
        (goal_dist < env.cfg.curriculum_place_pos_tol)
        & (orient_err < env.cfg.curriculum_orient_tol)
        & gripper_open
    )

    return torch.stack([sub_task_1, sub_task_2, sub_task_3, sub_task_4, sub_task_5], dim=1)


def _get_world(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns an ``(num_envs, world_dim)`` snapshot of robot joints + env-local object pose."""
    robot = env.scene["robot"]
    obj = env.scene["object"]
    return torch.cat(
        [
            robot.data.joint_pos,
            obj.data.root_pos_w - env.scene.env_origins,
            obj.data.root_quat_w,
        ],
        dim=1,
    )


def _update_distribution(env: ManagerBasedRLEnv):
    """Recompute the subtask sampling distribution from the (eval-only) EMA success rate."""
    C = env.lift_curriculum
    mask = ~C.is_curriculum_episode
    if mask.sum() == 0:
        return

    batch_success = C.progression[mask].float().mean(dim=0)

    alpha = env.cfg.success_rate_alpha
    C.success_rate = (1.0 - alpha) * C.success_rate + alpha * batch_success

    difficulty = 1.0 - C.success_rate
    previous = torch.cat([torch.zeros(1, device=env.device), difficulty[:-1]])
    gaps = difficulty - previous
    gaps = gaps - gaps.min()
    gaps = gaps + 1e-8

    soft = gaps.pow(env.cfg.prob_exp).softmax(dim=0)

    confidence = gaps / gaps.sum().clamp(min=1e-8)
    top2 = torch.topk(confidence, k=2)
    winner = top2.indices[0]
    margin = top2.values[0] - top2.values[1]

    hard = torch.zeros_like(gaps)
    hard[winner] = 1.0

    blend = torch.clamp(margin / env.cfg.greedy_margin, 0.0, 1.0)
    C.distribution = (1.0 - blend) * soft + blend * hard
    C.distribution /= C.distribution.sum()

    C.last_blend = blend.item()
    C.last_margin = margin.item()
    C.last_selected = winner.item()
    C.last_difficulty = difficulty
    C.last_gaps = gaps


def _run_curriculum_controller(env: ManagerBasedRLEnv):
    """Auto-disable the curriculum once overall success stops improving fast enough."""
    C = env.lift_curriculum
    if not C.curriculum_enabled:
        return

    success = C.overall_success.mean().item()

    if C.progress >= env.cfg.window_analysis_start and C.controller_snapshot is None:
        C.controller_snapshot = success

    if (
        C.progress >= env.cfg.window_analysis_start + env.cfg.window_analysis_size
        and not C.controller_checked
    ):
        C.controller_checked = True
        C.controller_snapshot_two = success
        delta_success = success - C.controller_snapshot
        slope = delta_success / env.cfg.window_analysis_size
        C.curriculum_enabled = slope > env.cfg.slope_threshold


class init_success_curriculum(ManagerTermBase):
    """Allocates the shared curriculum buffers once, at env startup."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        robot = env.scene["robot"]
        self.robot_joint_dim = robot.num_joints
        self.world_dim = self.robot_joint_dim + 7  # + object pos(3) + quat(4)
        self.gripper_joint_ids = robot.find_joints(env.cfg.curriculum_gripper_joint_names)[0]

        n, device = env.num_envs, env.device
        self.progression = torch.zeros(n, NUM_SUBTASKS, dtype=torch.bool, device=device)
        self.success_rate = torch.zeros(NUM_SUBTASKS, device=device)
        self.distribution = torch.softmax(torch.ones(NUM_SUBTASKS, device=device), dim=0)
        self.success_buffer = torch.zeros(NUM_SUBTASKS, env.cfg.success_buffer_size, self.world_dim, device=device)
        self.pose_buffer_idx = torch.zeros(NUM_SUBTASKS, dtype=torch.long, device=device)
        # how many valid (actually-written) entries exist per subtask; slots beyond this are
        # still the zero-init default and must not be sampled/replayed (an all-zero quaternion
        # is not a valid rotation and will crash the GPU sim)
        self.pose_buffer_count = torch.zeros(NUM_SUBTASKS, dtype=torch.long, device=device)
        self.is_curriculum_episode = torch.zeros(n, dtype=torch.bool, device=device)
        self.curriculum_subtask = torch.full((n,), -1, dtype=torch.long, device=device)
        self.overall_success = torch.zeros(n, device=device)

        self.progress = 0.0
        self.curriculum_enabled = env.cfg.reset_state_curriculum_enabled
        self.controller_snapshot = None
        self.controller_snapshot_two = None
        self.controller_checked = False

        # populated by _update_distribution / reset_success_curriculum, read back by
        # update_success_progression for logging once env.extras["log"] exists again
        self.last_blend = 0.0
        self.last_margin = 0.0
        self.last_selected = -1
        self.last_difficulty = torch.zeros(NUM_SUBTASKS, device=device)
        self.last_gaps = torch.zeros(NUM_SUBTASKS, device=device)
        self.last_reset_distance = 0.0
        self.last_reset_variance = 0.0
        self.last_sample_rate = 0.0

        env.lift_curriculum = self

    def __call__(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor | None):
        # all setup happens once in __init__; startup events only fire once anyway
        return None


def update_success_progression(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-step bookkeeping: track subtask completion, fill the replay buffer, update the
    sampling distribution. Always returns zero reward - this term exists purely for its side
    effects and must be given a nonzero weight so the reward manager doesn't skip calling it."""
    C = env.lift_curriculum

    completions = _get_subtasks(env)
    world = _get_world(env)

    new_completion = completions & (~C.progression)
    C.progression |= completions

    completed_envs, completed_tasks = torch.where(new_completion)
    if len(completed_envs) > 0:
        completed_worlds = world[completed_envs]
        for task in range(NUM_SUBTASKS):
            task_mask = completed_tasks == task
            if task_mask.any():
                worlds = completed_worlds[task_mask]
                start = C.pose_buffer_idx[task]
                count = worlds.shape[0]
                indices = (torch.arange(count, device=env.device) + start) % env.cfg.success_buffer_size
                C.success_buffer[task, indices] = worlds
                C.pose_buffer_idx[task] = (start + count) % env.cfg.success_buffer_size
                C.pose_buffer_count[task] = torch.clamp(
                    C.pose_buffer_count[task] + count, max=env.cfg.success_buffer_size
                )

    # instantaneous full-task-success proxy, consumed by _run_curriculum_controller (mirrors
    # the cabinet reference's `self.overall_success = torch.clamp(drawer_pos / 0.39, 0.0, 1.0)`
    # in `_get_dones` - the last subtask, "placed upright at the target", *is* full task success)
    C.overall_success = completions[:, -1].float()
    C.progress = min(env.common_step_counter / env.cfg.curriculum_total_iterations, 1.0)

    if env.cfg.reset_state_curriculum_enabled and env.cfg.controller_enabled:
        _run_curriculum_controller(env)

    if env.cfg.reset_state_curriculum_enabled:
        if torch.rand((), device=env.device) < 0.10:
            _update_distribution(env)
    else:
        # keep deterministic by not touching the rand generator when curriculum is off
        if env.common_step_counter % 10 == 0:
            _update_distribution(env)

    if "log" in env.extras:
        L = env.extras["log"]
        success = C.progression.float()
        for i in range(NUM_SUBTASKS):
            L[f"subtasks/success_{i + 1}"] = success[:, i].mean().item()
        L["env_compare/highest_subtask"] = success.sum(dim=1).mean().item()

        curriculum_mask = C.is_curriculum_episode
        eval_mask = ~curriculum_mask

        if eval_mask.any():
            eval_success = success[eval_mask].mean(dim=0)
            L["env_compare/eval_success_mean"] = eval_success.mean().item()
            for i in range(NUM_SUBTASKS):
                L[f"env_compare/eval_success_{i + 1}"] = eval_success[i].item()

        if curriculum_mask.any():
            replay_success = success[curriculum_mask].mean(dim=0)
            L["env_compare/replay_success_mean"] = replay_success.mean().item()
            for i in range(NUM_SUBTASKS):
                L[f"env_compare/replay_success_{i + 1}"] = replay_success[i].item()
            for task in range(NUM_SUBTASKS):
                mask = curriculum_mask & (C.curriculum_subtask == task)
                L[f"env_compare/replay_count_{task + 1}"] = mask.sum().item()
                if mask.any():
                    L[f"env_compare/replay_task_success_{task + 1}"] = success[mask, task].mean().item()

        for i in range(NUM_SUBTASKS):
            L[f"curriculum/success_rate_{i + 1}"] = C.success_rate[i].item()
            L[f"curriculum/difficulty_{i + 1}"] = C.last_difficulty[i].item()
            L[f"curriculum/distribution_{i + 1}"] = C.distribution[i].item()
            L[f"curriculum/gap_{i + 1}"] = C.last_gaps[i].item()
        L["curriculum/blend"] = C.last_blend
        L["curriculum/margin"] = C.last_margin
        L["curriculum/selected"] = C.last_selected
        L["curriculum/natural"] = (~C.is_curriculum_episode).float().mean().item()
        L["curriculum/sample_rate"] = C.last_sample_rate
        L["curriculum/sample_ratio_target"] = env.cfg.sampling_ratio
        L["curriculum/reset_distance"] = C.last_reset_distance
        L["curriculum/reset_variance"] = C.last_reset_variance

        L["controller/curriculum_enabled"] = int(C.curriculum_enabled)
        L["controller/first_snapshot_taken"] = 0.0 if C.controller_snapshot is None else C.controller_snapshot
        L["controller/second_snapshot_taken"] = (
            0.0 if C.controller_snapshot_two is None else C.controller_snapshot_two
        )
        if C.controller_checked:
            L["controller/slope"] = (C.controller_snapshot_two - C.controller_snapshot) / env.cfg.window_analysis_size

    return torch.zeros(env.num_envs, device=env.device)


def reset_success_curriculum(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """On reset: replay a distribution-sampled subtask's stored world state into a
    sampling_ratio-fraction of the resetting envs, and clear the per-episode completion flags
    for all of them. Must be declared after the default reset events in ``EventCfg`` so it only
    overwrites the curriculum-picked subset on top of the default randomization.

    Note: this function must not write to ``env.extras["log"]`` - at this point in
    ``ManagerBasedRLEnv._reset_idx``, ``event_manager.apply(mode="reset", ...)`` runs *before*
    ``self.extras["log"]`` is (re)created, so anything logged here would be silently discarded.
    Reset-time stats are stashed on ``env.lift_curriculum`` instead and logged one step later by
    ``update_success_progression``.
    """
    C = env.lift_curriculum
    robot = env.scene["robot"]
    obj = env.scene["object"]

    if env.cfg.reset_state_curriculum_enabled and C.curriculum_enabled:
        n = len(env_ids)
        sample_ratio = env.cfg.sampling_ratio
        num_curriculum = int(n * sample_ratio)

        perm = torch.randperm(n, device=env.device)
        picked = torch.zeros(n, dtype=torch.bool, device=env.device)
        picked[perm[:num_curriculum]] = True

        # subtasks with no recorded successes yet still hold their zero-init buffer rows (e.g.
        # an all-zero quaternion), which is not a valid pose to replay
        valid_subtasks = C.pose_buffer_count > 0
        if not valid_subtasks.any():
            picked[:] = False

        C.is_curriculum_episode[env_ids] = False
        C.is_curriculum_episode[env_ids[picked]] = True
        C.curriculum_subtask[env_ids] = -1

        if picked.any():
            num_picked = int(picked.sum().item())
            picked_ids = env_ids[picked]

            # sample subtasks, restricted to ones that actually have recorded poses
            masked_distribution = torch.where(valid_subtasks, C.distribution, torch.zeros_like(C.distribution))
            mass = masked_distribution.sum()
            if mass <= 0:
                # the curriculum's preferred subtask(s) have no recorded poses yet (e.g.
                # distribution has fully committed to a subtask with count == 0); fall back to
                # uniform sampling over whichever subtasks do have data
                masked_distribution = valid_subtasks.float()
                mass = masked_distribution.sum()
            masked_distribution = masked_distribution / mass
            subtasks = torch.multinomial(masked_distribution, num_picked, replacement=True)
            C.curriculum_subtask[picked_ids] = subtasks

            # clamp to the slots that have actually been written for that subtask
            max_valid = C.pose_buffer_count[subtasks].clamp(min=1)
            world_ids = (torch.rand(num_picked, device=env.device) * max_valid.float()).long()
            world_ids = world_ids.clamp(max=env.cfg.success_buffer_size - 1)
            worlds = C.success_buffer[subtasks, world_ids]

            robot_joint_pos = worlds[:, : C.robot_joint_dim]
            robot_joint_pos = robot_joint_pos + sample_uniform(
                -env.cfg.curriculum_dr, env.cfg.curriculum_dr, robot_joint_pos.shape, env.device
            )
            robot_joint_pos = torch.clamp(
                robot_joint_pos,
                robot.data.soft_joint_pos_limits[picked_ids, :, 0],
                robot.data.soft_joint_pos_limits[picked_ids, :, 1],
            )
            robot_joint_vel = torch.zeros_like(robot_joint_pos)
            robot.set_joint_position_target(robot_joint_pos, env_ids=picked_ids)
            robot.write_joint_state_to_sim(robot_joint_pos, robot_joint_vel, env_ids=picked_ids)

            obj_pos_local = worlds[:, C.robot_joint_dim : C.robot_joint_dim + 3]
            obj_quat = worlds[:, C.robot_joint_dim + 3 : C.robot_joint_dim + 7]
            obj_pos_w = obj_pos_local + env.scene.env_origins[picked_ids]
            obj.write_root_pose_to_sim(torch.cat([obj_pos_w, obj_quat], dim=-1), env_ids=picked_ids)
            obj.write_root_velocity_to_sim(torch.zeros(num_picked, 6, device=env.device), env_ids=picked_ids)

            variance = (robot_joint_pos - robot.data.default_joint_pos[picked_ids]).pow(2).mean()
            distance = torch.norm(robot_joint_pos - robot.data.default_joint_pos[picked_ids], dim=1)
            C.last_reset_distance = distance.mean().item()
            C.last_reset_variance = variance.item()
            C.last_sample_rate = picked.float().mean().item()

    C.progression[env_ids] = False
