# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-pose curriculum for the lift task.

This ports the reset-state curriculum used on the Franka Cabinet direct-workflow environment
(``isaaclab_tasks.direct.franka_cabinet.franka_cabinet_env.FrankaCabinetEnv``) to the manager-based workflow.
The idea is the same: instead of always resetting an episode to a random "from scratch" state, occasionally
teleport it directly into a state where part of the task is already done, so the agent gets practice on the
harder, later parts of the task without having to first solve the earlier parts every single episode.

Because the manager-based workflow has no single per-env-step override point (unlike a ``DirectRLEnv``
subclass), the curriculum is split across two terms:

* :class:`subtask_progression_tracker` is registered as a **reward** term (weight nonzero but its return
  value is *always* the literal scalar 0 for every environment, so it can never change the reward signal --
  see the note below on why it isn't an "interval" event). It runs every simulation step, for every
  environment: pure bookkeeping. Tracks, per environment and per subtask, whether the subtask has been
  completed at any point during the current episode; the first time a subtask is completed it snapshots the
  full robot/object state ("world") into a small per-subtask ring buffer; and periodically recomputes a
  sampling distribution over the four subtasks, weighted towards whichever subtask is currently lagging the
  others the most. It only *reads* scene state (besides its own private buffers), so it cannot influence
  observations or terminations either.

* :func:`sample_curriculum_reset_state` (``EventTermCfg``, mode="reset", must be registered *after* the
  default reset event terms in ``EventCfg``): for a sampled fraction of the environments being reset,
  overwrites the default reset that just ran with a world snapshot pulled from the tracker's ring buffer
  (plus a little domain randomization on the robot joints), teleporting that environment directly into an
  already-partially-completed state.

Both terms are gated by :attr:`LiftEnvCfg.reset_state_curriculum_enabled`; each checks it first and returns
immediately when False, touching nothing else.

Why the tracker is a reward term and not an "interval" event (as it was in an earlier version of this file):
registering *any* ``EventTermCfg(mode="interval")`` term makes :class:`EventManager` call ``torch.rand`` on
the global RNG -- once per environment reset (to resample the term's internal timer) and once per simulation
step (same reason) -- unconditionally, regardless of what the term's own function does or checks. That
RNG-stream shift alone was enough to make a run with the curriculum "disabled" diverge from the true
baseline, since everything else that draws from the same generator (physics domain randomization, other
resets, ...) shifts downstream of it. ``RewardManager`` has no such hidden timer/RNG mechanism -- it just
calls the term function every step -- so driving the tracker from there means the disabled path performs
*zero* extra torch RNG draws, matching the baseline bit-for-bit. There is also no action/observation noise
injection here (unlike the Franka Cabinet version) -- that was specific to that environment's own
experiments and isn't part of the reset-pose curriculum itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# subtask order: (1) reach within 10cm of the cube, (2) grasp it, (3) lift it into the air,
# (4) bring it very close to (but not exactly at) the commanded target pose.
NUM_SUBTASKS = 4


class subtask_progression_tracker(ManagerTermBase):
    """Tracks reset-pose-curriculum subtask completion and owns the resulting sampling distribution.

    Subtask completion re-uses the same quantities the lift task's own reward terms are built from
    (end-effector-to-object distance as in ``object_ee_distance``, object height as in ``object_is_lifted``,
    and object-to-goal distance as in ``object_goal_distance``), so the thresholds line up with the reward
    shaping already in ``RewardsCfg``:

    * Subtask 1 (10cm from cube): ``reach_threshold`` defaults to 0.10, the same value as the ``std`` used by
      the ``reaching_object`` reward term.
    * Subtask 2 (grasp): the end-effector must be almost touching the object (``grasp_distance_threshold``,
      default 0.02 -- the same "very close" distance used as a bonus cutoff in the Franka Cabinet reward) and
      the gripper fingers must actually be closed (``gripper_closed_threshold``, default 0.02, the midpoint of
      the Franka gripper's 0.0 closed / 0.04 open command range). Proximity alone would be indistinguishable
      from subtask 1, so both conditions are required.
    * Subtask 3 (lift in air): object height above ``lift_height_threshold`` (default 0.10) -- clearly above
      the ``minimal_height`` (0.04) the reward terms use just to detect "off the table", so this is a distinct,
      harder milestone.
    * Subtask 4 (near target, not at it): object-to-goal distance below ``near_goal_threshold`` (default 0.05),
      the same ``std`` used by the ``object_goal_tracking_fine_grained`` reward term.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
        robot: Articulation = env.scene[robot_cfg.name]
        # world snapshot = robot joint positions + object position (env-local) + object orientation
        self.world_dim = robot.num_joints + 3 + 4

        self.progression = torch.zeros(env.num_envs, NUM_SUBTASKS, dtype=torch.bool, device=env.device)
        self.success_buffer = torch.zeros(
            NUM_SUBTASKS, env.cfg.success_buffer_size, self.world_dim, device=env.device
        )
        # buffer slots default to a valid identity object orientation (rather than the all-zero, non-unit
        # quaternion torch.zeros would otherwise leave in place) so sampling a subtask bucket that no
        # environment has completed yet can never write a degenerate quaternion into the simulation
        self.success_buffer[..., -4] = 1.0
        self.pose_buffer_idx = torch.zeros(NUM_SUBTASKS, dtype=torch.long, device=env.device)
        self.success_rate = torch.zeros(NUM_SUBTASKS, device=env.device)
        self.distribution = torch.softmax(torch.ones(NUM_SUBTASKS, device=env.device), dim=0)

        # written externally by sample_curriculum_reset_state
        self.is_curriculum_episode = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.curriculum_subtask = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

        # latest stats, surfaced to TensorBoard by the reset_pose_curriculum_metrics curriculum term
        self._log: dict[str, float] = {}

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        # start each new episode with every subtask marked incomplete
        self.progression[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        reach_threshold: float = 0.10,
        grasp_distance_threshold: float = 0.02,
        gripper_closed_threshold: float = 0.02,
        lift_height_threshold: float = 0.10,
        near_goal_threshold: float = 0.05,
        command_name: str = "object_pose",
        gripper_joint_names: Sequence[str] = ("panda_finger.*",),
        update_distribution_prob: float = 0.10,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        # this term's contribution to the reward is *always* exactly zero, enabled or not -- see the module
        # docstring for why it is registered as a reward term (with some nonzero weight so RewardManager
        # doesn't skip calling it) rather than an "interval" event.
        zero_reward = torch.zeros(env.num_envs, device=env.device)
        if not env.cfg.reset_state_curriculum_enabled:
            return zero_reward

        # RewardManager.compute() always runs after this step's physics has already been integrated (and
        # after any reset from the *previous* step), so every environment here has had at least one physics
        # step since its last reset -- there is no "just teleported, no dynamics applied yet" state to guard
        # against.
        env_ids = torch.arange(env.num_envs, device=env.device)

        completions = self._get_subtasks(
            env,
            env_ids,
            reach_threshold,
            grasp_distance_threshold,
            gripper_closed_threshold,
            lift_height_threshold,
            near_goal_threshold,
            command_name,
            gripper_joint_names,
            robot_cfg,
            object_cfg,
            ee_frame_cfg,
        )
        world = self._get_world(env, env_ids, robot_cfg, object_cfg)

        previously_completed = self.progression
        new_completion = completions & (~previously_completed)
        self.progression = previously_completed | completions

        # add newly-completed worlds to the per-subtask ring buffer
        completed_rows, completed_tasks = torch.where(new_completion)
        if completed_rows.numel() > 0:
            completed_worlds = world[completed_rows]
            for task in range(NUM_SUBTASKS):
                task_mask = completed_tasks == task
                if not task_mask.any():
                    continue
                worlds = completed_worlds[task_mask]
                start = self.pose_buffer_idx[task]
                count = worlds.shape[0]
                indices = (torch.arange(count, device=env.device) + start) % env.cfg.success_buffer_size
                self.success_buffer[task, indices] = worlds
                self.pose_buffer_idx[task] = (start + count) % env.cfg.success_buffer_size

        if torch.rand((), device=env.device) < update_distribution_prob:
            self._update_distribution(env)

        self._log.update({
            f"subtasks/success_{i + 1}": self.progression[:, i].float().mean().item() for i in range(NUM_SUBTASKS)
        })
        self._log["curriculum/highest_subtask"] = self.progression.float().sum(dim=1).mean().item()

        return zero_reward

    def _get_subtasks(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        reach_threshold: float,
        grasp_distance_threshold: float,
        gripper_closed_threshold: float,
        lift_height_threshold: float,
        near_goal_threshold: float,
        command_name: str,
        gripper_joint_names: Sequence[str],
        robot_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        ee_frame_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        robot: Articulation = env.scene[robot_cfg.name]
        object: RigidObject = env.scene[object_cfg.name]
        ee_frame = env.scene[ee_frame_cfg.name]

        ee_pos_w = ee_frame.data.target_pos_w[env_ids, 0, :]
        object_pos_w = object.data.root_pos_w[env_ids]
        ee_object_distance = torch.norm(ee_pos_w - object_pos_w, dim=-1)

        gripper_joint_ids, _ = robot.find_joints(list(gripper_joint_names))
        gripper_pos = robot.data.joint_pos[env_ids][:, gripper_joint_ids].mean(dim=-1)

        sub_task_1 = ee_object_distance < reach_threshold
        sub_task_2 = (ee_object_distance < grasp_distance_threshold) & (gripper_pos < gripper_closed_threshold)
        sub_task_3 = object_pos_w[:, 2] > lift_height_threshold

        command = env.command_manager.get_command(command_name)
        des_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w[env_ids], robot.data.root_quat_w[env_ids], command[env_ids, :3]
        )
        goal_distance = torch.norm(des_pos_w - object_pos_w, dim=-1)
        sub_task_4 = goal_distance < near_goal_threshold

        return torch.stack([sub_task_1, sub_task_2, sub_task_3, sub_task_4], dim=1)

    def _get_world(
        self, env: ManagerBasedRLEnv, env_ids: torch.Tensor, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        robot: Articulation = env.scene[robot_cfg.name]
        object: RigidObject = env.scene[object_cfg.name]
        object_pos_local = object.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
        return torch.cat([robot.data.joint_pos[env_ids], object_pos_local, object.data.root_quat_w[env_ids]], dim=-1)

    def _update_distribution(self, env: ManagerBasedRLEnv):
        # exclude curriculum (replayed) episodes so the difficulty estimate reflects genuine policy performance
        eval_mask = ~self.is_curriculum_episode
        if eval_mask.sum() == 0:
            return

        batch_success = self.progression[eval_mask].float().mean(dim=0)

        # EMA on the success rate to filter noise
        alpha = env.cfg.success_rate_alpha
        self.success_rate = (1.0 - alpha) * self.success_rate + alpha * batch_success

        # difficulty (failure rate) and the gap it has over the previous (easier) subtask
        difficulty = 1.0 - self.success_rate
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
        self.distribution = (1.0 - blend) * soft + blend * hard
        self.distribution = self.distribution / self.distribution.sum()

        self._log["curriculum/blend"] = blend.item()
        self._log["curriculum/margin"] = margin.item()
        self._log["curriculum/selected"] = winner.item()
        for i in range(NUM_SUBTASKS):
            self._log[f"curriculum/success_rate_{i + 1}"] = self.success_rate[i].item()
            self._log[f"curriculum/difficulty_{i + 1}"] = difficulty[i].item()
            self._log[f"curriculum/distribution_{i + 1}"] = self.distribution[i].item()


def sample_curriculum_reset_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    tracker_term_name: str = "subtask_progression_tracker",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Overrides the default reset for a sampled fraction of ``env_ids`` with a replayed subtask state.

    Must be registered on ``EventCfg`` *after* the default reset terms (``reset_scene_to_default`` /
    ``reset_root_state_uniform``) so that, for the picked environments, this term's writes are the ones that
    stick. When :attr:`LiftEnvCfg.reset_state_curriculum_enabled` is False, this is a no-op.
    """
    # subtask_progression_tracker is registered as a reward term (see the module docstring for why), so it
    # must be looked up through the reward manager rather than the event manager.
    tracker: subtask_progression_tracker = getattr(env.reward_manager.cfg, tracker_term_name).func

    if not env.cfg.reset_state_curriculum_enabled:
        tracker.is_curriculum_episode[env_ids] = False
        tracker.curriculum_subtask[env_ids] = -1
        return

    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    picked = torch.rand(len(env_ids), device=env.device) < env.cfg.sampling_ratio

    tracker.is_curriculum_episode[env_ids] = False
    tracker.is_curriculum_episode[env_ids[picked]] = True
    tracker.curriculum_subtask[env_ids] = -1

    if not picked.any():
        tracker._log["curriculum/sample_rate"] = 0.0
        tracker._log["curriculum/natural"] = tracker.is_curriculum_episode.float().mean().item()
        return

    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    picked_ids = env_ids[picked]
    num_picked = picked_ids.numel()

    subtasks = torch.multinomial(tracker.distribution, num_picked, replacement=True)
    tracker.curriculum_subtask[picked_ids] = subtasks

    world_ids = torch.randint(0, env.cfg.success_buffer_size, (num_picked,), device=env.device)
    worlds = tracker.success_buffer[subtasks, world_ids]

    num_joints = robot.num_joints
    robot_joint_pos = worlds[:, :num_joints] + sample_uniform(
        -env.cfg.curriculum_dr, env.cfg.curriculum_dr, (num_picked, num_joints), env.device
    )
    joint_pos_limits = robot.data.soft_joint_pos_limits[picked_ids]
    robot_joint_pos = torch.clamp(robot_joint_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1])
    joint_vel = torch.zeros_like(robot_joint_pos)
    robot.write_joint_state_to_sim(robot_joint_pos, joint_vel, env_ids=picked_ids)

    object_pos_local = worlds[:, num_joints : num_joints + 3]
    object_quat = worlds[:, num_joints + 3 : num_joints + 7]
    object_pos_w = object_pos_local + env.scene.env_origins[picked_ids]
    object.write_root_pose_to_sim(torch.cat([object_pos_w, object_quat], dim=-1), env_ids=picked_ids)
    object.write_root_velocity_to_sim(torch.zeros(num_picked, 6, device=env.device), env_ids=picked_ids)

    default_joint_pos = robot.data.default_joint_pos[env_ids]
    all_robot_joint_pos = default_joint_pos.clone()
    all_robot_joint_pos[picked] = robot_joint_pos
    distance = torch.norm(all_robot_joint_pos - default_joint_pos, dim=1)
    variance = (all_robot_joint_pos - default_joint_pos).pow(2).mean()

    tracker._log["curriculum/reset_distance"] = distance.mean().item()
    tracker._log["curriculum/reset_variance"] = variance.item()
    tracker._log["curriculum/sample_rate"] = picked.float().mean().item()
    tracker._log["curriculum/natural"] = tracker.is_curriculum_episode.float().mean().item()
