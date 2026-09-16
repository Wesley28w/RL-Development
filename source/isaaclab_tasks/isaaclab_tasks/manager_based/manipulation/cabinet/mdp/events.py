# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-pose curriculum for the cabinet-drawer task.

This ports the reset-state curriculum used on the Franka Cabinet direct-workflow environment
(``isaaclab_tasks.direct.franka_cabinet.franka_cabinet_env.FrankaCabinetEnv``) to the manager-based workflow,
following the same split used by the Lift task's port
(``isaaclab_tasks.manager_based.manipulation.lift.mdp.events``): instead of always resetting an episode to
the default closed-drawer state, occasionally teleport it directly into a state where part of the task is
already done, so the agent gets practice on the harder, later parts of the task without having to first solve
the earlier parts every single episode.

Unlike Lift (a free rigid-body object with a freshly-resampled goal pose every episode), the cabinet has no
per-episode randomized goal at all: the drawer always starts closed and "open" is a fixed target. This makes
the subtask thresholds plain physical/kinematic milestones with no goal-resampling confound to work around,
and means the world snapshot only needs robot joint positions and cabinet joint positions -- no object pose.

As with Lift, the curriculum is split across two terms:

* :class:`subtask_progression_tracker` is registered as a **reward** term (weight nonzero but its return
  value is *always* the literal scalar 0 for every environment, so it can never change the reward signal --
  see ``lift/mdp/events.py``'s module docstring for why a reward term is used here instead of an "interval"
  event: registering any interval event term consumes the global RNG unconditionally on every reset and
  step, which would make a "disabled" run diverge from the true baseline). It runs every simulation step, for
  every environment: pure bookkeeping. Tracks, per environment and per subtask, whether the subtask has been
  completed at any point during the current episode; the first time a subtask is completed it snapshots the
  robot/cabinet joint state ("world") into a small per-subtask ring buffer; and periodically recomputes a
  sampling distribution over the four subtasks, weighted towards whichever subtask is currently lagging the
  others the most.

* :func:`sample_curriculum_reset_state` (``EventTermCfg``, mode="reset", must be registered *after* the
  default reset event terms in ``EventCfg``): for a sampled fraction of the environments being reset,
  overwrites the default reset that just ran with a world snapshot pulled from the tracker's ring buffer
  (plus a little domain randomization on the robot joints), teleporting that environment directly into an
  already-partially-completed state.

Both terms are gated by :attr:`CabinetEnvCfg.reset_state_curriculum_enabled`; each checks it first and
returns immediately when False, touching nothing else.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# subtask order: (1) approach within reach of the handle, (2) touch/grasp the handle, (3) drawer slightly
# open, (4) drawer almost fully open.
NUM_SUBTASKS = 4


class subtask_progression_tracker(ManagerTermBase):
    """Tracks reset-pose-curriculum subtask completion and owns the resulting sampling distribution.

    Subtask completion re-uses the same quantities the cabinet task's own reward terms are built from
    (end-effector-to-handle distance as in ``approach_ee_handle``, the finger-straddle pose as in
    ``align_grasp_around_handle``, and drawer joint position as in ``multi_stage_open_drawer``), so the
    thresholds line up with the reward shaping already in ``RewardsCfg``:

    * Subtask 1 (proximity): ``proximity_threshold`` defaults to 0.20, the same near-field cutoff
      ``approach_ee_handle`` already uses.
    * Subtask 2 (touching the handle): the end-effector must be almost touching the handle
      (``touch_distance_threshold``, default 0.03 -- the same distance ``grasp_handle`` uses as its "close
      enough to grasp" cutoff) *and* the gripper must be in the correct straddle pose (left finger above the
      handle, right finger below -- the same condition ``align_grasp_around_handle`` rewards). Distance alone
      would be indistinguishable from an overshoot past the handle without a grasp-capable orientation.
    * Subtask 3 (drawer slightly open): drawer joint position above ``slightly_open_fraction`` (default 0.5)
      of the drawer joint's own travel range, unless ``slightly_open_threshold`` is set (see below).
    * Subtask 4 (drawer almost fully open): drawer joint position above ``almost_open_fraction`` (default
      0.90) of the drawer joint's own travel range, unless ``almost_open_threshold`` is set (see below) --
      kept just short of full travel, mirroring the direct Franka Cabinet environment's own final subtask
      threshold (0.38 out of a ~0.39-0.40 max, i.e. ~96%).

    Subtasks 3 and 4 default to **fractions of the drawer joint's own runtime travel range**
    (``cabinet.data.soft_joint_pos_limits``) rather than fixed distances in meters, because scaling a USD
    actor scales its prismatic joint limits along with it (documented PhysX behavior -- see e.g. Isaac
    Gym's physics docs: "Scaling an actor will change its collision geometry, mass properties, joint
    positions, and prismatic joint limits"). The OpenArm variant of this scene spawns the cabinet at
    ``scale=(0.75, 0.75, 0.75)`` (see ``config/openarm/cabinet_openarm_env_cfg.py``), so a fixed 0.35 m
    threshold would sit at or past its *scaled* drawer's true max travel (~0.75 x 0.39-0.40 =~ 0.29-0.30 m),
    making subtask 4 unreachable there. For an *unscaled* cabinet (Franka's), the original direct-workflow
    environment's literal absolute thresholds (0.20 m / 0.38 m) remain valid and are what that port should
    reproduce exactly; pass ``slightly_open_threshold``/``almost_open_threshold`` to use those fixed values
    directly instead of a fraction of the runtime range.

    Subtask 2 additionally requires the correct finger-straddle pose by default
    (``require_graspable_pose=True``); set it to ``False`` to reproduce the direct-workflow environment's
    original subtask 2, which is a pure distance check with no grasp-pose condition.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
        cabinet_cfg: SceneEntityCfg = cfg.params.get("cabinet_cfg", SceneEntityCfg("cabinet"))
        robot: Articulation = env.scene[robot_cfg.name]
        cabinet: Articulation = env.scene[cabinet_cfg.name]
        # world snapshot = robot joint positions + all cabinet joint positions
        self.world_dim = robot.num_joints + cabinet.num_joints

        self.progression = torch.zeros(env.num_envs, NUM_SUBTASKS, dtype=torch.bool, device=env.device)
        self.success_buffer = torch.zeros(
            NUM_SUBTASKS, env.cfg.success_buffer_size, self.world_dim, device=env.device
        )
        self.pose_buffer_idx = torch.zeros(NUM_SUBTASKS, dtype=torch.long, device=env.device)
        # how many slots of each subtask's ring buffer have actually been written. Sampling a slot that has
        # never been written replays all-zero joint positions, not a state the policy reached, so this gates
        # both which subtasks may be sampled at all and which slots within them -- see
        # sample_curriculum_reset_state. Mirrors the Factory environment's pose_buffer_count.
        self.pose_buffer_count = torch.zeros(NUM_SUBTASKS, dtype=torch.long, device=env.device)
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
        proximity_threshold: float = 0.20,
        touch_distance_threshold: float = 0.03,
        slightly_open_fraction: float = 0.5,
        almost_open_fraction: float = 0.90,
        slightly_open_threshold: float | None = None,
        almost_open_threshold: float | None = None,
        require_graspable_pose: bool = True,
        update_distribution_prob: float = 0.10,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet"),
        drawer_joint_name: str = "drawer_bottom_joint",
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        cabinet_frame_cfg: SceneEntityCfg = SceneEntityCfg("cabinet_frame"),
    ) -> torch.Tensor:
        # this term's contribution to the reward is *always* exactly zero, enabled or not -- see the module
        # docstring for why it is registered as a reward term rather than an "interval" event.
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
            proximity_threshold,
            touch_distance_threshold,
            slightly_open_fraction,
            almost_open_fraction,
            slightly_open_threshold,
            almost_open_threshold,
            require_graspable_pose,
            cabinet_cfg,
            drawer_joint_name,
            ee_frame_cfg,
            cabinet_frame_cfg,
        )
        world = self._get_world(env, env_ids, robot_cfg, cabinet_cfg)

        previously_completed = self.progression
        new_completion = completions & (~previously_completed)
        self.progression = previously_completed | completions

        # add newly-completed worlds to the per-subtask ring buffer -- only from NATURAL episodes.
        # self.progression is cleared for every env at reset, so a curriculum env satisfies the subtask it
        # was teleported into on the first evaluation after the teleport; without this mask it re-deposits
        # its own start state (plus the curriculum_dr noise) one step after drawing it, and the buffer turns
        # into a self-replicating population of its own output rather than a record of what the policy did.
        # A rung-4 teleport also satisfies rung 3 immediately, so one teleport would seed several buffers.
        buffer_completion = new_completion & (~self.is_curriculum_episode).unsqueeze(1)
        completed_rows, completed_tasks = torch.where(buffer_completion)
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
                self.pose_buffer_count[task] = torch.clamp(
                    self.pose_buffer_count[task] + count, max=env.cfg.success_buffer_size
                )

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
        proximity_threshold: float,
        touch_distance_threshold: float,
        slightly_open_fraction: float,
        almost_open_fraction: float,
        slightly_open_threshold: float | None,
        almost_open_threshold: float | None,
        require_graspable_pose: bool,
        cabinet_cfg: SceneEntityCfg,
        drawer_joint_name: str,
        ee_frame_cfg: SceneEntityCfg,
        cabinet_frame_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        ee_frame = env.scene[ee_frame_cfg.name]
        ee_tcp_pos = ee_frame.data.target_pos_w[env_ids, 0, :]
        ee_fingertips_w = ee_frame.data.target_pos_w[env_ids, 1:, :]
        lfinger_pos = ee_fingertips_w[..., 0, :]
        rfinger_pos = ee_fingertips_w[..., 1, :]
        handle_pos = env.scene[cabinet_frame_cfg.name].data.target_pos_w[env_ids, 0, :]

        distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)
        is_graspable = (rfinger_pos[:, 2] < handle_pos[:, 2]) & (lfinger_pos[:, 2] > handle_pos[:, 2])

        cabinet: Articulation = env.scene[cabinet_cfg.name]
        drawer_joint_id, _ = cabinet.find_joints([drawer_joint_name])
        drawer_pos = cabinet.data.joint_pos[env_ids][:, drawer_joint_id[0]]
        # thresholds default to fractions of the drawer's own runtime travel range (see class docstring for
        # why this must not always be a fixed distance in meters -- the cabinet's spawn scale differs per
        # robot config), but an explicit absolute threshold takes precedence when given.
        if slightly_open_threshold is None or almost_open_threshold is None:
            drawer_limits = cabinet.data.soft_joint_pos_limits[env_ids][:, drawer_joint_id[0], :]
            drawer_lower, drawer_upper = drawer_limits[:, 0], drawer_limits[:, 1]
            drawer_range = drawer_upper - drawer_lower
        slightly_open_value = (
            slightly_open_threshold if slightly_open_threshold is not None else drawer_lower + slightly_open_fraction * drawer_range
        )
        almost_open_value = (
            almost_open_threshold if almost_open_threshold is not None else drawer_lower + almost_open_fraction * drawer_range
        )

        sub_task_1 = distance < proximity_threshold
        sub_task_2 = distance < touch_distance_threshold
        if require_graspable_pose:
            sub_task_2 = sub_task_2 & is_graspable
        sub_task_3 = drawer_pos > slightly_open_value
        sub_task_4 = drawer_pos > almost_open_value

        return torch.stack([sub_task_1, sub_task_2, sub_task_3, sub_task_4], dim=1)

    def _get_world(
        self, env: ManagerBasedRLEnv, env_ids: torch.Tensor, robot_cfg: SceneEntityCfg, cabinet_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        robot: Articulation = env.scene[robot_cfg.name]
        cabinet: Articulation = env.scene[cabinet_cfg.name]
        return torch.cat([robot.data.joint_pos[env_ids], cabinet.data.joint_pos[env_ids]], dim=-1)

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

        # put the gaps on the simplex first, then shape them with a real temperature. the old form was
        # gaps.pow(prob_exp).softmax(): gaps live in [0, 1], so raising them to a power pushes them toward
        # zero, and a softmax over values spanning ~0.04 is flat to within 4%. measured on the paper runs
        # the soft branch stayed within 0.05 of uniform for entire runs -- it was uniform sampling, not a
        # soft preference, which left the greedy branch as the only thing shaping the distribution.
        confidence = gaps / gaps.sum().clamp(min=1e-8)
        soft = (confidence / env.cfg.softmax_temperature).softmax(dim=0)

        top2 = torch.topk(confidence, k=2)
        winner = top2.indices[0]
        margin = top2.values[0] - top2.values[1]

        # `gaps - gaps.min()` forces one entry to zero, so confidence spreads over at most NUM_SUBTASKS-1
        # non-zero entries. under an uninformative difficulty signal E[margin] is exactly 1/(NUM_SUBTASKS-1),
        # so dividing by that makes the margin comparable across tasks with different subtask counts: ~1
        # means "no more informative than noise", >1 means one subtask genuinely stands out.
        margin_ref = 1.0 / max(NUM_SUBTASKS - 1, 1)
        margin_norm = margin / margin_ref

        hard = torch.zeros_like(gaps)
        hard[winner] = 1.0

        # traverse soft -> greedy across a band instead of saturating. the old clamp(margin / 0.10) hit its
        # ceiling on ~85% of updates, making the controller bang-bang between one-hot and uniform.
        blend = torch.clamp(
            (margin_norm - env.cfg.greedy_margin_lo)
            / max(env.cfg.greedy_margin_hi - env.cfg.greedy_margin_lo, 1e-8),
            0.0,
            1.0,
        )
        distribution = (1.0 - blend) * soft + blend * hard

        # floor every subtask so none is starved to exactly zero: a subtask that stops being sampled stops
        # generating buffer entries and stops being re-evaluated, which makes the collapse permanent.
        eps = env.cfg.min_subtask_prob
        distribution = (1.0 - eps * NUM_SUBTASKS) * distribution + eps
        self.distribution = distribution / distribution.sum()

        self._log["curriculum/blend"] = blend.item()
        self._log["curriculum/margin"] = margin.item()
        self._log["curriculum/margin_norm"] = margin_norm.item()
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
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet"),
) -> None:
    """Overrides the default reset for a sampled fraction of ``env_ids`` with a replayed subtask state.

    Must be registered on ``EventCfg`` *after* the default reset terms (``reset_scene_to_default`` /
    ``reset_robot_joints``) so that, for the picked environments, this term's writes are the ones that stick.
    When :attr:`CabinetEnvCfg.reset_state_curriculum_enabled` is False, this is a no-op.
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

    # a subtask may only be replayed once the policy has actually reached it at least once; until then its
    # buffer holds all-zero joint positions, and replaying those teleports the arm to a configuration no
    # rollout ever visited. The difficulty rule concentrates on whichever subtask the policy reaches *least*,
    # i.e. exactly the one most likely to have an empty buffer, so without this gate the curriculum spends its
    # budget on garbage states precisely when it can least afford to.
    valid_subtasks = tracker.pose_buffer_count > 0
    if not valid_subtasks.any():
        # nothing recorded yet anywhere -- leave every environment on its default reset
        tracker.is_curriculum_episode[env_ids] = False
        tracker.curriculum_subtask[env_ids] = -1
        tracker._log["curriculum/sample_rate"] = 0.0
        tracker._log["curriculum/natural"] = tracker.is_curriculum_episode.float().mean().item()
        tracker._log["curriculum/valid_subtasks"] = 0.0
        return

    tracker.is_curriculum_episode[env_ids] = False
    tracker.is_curriculum_episode[env_ids[picked]] = True
    tracker.curriculum_subtask[env_ids] = -1

    if not picked.any():
        tracker._log["curriculum/sample_rate"] = 0.0
        tracker._log["curriculum/natural"] = tracker.is_curriculum_episode.float().mean().item()
        tracker._log["curriculum/valid_subtasks"] = valid_subtasks.float().sum().item()
        return

    robot: Articulation = env.scene[robot_cfg.name]
    cabinet: Articulation = env.scene[cabinet_cfg.name]
    picked_ids = env_ids[picked]
    num_picked = picked_ids.numel()

    # restrict the sampling distribution to subtasks that have recorded poses; if the curriculum has
    # committed all of its mass to subtasks that have none, fall back to uniform over the ones that do
    masked_distribution = torch.where(valid_subtasks, tracker.distribution, torch.zeros_like(tracker.distribution))
    if masked_distribution.sum() <= 0:
        masked_distribution = valid_subtasks.float()
    masked_distribution = masked_distribution / masked_distribution.sum()

    subtasks = torch.multinomial(masked_distribution, num_picked, replacement=True)
    tracker.curriculum_subtask[picked_ids] = subtasks

    # draw only from slots that have actually been written for the chosen subtask
    max_valid = tracker.pose_buffer_count[subtasks].clamp(min=1)
    world_ids = (torch.rand(num_picked, device=env.device) * max_valid.float()).long()
    world_ids = world_ids.clamp(max=env.cfg.success_buffer_size - 1)
    worlds = tracker.success_buffer[subtasks, world_ids]

    num_robot_joints = robot.num_joints
    robot_joint_pos = worlds[:, :num_robot_joints] + sample_uniform(
        -env.cfg.curriculum_dr, env.cfg.curriculum_dr, (num_picked, num_robot_joints), env.device
    )
    joint_pos_limits = robot.data.soft_joint_pos_limits[picked_ids]
    robot_joint_pos = torch.clamp(robot_joint_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1])
    robot_joint_vel = torch.zeros_like(robot_joint_pos)
    robot.write_joint_state_to_sim(robot_joint_pos, robot_joint_vel, env_ids=picked_ids)

    cabinet_joint_pos = worlds[:, num_robot_joints:]
    cabinet_joint_vel = torch.zeros_like(cabinet_joint_pos)
    cabinet.write_joint_state_to_sim(cabinet_joint_pos, cabinet_joint_vel, env_ids=picked_ids)

    default_robot_joint_pos = robot.data.default_joint_pos[env_ids]
    all_robot_joint_pos = default_robot_joint_pos.clone()
    all_robot_joint_pos[picked] = robot_joint_pos
    distance = torch.norm(all_robot_joint_pos - default_robot_joint_pos, dim=1)
    variance = (all_robot_joint_pos - default_robot_joint_pos).pow(2).mean()

    tracker._log["curriculum/reset_distance"] = distance.mean().item()
    tracker._log["curriculum/reset_variance"] = variance.item()
    tracker._log["curriculum/sample_rate"] = picked.float().mean().item()
    tracker._log["curriculum/natural"] = tracker.is_curriculum_episode.float().mean().item()
    tracker._log["curriculum/valid_subtasks"] = valid_subtasks.float().sum().item()
    for i in range(NUM_SUBTASKS):
        tracker._log[f"curriculum/buffer_fill_{i + 1}"] = tracker.pose_buffer_count[i].item()
