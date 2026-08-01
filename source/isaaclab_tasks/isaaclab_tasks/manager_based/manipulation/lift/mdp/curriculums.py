# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms that also double as TensorBoard-logged training metrics for the lift task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lift_success_rate(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Logs the raw (unsmoothed) success rate for bringing the object to its commanded target pose.

    An environment counts as a success on reset if the object ends up within ``threshold`` of the
    commanded goal position. Since the episode only ever ends via time-out or by dropping the object,
    this only fires "success" for episodes where the object was actually carried to the goal -- an
    agent that never picks up the object stays far from the goal and is correctly scored as a failure,
    while an agent that drops the object ends up far below/away from the goal as well.

    This is a plain (non-parameter-modifying) curriculum term: it is only ever used for its return value,
    which the curriculum manager logs to TensorBoard as ``Curriculum/lift_success_rate``. It does not touch
    any reward, observation, or termination term, so it has no effect on the reward signal, the policy's
    inputs, or when episodes end -- i.e. no effect on training. The value is the fraction of the environments
    resetting on this particular call that succeeded, so it is intentionally noisy call-to-call (small reset
    batches swing between 0 and 1) rather than smoothed -- that noise is real information about how variable
    the task's success is across environments and time.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # goal position in the world frame for the environments that just finished an episode
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w[env_ids], robot.data.root_quat_w[env_ids], command[env_ids, :3]
    )
    distance = torch.norm(des_pos_w - object.data.root_pos_w[env_ids], dim=1)
    return (distance < threshold).float().mean()


def reset_pose_curriculum_metrics(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    tracker_term_name: str = "subtask_progression_tracker",
) -> dict[str, float]:
    """Surfaces the reset-pose curriculum's internal bookkeeping (see ``mdp/events.py``) to TensorBoard.

    This reaches into the :class:`isaaclab_tasks.manager_based.manipulation.lift.mdp.events.subtask_progression_tracker`
    event term (the same reflection pattern dexsuite's ``DifficultyScheduler``/``initial_final_interpolate_fn``
    use to share state between terms) and returns its latest cached stats dict, which the curriculum manager
    logs under ``Curriculum/reset_pose_curriculum_metrics/<key>``. Returns an empty dict (nothing logged) when
    the curriculum is disabled, matching that term's own no-op behavior.
    """
    if not env.cfg.reset_state_curriculum_enabled:
        return {}
    # subtask_progression_tracker is registered as a reward term (see mdp/events.py's module docstring for
    # why), so it must be looked up through the reward manager rather than the event manager.
    tracker = getattr(env.reward_manager.cfg, tracker_term_name).func
    return dict(tracker._log)
