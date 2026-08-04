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
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class lift_success_rate(ManagerTermBase):
    """Logs the raw (unsmoothed) success rate for bringing the object to its commanded target pose.

    An environment counts as a success on reset if the object ends up within ``threshold`` of the
    commanded goal position. Since the episode only ever ends via time-out or by dropping the object,
    this only fires "success" for episodes where the object was actually carried to the goal -- an
    agent that never picks up the object stays far from the goal and is correctly scored as a failure,
    while an agent that drops the object ends up far below/away from the goal as well.

    Environments currently replaying a reset-pose-curriculum state (see ``mdp/events.py``) are excluded from
    this metric: those episodes were teleported into an already-partially-completed state, so crediting them
    as a "success" the policy earned on its own would inflate the number -- there is no separate eval phase
    here to measure true performance, so this *is* the number that has to be trustworthy. When the curriculum
    is disabled (or hasn't marked anything as a curriculum episode), every environment is "natural", so this
    exclusion has no effect. If a whole batch of resets on a given call happens to be entirely curriculum
    episodes (rare, since curriculum sampling only takes a fraction of resets), the last known natural-only
    value is returned instead of fabricating a misleading number from zero natural samples.

    This is a read-only curriculum term: it is only ever used for its return value, which the curriculum
    manager logs to TensorBoard as ``Curriculum/lift_success_rate``. It does not touch any reward,
    observation, or termination term, so it has no effect on the reward signal, the policy's inputs, or when
    episodes end -- i.e. no effect on training. Beyond the curriculum-episode exclusion above, the value is
    the fraction of (natural) environments resetting on this particular call that succeeded, so it is
    intentionally noisy call-to-call (small reset batches swing between 0 and 1) rather than smoothed -- that
    noise is real information about how variable the task's success is across environments and time.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._last_value = torch.zeros((), device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str = "object_pose",
        threshold: float = 0.02,
        tracker_term_name: str = "subtask_progression_tracker",
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

        # subtask_progression_tracker is registered as a reward term (see mdp/events.py's module docstring
        # for why), so it must be looked up through the reward manager rather than the event manager.
        tracker_term_cfg = getattr(env.reward_manager.cfg, tracker_term_name, None)
        if tracker_term_cfg is not None:
            is_curriculum_episode = tracker_term_cfg.func.is_curriculum_episode[env_ids]
            natural_ids = env_ids[~is_curriculum_episode]
        else:
            natural_ids = env_ids

        if natural_ids.numel() == 0:
            return self._last_value

        robot: RigidObject = env.scene[robot_cfg.name]
        object: RigidObject = env.scene[object_cfg.name]
        command = env.command_manager.get_command(command_name)

        # goal position in the world frame for the (natural) environments that just finished an episode
        des_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w[natural_ids], robot.data.root_quat_w[natural_ids], command[natural_ids, :3]
        )
        distance = torch.norm(des_pos_w - object.data.root_pos_w[natural_ids], dim=1)
        self._last_value = (distance < threshold).float().mean()
        return self._last_value


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
