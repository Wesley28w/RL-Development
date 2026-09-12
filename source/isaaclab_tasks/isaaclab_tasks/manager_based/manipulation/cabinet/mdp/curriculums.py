# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms that also double as TensorBoard-logged training metrics for the cabinet task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class cabinet_success_rate(ManagerTermBase):
    """Logs the raw (unsmoothed) success rate for opening the drawer.

    An environment counts as a success on reset if the drawer joint position ends up past
    ``success_fraction`` of the drawer joint's own runtime travel range (``soft_joint_pos_limits``), read at
    call time rather than assumed as a fixed distance in meters -- the cabinet's spawn scale (and therefore
    its true drawer travel) differs between robot configs (e.g. OpenArm's 0.75x-scaled cabinet vs. Franka's
    unscaled one; scaling a USD actor scales its prismatic joint limits along with it), so a hardcoded meter
    threshold tuned for one would silently be wrong -- possibly unreachable -- for the other. Since the only
    termination is time-out, this fires "success" only for episodes where the drawer was actually opened --
    an agent that never approaches the handle stays at drawer position 0 and is correctly scored as a
    failure.

    Environments currently replaying a reset-pose-curriculum state (see ``mdp/events.py``) are excluded from
    this metric: those episodes were teleported into an already-partially-opened state, so crediting them as
    a "success" the policy earned on its own would inflate the number -- there is no separate eval phase here
    to measure true performance, so this *is* the number that has to be trustworthy. When the curriculum is
    disabled (or hasn't marked anything as a curriculum episode), every environment is "natural", so this
    exclusion has no effect. If a whole batch of resets on a given call happens to be entirely curriculum
    episodes (rare, since curriculum sampling only takes a fraction of resets), the last known natural-only
    value is returned instead of fabricating a misleading number from zero natural samples.

    This is a read-only curriculum term: it is only ever used for its return value, which the curriculum
    manager logs to TensorBoard as ``Curriculum/cabinet_success_rate``. It does not touch any reward,
    observation, or termination term, so it has no effect on the reward signal, the policy's inputs, or when
    episodes end -- i.e. no effect on training.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._last_value = torch.zeros((), device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        success_fraction: float = 0.90,
        tracker_term_name: str = "subtask_progression_tracker",
        cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet"),
        drawer_joint_name: str = "drawer_bottom_joint",
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

        cabinet: Articulation = env.scene[cabinet_cfg.name]
        drawer_joint_id, _ = cabinet.find_joints([drawer_joint_name])
        drawer_pos = cabinet.data.joint_pos[natural_ids, drawer_joint_id[0]]
        drawer_limits = cabinet.data.soft_joint_pos_limits[natural_ids, drawer_joint_id[0], :]
        threshold = drawer_limits[:, 0] + success_fraction * (drawer_limits[:, 1] - drawer_limits[:, 0])
        self._last_value = (drawer_pos > threshold).float().mean()
        return self._last_value


def reset_pose_curriculum_metrics(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    tracker_term_name: str = "subtask_progression_tracker",
) -> dict[str, float]:
    """Surfaces the reset-pose curriculum's internal bookkeeping (see ``mdp/events.py``) to TensorBoard.

    This reaches into the :class:`isaaclab_tasks.manager_based.manipulation.cabinet.mdp.events.subtask_progression_tracker`
    event term (the same reflection pattern the Lift task's curriculum port uses) and returns its latest
    cached stats dict, which the curriculum manager logs under
    ``Curriculum/reset_pose_curriculum_metrics/<key>``. Returns an empty dict (nothing logged) when the
    curriculum is disabled, matching that term's own no-op behavior.
    """
    if not env.cfg.reset_state_curriculum_enabled:
        return {}
    # subtask_progression_tracker is registered as a reward term (see mdp/events.py's module docstring for
    # why), so it must be looked up through the reward manager rather than the event manager.
    tracker = getattr(env.reward_manager.cfg, tracker_term_name).func
    return dict(tracker._log)
