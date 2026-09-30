# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reverse Curriculum Generation, shared by every task that implements it.

Reference:
    C. Florensa, D. Held, M. Wulfmeier, M. Zhang, P. Abbeel,
    "Reverse Curriculum Generation for Reinforcement Learning", CoRL 2017.
    https://arxiv.org/abs/1707.05300

RCG changes **where episodes start** and nothing else: not the reward, not the observation, not
the termination condition, not the policy. Training begins from states close to a known goal
state ``s^g`` and the start distribution expands backwards as the policy improves, so the agent
always trains on starts it can sometimes but not always solve.

Two tasks currently use it, and they share this implementation rather than a copy of it, which
is what makes a claim about "the same curriculum on two tasks" checkable:

* ``isaaclab_tasks.direct.franka_cabinet`` -- direct workflow.
* ``isaaclab_tasks.manager_based.manipulation.lift`` -- manager-based workflow.

Adding a third is a matter of picking the adapter for its workflow and implementing three hooks:
:meth:`~.rcg_mixin.RCGMixin._rcg_capture_state`,
:meth:`~.rcg_mixin.RCGMixin._rcg_restore_state` and
:meth:`~.rcg_mixin.RCGMixin._rcg_is_solved`.
"""

from .rcg_cfg import RCGCfg
from .rcg_mixin import RCGMixin, StatePool
from .workflow_mixins import RCGDirectMixin, RCGManagerBasedMixin

__all__ = ["RCGCfg", "RCGDirectMixin", "RCGManagerBasedMixin", "RCGMixin", "StatePool"]
