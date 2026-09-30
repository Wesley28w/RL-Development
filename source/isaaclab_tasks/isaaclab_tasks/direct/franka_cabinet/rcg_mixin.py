# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Re-export of the shared RCG mixin, bound to the direct workflow.

The implementation used to live here, when Franka Cabinet was the only task with a reverse
curriculum. It moved to ``isaaclab_tasks.utils.rcg`` when Franka Lift was added -- the
curriculum logic was already task-agnostic, and the only parts that were not were the four
places it touches the environment loop, which are now supplied per workflow by
:class:`~isaaclab_tasks.utils.rcg.workflow_mixins.RCGDirectMixin` and
:class:`~isaaclab_tasks.utils.rcg.workflow_mixins.RCGManagerBasedMixin`.

``RCGMixin`` here is an alias for the *direct-workflow* adapter, which is what this task always
meant by it, so ``class FrankaCabinetEnv(RCGMixin, DirectRLEnv)`` is unchanged in behaviour.
"""

from isaaclab_tasks.utils.rcg import RCGDirectMixin, StatePool

RCGMixin = RCGDirectMixin
"""The direct-workflow RCG adapter, under the name this task has always used for it."""

__all__ = ["RCGMixin", "StatePool"]
