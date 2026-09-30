# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Re-export of the shared RCG configuration.

:class:`~isaaclab_tasks.utils.rcg.rcg_cfg.RCGCfg` used to live here, when Franka Cabinet was the
only task with a reverse curriculum. It moved to ``isaaclab_tasks.utils.rcg`` when Franka Lift
was added, so that the two benchmark tasks share one implementation rather than a copy of it.
This module stays behind so that ``from .rcg_cfg import RCGCfg`` keeps resolving, and because a
recorded goal-state file or a Hydra override written against the old path still names ``rcg``
fields that have not changed.
"""

from isaaclab_tasks.utils.rcg import RCGCfg

__all__ = ["RCGCfg"]
