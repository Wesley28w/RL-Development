# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The two arms of the Franka Lift reverse-curriculum benchmark.

Both use the same environment class, :class:`~..lift_rcg_env.LiftRCGEnv`, and the same MDP. The
only difference between them is one boolean: where an episode starts.

:class:`FrankaCubeLiftBaselineEnvCfg`
    Plain PPO from the task's own start distribution ``rho_0``, plus the success metrics.

:class:`FrankaCubeLiftRCGEnvCfg`
    The same thing with the reverse curriculum switched on.

Why a separate baseline id rather than reusing ``Isaac-Lift-Cube-Franka-v0``: that id stays exactly
as upstream ships it, so nothing outside this benchmark changes behaviour, and the *measured*
baseline is still code-identical to the RCG arm rather than merely configured to look like it. The
two differ from upstream Franka Lift only by the ``rcg_success_tracker`` reward term, whose
contribution to the reward is exactly zero and which draws nothing from the RNG -- see
``lift/mdp/rcg.py``.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import RewardsCfg

from .joint_pos_env_cfg import FrankaCubeLiftEnvCfg


@configclass
class LiftRCGRewardsCfg(RewardsCfg):
    """Upstream's reward terms, plus the per-step goal tracker.

    ``rcg_success_tracker`` returns exact zeros for every environment on every step, so the reward
    signal is untouched. The weight has to be non-zero only because ``RewardManager.compute`` skips
    zero-weight terms and a term that is never called tracks nothing.
    """

    rcg_success_tracker = RewTerm(func=mdp.rcg_success_tracker, weight=1.0)


@configclass
class FrankaCubeLiftBaselineEnvCfg(FrankaCubeLiftEnvCfg):
    """PPO baseline arm: ``rho_0`` starts, with the benchmark's success metrics."""

    rewards: LiftRCGRewardsCfg = LiftRCGRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rcg.enabled = False


@configclass
class FrankaCubeLiftRCGEnvCfg(FrankaCubeLiftBaselineEnvCfg):
    """RCG arm: start states drawn from the reverse curriculum pool.

    Identical to :class:`FrankaCubeLiftBaselineEnvCfg` in every respect that affects the MDP -- same
    observations, same dense reward, same terminations, same episode length -- so that a comparison
    between the two isolates the effect of the curriculum.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rcg.enabled = True


@configclass
class FrankaCubeLiftRCGEnvCfg_PLAY(FrankaCubeLiftRCGEnvCfg):
    """Small scene for interactive inspection.

    ``play.py`` forces ``rcg.enabled = False`` regardless, so this always shows the policy on the
    task's own start distribution -- a curriculum start distribution is a training device, and
    watching a policy succeed from a state that was handed to it says nothing.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
