# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The one per-step hook Reverse Curriculum Generation needs on the lift task.

Franka Lift does not terminate on success -- its only termination terms are ``time_out`` and
``object_dropping`` -- so ``reset_terminated`` carries no information about whether the task was
achieved, and the curriculum has to accumulate that itself. That requires a per-step evaluation of
the goal condition, and the manager-based workflow offers no per-step override point on the
environment class the way ``DirectRLEnv._get_dones`` does.

:func:`rcg_success_tracker` is that evaluation, registered as a **reward** term. It returns
exactly zero for every environment on every step, so it can never change the reward signal, and it
touches no observation and no termination. What it does is call the environment's own
``_rcg_is_solved()`` -- the single definition of the goal condition, shared with the curriculum's
candidate filtering -- latch the result into two buffers, and refresh the logged metrics (see
:meth:`~..lift_rcg_env.LiftRCGEnv.publish_rcg_log`, which explains why every step and not just
every reset):

* ``env.rcg_currently_solved`` -- the instantaneous goal test. Read from ``_reset_idx``, where it
  still holds the *final* step's value for the episodes that are ending, which is terminal success.
* ``env.rcg_episode_solved`` -- the same thing made sticky for the duration of an episode, i.e.
  "did this episode ever reach the goal". This is the paper's reading of ``R(pi_i, s_0)``, whose
  episodes end the moment the goal set is entered.

Why a reward term and not an ``EventTermCfg(mode="interval")``: registering any interval event
makes :class:`~isaaclab.managers.EventManager` draw from the global torch RNG once per reset and
once per step, unconditionally, regardless of what the term does. That shift alone is enough to
make the baseline arm diverge from an unmodified run, because everything else drawing on the same
generator moves downstream of it. :class:`~isaaclab.managers.RewardManager` has no hidden timer, so
the term costs zero RNG draws and the baseline stays comparable to upstream Franka Lift.

The weight has to be non-zero: ``RewardManager.compute`` skips a term whose weight is ``0.0`` as a
micro-optimisation, and a term that is never called tracks nothing. ``1.0`` times a tensor of
exact zeros times ``dt`` is still exactly zero, so the reward is unaffected either way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rcg_success_tracker(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Latch the goal condition into the environment's per-episode success buffers.

    Returns:
        A tensor of exact zeros, shape ``(num_envs,)``. This term's contribution to the reward is
        always zero; it exists only for its side effect on the buffers described in the module
        docstring.
    """
    solved = getattr(env, "_rcg_is_solved", None)
    if solved is not None:
        flags = solved()
        # in place, into buffers the environment allocated outside inference mode: the RL runner
        # calls RewardManager.compute from inside torch.inference_mode(), and rebinding these
        # attributes to inference tensors here would make them unwritable from a plain scripted
        # loop later on
        env.rcg_currently_solved[:] = flags
        env.rcg_episode_solved |= flags
        # this term is also the only thing that runs every step, which is what the metrics need --
        # see LiftRCGEnv.publish_rcg_log for why publishing them only from _reset_idx is not enough
        env.publish_rcg_log()
    return torch.zeros(env.num_envs, device=env.device)
