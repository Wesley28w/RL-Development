# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Map milestone difficulty categories to empirical reset states.

Completion bank ``i`` stores states AFTER milestone ``i``. A difficulty-driven
category ``i`` therefore replays bank ``i - 1`` to practice reaching milestone
``i``. Category zero retains the task's normal reset; it never indexes bank -1.
TSCL categories already name reset banks directly and bypass this mapping.
"""

import torch


def natural_reset_ids(env, env_ids: torch.Tensor, tracker_term_name: str = "subtask_progression_tracker"):
    """Exclude already-planned replay episodes from the task's original reset events."""
    if not env.cfg.reset_state_curriculum_enabled:
        return env_ids
    tracker = getattr(env.reward_manager.cfg, tracker_term_name).func
    return env_ids[~tracker.is_curriculum_episode[env_ids]]


def available_reset_targets(counts: torch.Tensor, *, use_predecessor: bool = True) -> torch.Tensor:
    """Return availability in the scheduler's category coordinates, not bank coordinates."""
    if not use_predecessor:
        return counts > 0
    available = torch.ones_like(counts, dtype=torch.bool)
    available[1:] = counts[:-1] > 0
    return available


def sample_reset_worlds(
    banks: torch.Tensor,
    counts: torch.Tensor,
    targets: torch.Tensor,
    *,
    use_predecessor: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a replay mask over ``targets`` and the corresponding unperturbed worlds.

    False mask entries retain their normal reset. Only populated slots may be
    sampled; selecting an empty required bank is an error, not a zero-state reset.
    """
    bank_ids = targets - 1 if use_predecessor else targets
    replay = bank_ids >= 0
    bank_ids = bank_ids[replay]
    if bank_ids.numel() == 0:
        return replay, banks.new_empty((0, banks.shape[-1]))
    valid_counts = counts[bank_ids]
    if bool((valid_counts <= 0).any()):
        raise ValueError("Cannot replay an unpopulated predecessor/reset bank.")
    slots = (torch.rand(bank_ids.numel(), device=banks.device) * valid_counts).long()
    slots = torch.minimum(slots, valid_counts - 1)
    return replay, banks[bank_ids, slots]
