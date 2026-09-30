# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Workflow adapters for :class:`~isaaclab_tasks.utils.rcg.rcg_mixin.RCGMixin`.

:class:`RCGMixin` never touches the environment loop directly. Everything it needs from it is
declared there as four hooks, and supplied here once per workflow:

=========================== ======================================================================
hook                        why the curriculum needs it
=========================== ======================================================================
``_rcg_action_dim``         to draw Brownian actions ``a_t ~ N(0, Sigma)`` of the right width
``_rcg_physics_step``       to run ``SampleNearby`` without the RL algorithm ever seeing the steps
``_rcg_reset_all_envs``     to teleport every environment at a stage boundary
``_rcg_episode_success``    to estimate ``R(pi_i, s_0)`` for a finished episode
=========================== ======================================================================

A task mixes in the adapter for its workflow:

.. code-block:: python

    class FrankaCabinetEnv(RCGDirectMixin, DirectRLEnv): ...


    class LiftRCGEnv(RCGManagerBasedMixin, ManagerBasedRLEnv): ...

Both adapters deliberately reimplement the *physics* portion of their workflow's ``step()`` and
nothing else. No rewards, no terminations, no resets, no observation computation, and in
particular no update of ``episode_length_buf`` or ``common_step_counter``: a Brownian rollout is
not part of any episode and must not age one.
"""

from __future__ import annotations

import torch

from .rcg_mixin import RCGMixin


class RCGDirectMixin(RCGMixin):
    """Reverse curriculum generation for a :class:`~isaaclab.envs.DirectRLEnv` subclass."""

    @property
    def _rcg_action_dim(self) -> int:
        return self.actions.shape[-1]

    def _rcg_physics_step(self, actions: torch.Tensor) -> None:
        """The decimated physics portion of :meth:`~isaaclab.envs.DirectRLEnv.step`."""
        self._pre_physics_step(actions)
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

    def _rcg_reset_all_envs(self) -> None:
        """Follows :meth:`~isaaclab.envs.DirectRLEnv.reset`."""
        # allocate a fresh action buffer rather than zeroing in place: the RL runner collects its
        # rollout inside torch.inference_mode(), so `_pre_physics_step` has rebound self.actions
        # to an inference tensor, and an in-place write to one from out here is an error
        self.actions = torch.zeros(self.actions.shape, dtype=self.actions.dtype, device=self.device)
        self._reset_idx(self._rcg_all_env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.obs_buf = self._get_observations()

    def _rcg_episode_success(self, env_ids: torch.Tensor) -> torch.Tensor:
        """``reset_terminated``, which is the success condition for a task that terminates on it.

        :meth:`~isaaclab.envs.DirectRLEnv.step` has already computed it from ``_get_dones`` for
        this step, so success is read rather than recomputed. A direct-workflow task that does
        *not* terminate on success must override this.
        """
        return self.reset_terminated[env_ids]


class RCGManagerBasedMixin(RCGMixin):
    """Reverse curriculum generation for a :class:`~isaaclab.envs.ManagerBasedRLEnv` subclass.

    The manager-based workflow has no per-step override point on the environment class, so a
    task whose success is not one of its termination terms has to accumulate that flag from a
    manager term and override :meth:`_rcg_episode_success`. See
    ``isaaclab_tasks.manager_based.manipulation.lift.mdp.rcg`` for the shape of that.
    """

    @property
    def _rcg_action_dim(self) -> int:
        return self.action_manager.total_action_dim

    def _rcg_physics_step(self, actions: torch.Tensor) -> None:
        """The decimated physics portion of :meth:`~isaaclab.envs.ManagerBasedRLEnv.step`.

        ``command_manager.compute`` is deliberately *not* called. It decrements each command
        term's ``time_left`` and resamples the command when it runs out, and a Brownian rollout
        is not episode time -- letting it run would hand a candidate state a different goal from
        the seed state it was expanded from, which is exactly the thing that makes a start
        state's success probability ill-defined.
        """
        self.action_manager.process_action(actions.to(self.device))
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

    def _rcg_reset_all_envs(self) -> None:
        """Follows :meth:`~isaaclab.envs.ManagerBasedEnv.reset`.

        ``_reset_idx`` runs every manager's ``reset``, which is what zeroes the action history --
        so unlike the direct adapter there is no action buffer to reallocate here.
        """
        self._reset_idx(self._rcg_all_env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        # no command_manager.compute: the reset above has already resampled every command, and
        # the task's restore has then overwritten it with the pool's own goal
        self.obs_buf = self.observation_manager.compute(update_history=True)

    def _rcg_episode_success(self, env_ids: torch.Tensor) -> torch.Tensor:
        """``reset_terminated``, for a task whose success *is* one of its termination terms.

        :meth:`~isaaclab.envs.ManagerBasedRLEnv.step` has already computed it from the
        termination manager for this step. A task that does not terminate on success -- the
        common case in the manager-based workflow, where success is usually measured rather than
        terminated on -- must override this.
        """
        return self.reset_terminated[env_ids]
