# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Lift with Reverse Curriculum Generation over start states.

Reference:
    C. Florensa, D. Held, M. Wulfmeier, M. Zhang, P. Abbeel,
    "Reverse Curriculum Generation for Reinforcement Learning", CoRL 2017.
    https://arxiv.org/abs/1707.05300

The curriculum itself lives in :mod:`isaaclab_tasks.utils.rcg` and is shared with Franka Cabinet;
this module supplies only what is specific to lifting a cube:

* the three hooks ``_rcg_capture_state`` / ``_rcg_restore_state`` / ``_rcg_is_solved``,
* an override of ``_rcg_episode_success``, because this task does not terminate on success,
* the reset path, which is the only thing that differs between the two benchmark arms,
* the success metrics both arms log.

One environment class serves both arms. ``Isaac-Lift-Cube-Franka-Baseline-v0`` runs it with
``rcg.enabled = False`` and ``Isaac-Lift-Cube-Franka-RCG-v0`` with ``rcg.enabled = True``; nothing
else differs between them, so a comparison isolates the start-state distribution. With the
curriculum off, this class performs **zero** extra ``torch`` RNG draws and leaves every reset event,
reward, observation and termination exactly as upstream Franka Lift computes them -- see
``mdp/rcg.py`` for why that is worth being careful about.

Three things about this task that RCG has to be told about, and that Franka Cabinet did not have:

**The goal is part of the state.** ``object_pose`` is a
:class:`~isaaclab.envs.mdp.commands.UniformPoseCommand` resampled at every reset, and it enters both
the observation and the reward. A start state restored without its goal would be paired with a fresh
random one, so ``R(pi_i, s_0)`` -- the success probability *of that start state*, which is the entire
basis of the good-start criterion -- would not be a property of the start state at all. The
commanded pose therefore travels with the state, in the robot's root frame exactly as the command
buffer holds it. This is a deliberate difference from the reset-pose curriculum, which replays a
recorded pose against a freshly sampled goal.

**There is a free-floating object.** ``_rcg_capture_state`` stores the cube's position relative to
``scene.env_origins``, because environment clones sit at different world offsets and storing an
absolute world position is the easiest way to break a state pool silently.

**Nothing terminates on success.** Upstream Franka Lift ends an episode only on time-out or on
dropping the object, and that is left untouched here -- adding a success termination would change
the task rather than the curriculum. Success is instead measured every step by the
``rcg_success_tracker`` reward term (which contributes exactly zero to the reward) and scored per
episode according to :attr:`~isaaclab_tasks.utils.rcg.rcg_cfg.RCGCfg.episode_success_mode`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.commands import UniformPoseCommand
from isaaclab.utils.math import combine_frame_transforms

from isaaclab_tasks.utils.rcg import RCGManagerBasedMixin, StatePool

from .lift_env_cfg import LiftEnvCfg


class LiftRCGEnv(RCGManagerBasedMixin, ManagerBasedRLEnv):
    """Franka Lift, able to draw its start states from a reverse curriculum."""

    cfg: LiftEnvCfg

    def __init__(self, cfg: LiftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # -- scene handles, resolved once
        self._robot: Articulation = self.scene["robot"]
        self._object: RigidObject = self.scene["object"]
        self._goal_command: UniformPoseCommand = self.command_manager.get_term("object_pose")

        # The arm is fixed-base and its root pose is never written, so the transform from an
        # environment's origin to its robot's root frame is a constant, identical in every clone.
        # Cached here so that a *pool* of states -- which has no environment index -- can still be
        # reported in one frame. Nothing in the curriculum depends on it; only the diagnostics do.
        self._root_pos_local = (self._robot.data.root_pos_w[0] - self.scene.env_origins[0]).clone()
        self._root_quat = self._robot.data.root_quat_w[0].clone()

        # -- success bookkeeping, written every step by the rcg_success_tracker reward term
        # instantaneous goal test; in `_reset_idx` it still holds the *final* step's value for the
        # episodes that are ending, which is terminal success
        self.rcg_currently_solved = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # the same thing made sticky for the duration of an episode: "did this episode ever reach
        # the goal". Needed because the object can be carried to the goal and then dropped, and
        # because with no success termination every episode runs to its time limit.
        self.rcg_episode_solved = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # -- metrics, logged by both arms of the benchmark
        self._success_log: dict[str, torch.Tensor] = {}
        # EMAs over completed episodes, so the curves are readable without a separate eval pass
        self._terminal_success_rate = torch.zeros((), device=self.device)
        self._ever_success_rate = torch.zeros((), device=self.device)
        self._episodes_completed = torch.zeros((), device=self.device)
        self._eval_terminal_success_rate = torch.zeros((), device=self.device)
        self._eval_ever_success_rate = torch.zeros((), device=self.device)
        self._eval_episodes_completed = torch.zeros((), device=self.device)

        # environments held out of the curriculum and always reset from rho_0. A contiguous block
        # rather than a random draw, so the split is identical across seeds and runs.
        self._is_eval_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        num_eval = int(round(self.cfg.rcg.eval_env_fraction * self.num_envs))
        num_eval = max(0, min(num_eval, self.num_envs - 1))
        if num_eval > 0:
            self._is_eval_env[:num_eval] = True
            print(
                f"[RCG] Holding {num_eval} of {self.num_envs} environments out of the curriculum; they always reset"
                " from the task's own start distribution and drive the 'dones/eval_*' metrics."
            )

        # reverse curriculum buffers; must come last, since the mixin derives the start-state schema
        # by calling _rcg_capture_state on an empty index set
        self._rcg_init_buffers()

    ##
    # Reset.
    ##

    def _reset_idx(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        # attribute the outcome of the finishing episodes to the start states that produced them,
        # and snapshot any success state, before this episode's state is wiped
        self._record_rcg_episode_results(env_ids)
        self._rcg_record_goal_states(env_ids)
        self._update_success_metrics(env_ids)

        # the task's own reset: reset events (rho_0), then every manager, then episode_length_buf.
        # The command manager resamples `object_pose` in here, which is why a curriculum restore has
        # to come afterwards to overwrite it with the pool's own goal.
        super()._reset_idx(env_ids)

        self.rcg_episode_solved[env_ids] = False
        self.rcg_currently_solved[env_ids] = False

        if self.rcg_active:
            # environments held out of the curriculum keep starting from rho_0 -- which is what
            # super() has just given them -- so that something comparable to the baseline is
            # measurable while the curriculum runs
            eval_mask = self._is_eval_env[env_ids]
            curriculum_ids = env_ids[~eval_mask] if bool(eval_mask.any()) else env_ids
            if curriculum_ids.numel() > 0:
                self._rcg_reset_from_pool(curriculum_ids)
        # with the curriculum inactive there is nothing to do: super() already applied rho_0

        # `super()._reset_idx` reassigns `extras["log"]`, so the metrics have to be merged back in
        # afterwards -- but not *only* here, see `publish_rcg_log`
        self.publish_rcg_log()

    def publish_rcg_log(self) -> None:
        """Merge the RCG and success metrics into ``extras["log"]``.

        Called from ``_reset_idx`` (which reassigns ``extras["log"]`` and so would otherwise drop
        them) *and* from the ``rcg_success_tracker`` reward term on every step. Both are needed, for
        two separate reasons.

        rsl-rl's logger only records the keys present in ``extras["log"]`` on the **first step** of a
        learning iteration, and averages each one over the steps where it appears. In the manager-based
        workflow ``extras["log"]`` is otherwise rebuilt only on steps where something resets -- and
        immediately after a curriculum stage boundary nothing does, because ``advance_rcg_stage``
        teleports every environment at once and none of them finishes an episode again for a full
        250 steps. At 24 steps per iteration that is ten iterations during which ``rcg/stage`` and
        ``rcg/pool_*`` would report the *previous* stage's values. Publishing every step keeps the
        curves aligned with the ``[RCG] Stage N: ...`` console lines instead of trailing an episode
        behind them.

        Reading :attr:`rcg_log` recomputes the within-stage diagnostics, which is deliberately cheap:
        every value stays a 0-dim device tensor, so this costs no host/device synchronisation.
        """
        log = self.extras.setdefault("log", {})
        log.update(self._success_log)
        if self.cfg.rcg.enabled:
            log.update(self.rcg_log)

    def _update_success_metrics(self, env_ids: torch.Tensor):
        """Track per-episode success for the episodes finishing this step.

        Two rates, the same pair Franka Cabinet reports:

        * ``dones/success_rate`` -- the episode *ended* holding the object at the goal. The primary
          metric. It has no approach-phase ceiling and cannot be earned by touching the goal once
          and dropping the cube afterwards.
        * ``dones/success_rate_ever`` -- the episode reached the goal at any point. The lenient
          counterpart; it says whether the task is being solved at all, not how reliably.

        Both are EMAs over *completed episodes*. A per-step mean would weight steps rather than
        episodes, and most steps end no episode at all.
        """
        # `episode_length_buf` is zeroed at the end of `super()._reset_idx`, so it still holds the
        # finished episode's length here. Zero means this is an environment's very first reset,
        # which is not a completed episode.
        finished = self.episode_length_buf[env_ids] > 0
        if not bool(finished.any()):
            return

        alpha = self.cfg.success_rate_ema_alpha

        def update(current: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
            # applying a per-episode alpha to a batch of n episodes at once
            keep = (1.0 - alpha) ** batch.numel()
            return current * keep + batch.mean() * (1.0 - keep)

        ids = env_ids[finished]
        terminal = self.rcg_currently_solved[ids].float()
        ever = self.rcg_episode_solved[ids].float()

        self._terminal_success_rate = update(self._terminal_success_rate, terminal)
        self._ever_success_rate = update(self._ever_success_rate, ever)
        self._episodes_completed += terminal.numel()
        self._success_log["dones/success_rate"] = self._terminal_success_rate
        self._success_log["dones/success_rate_ever"] = self._ever_success_rate
        self._success_log["dones/episodes_completed"] = self._episodes_completed

        if bool(self._is_eval_env.any()):
            eval_ids = ids[self._is_eval_env[ids]]
            if eval_ids.numel() > 0:
                self._eval_terminal_success_rate = update(
                    self._eval_terminal_success_rate, self.rcg_currently_solved[eval_ids].float()
                )
                self._eval_ever_success_rate = update(
                    self._eval_ever_success_rate, self.rcg_episode_solved[eval_ids].float()
                )
                self._eval_episodes_completed += eval_ids.numel()
            self._success_log["dones/eval_success_rate"] = self._eval_terminal_success_rate
            self._success_log["dones/eval_success_rate_ever"] = self._eval_ever_success_rate
            self._success_log["dones/eval_episodes_completed"] = self._eval_episodes_completed

    ##
    # Reverse curriculum generation hooks.
    ##

    def _rcg_capture_state(self, env_ids: torch.Tensor) -> StatePool:
        """Capture the start state: arm pose, cube pose, and the commanded goal.

        ``object_pos`` is stored **relative to** ``scene.env_origins``, since environment clones sit
        at different world offsets. ``goal_pose_b`` is stored in the robot's root frame, which is
        exactly how the command term holds it, so capture and restore are bit-exact inverses.

        Velocities are stored only when :attr:`~isaaclab_tasks.utils.rcg.rcg_cfg.RCGCfg.capture_full_state`
        is set; by default they are not, and are zeroed on restore. That default matches Franka
        Cabinet and the reset-pose curriculum, so no curriculum carries state another does not -- but
        it is a documented deviation from Florensa et al., and it is the reason the gate test's
        dynamics-equivalence check does not gate at the default setting. See that flag's docstring.

        There is no action-target buffer to store, unlike Franka Cabinet. This task's
        ``JointPositionAction`` writes an absolute target, ``action * scale + default_joint_pos``,
        on every step, so it integrates nothing across episodes and nothing leaks.
        """
        state = {
            # 7 arm joints + 2 gripper fingers
            "robot_joint_pos": self._robot.data.joint_pos[env_ids].clone(),
            "object_pos": self._object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids],
            "object_quat": self._object.data.root_quat_w[env_ids].clone(),
            # (x, y, z, qw, qx, qy, qz) in the robot's root frame -- the goal is part of the state
            "goal_pose_b": self._goal_command.pose_command_b[env_ids].clone(),
        }
        if self.cfg.rcg.capture_full_state:
            state["robot_joint_vel"] = self._robot.data.joint_vel[env_ids].clone()
            # linear and angular velocity of the cube's centre of mass, which is what
            # `write_root_velocity_to_sim` takes, so capture and restore stay symmetric
            state["object_vel"] = self._object.data.root_vel_w[env_ids].clone()
        return state

    def _rcg_restore_state(self, env_ids: torch.Tensor, state: StatePool) -> None:
        """Write a captured state back into the simulation. Inverse of :meth:`_rcg_capture_state`.

        Fields the pool does not carry are restored to a canonical rest, which is what makes the
        position-only default a valid state rather than an arbitrary one.
        """
        robot_joint_pos = state["robot_joint_pos"]
        num = len(env_ids)
        robot_joint_vel = state.get("robot_joint_vel")
        if robot_joint_vel is None:
            robot_joint_vel = torch.zeros_like(robot_joint_pos)
        self._robot.write_joint_state_to_sim(robot_joint_pos, robot_joint_vel, env_ids=env_ids)

        # note: no joint-position *target* is written. The action term computes an absolute target
        # from the policy's action every step, and `apply_action` runs before the first physics step
        # of the next episode, so the stale target from the previous episode is never integrated.
        object_pos_w = state["object_pos"] + self.scene.env_origins[env_ids]
        self._object.write_root_pose_to_sim(torch.cat([object_pos_w, state["object_quat"]], dim=-1), env_ids=env_ids)
        object_vel = state.get("object_vel")
        if object_vel is None:
            object_vel = torch.zeros((num, 6), device=self.device)
        self._object.write_root_velocity_to_sim(object_vel, env_ids=env_ids)

        # the goal travels with the start state. `time_left` is deliberately left at what the
        # command manager's own reset set it to: the restored state begins a *new* episode, so its
        # resampling clock starts from the full interval, exactly as a natural episode's does.
        self._goal_command.pose_command_b[env_ids] = state["goal_pose_b"]

    def _rcg_is_solved(self) -> torch.Tensor:
        """Binary task success: the object is within ``success_threshold`` of the commanded goal.

        The same test the ``rcg_success_tracker`` reward term evaluates every step and the same one
        the curriculum uses to reject already-solved candidates, so the two cannot drift apart. Not
        a termination condition -- see this module's docstring.
        """
        des_pos_w, _ = combine_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self.command_manager.get_command("object_pose")[:, :3],
        )
        distance = torch.norm(des_pos_w - self._object.data.root_pos_w, dim=-1)
        return distance < self.cfg.success_threshold

    def _rcg_episode_success(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Per-episode success, as selected by ``rcg.episode_success_mode``.

        ``"ever"`` (the default for this task) is the paper's reading: its episodes end the moment
        the goal set is entered, so ``R(pi_i, s_0)`` is the probability of *reaching* the goal from
        ``s_0``. Franka Lift never terminates on success, so under ``"terminal"`` a start state one
        step from the goal would instead be scored on whether the policy can *hold* the cube in
        place for a full 250-step episode -- a much harder question, and one that answers ``0`` for
        nearly every start early in training, which leaves ``select()`` empty and stalls the
        curriculum at stage 0.
        """
        if self.cfg.rcg.episode_success_mode == "ever":
            return self.rcg_episode_solved[env_ids]
        return self.rcg_currently_solved[env_ids]

    def _rcg_pool_diagnostics(self, pool: StatePool) -> dict[str, torch.Tensor | float]:
        """How far along the task the current start-state pool sits.

        ``rcg/pool_goal_distance_mean`` rising and ``rcg/pool_object_height_mean`` falling across
        stages is the direct evidence that the curriculum is expanding backwards away from the goal.
        """
        if self._pool_size(pool) == 0:
            return {}
        distance = self._pool_goal_distance(pool)
        height = pool["object_pos"][:, 2]
        return {
            "rcg/pool_goal_distance_mean": distance.mean(),
            "rcg/pool_goal_distance_min": distance.min(),
            "rcg/pool_goal_distance_max": distance.max(),
            "rcg/pool_object_height_mean": height.mean(),
        }

    def _rcg_pool_progress(self, pool: StatePool) -> torch.Tensor:
        """Object-to-goal distance mapped onto ``[0, 1]``, where ``1`` means "on top of the goal".

        ``0`` is :attr:`~.lift_env_cfg.LiftEnvCfg.progress_reference_distance`, roughly the distance
        at a fresh ``rho_0`` start, so a value near ``0`` means a start state no further along the
        task than a normal episode's. Purely a reporting quantity.
        """
        reference = max(self.cfg.progress_reference_distance, 1e-6)
        return torch.clamp(1.0 - self._pool_goal_distance(pool) / reference, 0.0, 1.0)

    def _pool_goal_distance(self, pool: StatePool) -> torch.Tensor:
        """Object-to-goal distance for every state in a pool, in metres.

        The pool has no environment index, so the goal -- stored in the robot's root frame -- is
        brought into the environment-local frame with the cached constant root transform.
        """
        num_states = self._pool_size(pool)
        goal_pos_local, _ = combine_frame_transforms(
            self._root_pos_local.unsqueeze(0).expand(num_states, 3),
            self._root_quat.unsqueeze(0).expand(num_states, 4),
            pool["goal_pose_b"][:, :3],
            pool["goal_pose_b"][:, 3:],
        )
        return torch.norm(goal_pos_local - pool["object_pos"], dim=-1)
