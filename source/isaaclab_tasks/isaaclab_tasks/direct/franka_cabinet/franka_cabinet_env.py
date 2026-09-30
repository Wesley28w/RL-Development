# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import torch

from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform

from .rcg_cfg import RCGCfg
from .rcg_mixin import RCGMixin, StatePool

DEFAULT_GOAL_STATE_PATH = os.path.join(os.path.dirname(__file__), "data", "goal_states_franka_cabinet.pt")
"""Default location of the recorded goal states, written by ``scripts/rcg/record_goal_states.py``."""


@configclass
class FrankaCabinetEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 9
    observation_space = 23
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # cabinet
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0, 0.4),
            rot=(0.1, 0.0, 0.0, 0.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0

    # task success: how far the top drawer must be pulled out
    drawer_open_threshold = 0.39

    # smoothing for the episodic success-rate metric, applied once per completed episode
    success_rate_ema_alpha = 0.01

    # reverse curriculum generation; disabled here so this configuration is the unmodified
    # baseline, and enabled by FrankaCabinetRCGEnvCfg below. The goal-state path is set on this
    # configuration rather than only on the RCG one because it is never read while 'enabled' is
    # False, and because the RCG tooling (the recorder, the gate test) runs against the baseline
    # task with the curriculum switched off and still has to know where the file lives.
    rcg: RCGCfg = RCGCfg(goal_state_path=DEFAULT_GOAL_STATE_PATH)


@configclass
class FrankaCabinetRCGEnvCfg(FrankaCabinetEnvCfg):
    """Franka Cabinet with reverse curriculum generation over start states.

    Identical to :class:`FrankaCabinetEnvCfg` in every respect that affects the MDP -- same
    observations, same dense reward, same termination, same episode length -- so that the two
    can be benchmarked against each other. Only the *start-state distribution* differs.
    """

    rcg: RCGCfg = RCGCfg(
        enabled=True,
        goal_state_path=DEFAULT_GOAL_STATE_PATH,
    )


class FrankaCabinetEnv(RCGMixin, DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaCabinetEnvCfg

    def __init__(self, cfg: FrankaCabinetEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
            self.device,
        )

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.drawer_link_idx = self._cabinet.find_bodies("drawer_top")[0][0]
        self.drawer_joint_idx = self._cabinet.find_joints("drawer_top_joint")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.drawer_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.drawer_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # -- success metrics, logged in both arms of the benchmark
        self._success_log: dict[str, torch.Tensor] = {}
        # EMA over completed episodes, so the curve is readable without a separate eval pass
        self._episode_success_rate = torch.zeros((), device=self.device)
        self._episodes_completed = torch.zeros((), device=self.device)
        self._eval_success_rate = torch.zeros((), device=self.device)
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

        # reverse curriculum generation buffers; must come last, since the mixin derives the
        # start-state schema by calling _rcg_capture_state on an empty index set
        self._rcg_init_buffers()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._cabinet = Articulation(self.cfg.cabinet)
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["cabinet"] = self._cabinet

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        drawer_pos = self._cabinet.data.joint_pos[:, self.drawer_joint_idx]
        # via the shared helper, so the termination test and the success test cannot drift apart
        terminated = self._rcg_is_solved()
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        # graded progress toward the goal, 0.0 = closed, 1.0 = open past the threshold. Defined
        # identically to the reset-pose-curriculum branch's 'dones/success_rate_margin' so that
        # curves from the two branches can be overlaid directly.
        margin = torch.clamp(drawer_pos / self.cfg.drawer_open_threshold, 0.0, 1.0)
        self._success_log["dones/success_rate_margin"] = margin.mean()
        if bool(self._is_eval_env.any()):
            self._success_log["dones/eval_success_rate_margin"] = margin[self._is_eval_env].mean()

        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        rewards = self._compute_rewards(
            self.actions,
            self._cabinet.data.joint_pos,
            self.robot_grasp_pos,
            self.drawer_grasp_pos,
            self.robot_grasp_rot,
            self.drawer_grasp_rot,
            robot_left_finger_pos,
            robot_right_finger_pos,
            self.gripper_forward_axis,
            self.drawer_inward_axis,
            self.gripper_up_axis,
            self.drawer_up_axis,
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.open_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.finger_reward_scale,
            self._robot.data.joint_pos,
        )

        # the paper's sparse objective, r(s) = 1{s in S^g}. The dense terms above stay in the
        # logs either way so that the two variants remain comparable.
        if self.cfg.rcg.sparse_reward:
            rewards = self._rcg_is_solved().float()

        # merge the metrics in here: _compute_rewards assigns extras["log"] wholesale, so
        # anything written to it earlier in the step (by _get_dones or _reset_idx) is discarded
        self.extras["log"].update(self._success_log)
        if self.cfg.rcg.enabled:
            self.extras["log"].update(self.rcg_log)

        return rewards

    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        to_target = self.drawer_grasp_pos - self.robot_grasp_pos

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                to_target,
                self._cabinet.data.joint_pos[:, self.drawer_joint_idx].unsqueeze(-1),
                self._cabinet.data.joint_vel[:, self.drawer_joint_idx].unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # reset

    def _reset_idx(self, env_ids: torch.Tensor):
        # attribute the outcome of the finishing episodes to the start states that produced
        # them, and snapshot any success state, before this episode's state is wiped
        self._record_rcg_episode_results(env_ids)
        self._rcg_record_goal_states(env_ids)
        self._update_success_metrics(env_ids)

        super()._reset_idx(env_ids)

        if self.rcg_active:
            # environments held out of the curriculum keep starting from rho_0, so that something
            # comparable to the baseline is measurable while the curriculum runs
            eval_mask = self._is_eval_env[env_ids]
            if bool(eval_mask.any()):
                self._normal_reset(env_ids[eval_mask])
                curriculum_ids = env_ids[~eval_mask]
            else:
                curriculum_ids = env_ids
            if curriculum_ids.numel() > 0:
                self._rcg_reset_from_pool(curriculum_ids)
        else:
            self._normal_reset(env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

    def _update_success_metrics(self, env_ids: torch.Tensor):
        """Track binary per-episode success for the episodes finishing this step.

        ``reset_terminated`` is the task's success condition, already computed by
        :meth:`~isaaclab.envs.DirectRLEnv.step` for this step, so success is read rather than
        recomputed. Kept as an EMA over *completed episodes* -- a per-step mean would weight
        steps rather than episodes, and most steps end no episode at all.
        """
        successes = self.reset_terminated[env_ids].float()
        if successes.numel() == 0:
            return

        alpha = self.cfg.success_rate_ema_alpha

        def update(current: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
            # applying a per-episode alpha to a batch of n episodes at once
            keep = (1.0 - alpha) ** batch.numel()
            return current * keep + batch.mean() * (1.0 - keep)

        self._episode_success_rate = update(self._episode_success_rate, successes)
        self._episodes_completed += successes.numel()
        self._success_log["dones/success_rate"] = self._episode_success_rate
        self._success_log["dones/episodes_completed"] = self._episodes_completed

        if bool(self._is_eval_env.any()):
            eval_successes = self.reset_terminated[env_ids][self._is_eval_env[env_ids]].float()
            if eval_successes.numel() > 0:
                self._eval_success_rate = update(self._eval_success_rate, eval_successes)
                self._eval_episodes_completed += eval_successes.numel()
            self._success_log["dones/eval_success_rate"] = self._eval_success_rate
            self._success_log["dones/eval_episodes_completed"] = self._eval_episodes_completed

    def _normal_reset(self, env_ids: torch.Tensor):
        """The task's default start distribution, rho_0 (unchanged from upstream)."""
        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        # note: upstream sets the articulation's target but leaves self.robot_dof_targets at
        # its previous-episode value, even though _pre_physics_step integrates from it. That
        # leaks state across episodes, so the buffer is reset here too. Applied to this path as
        # well as the curriculum path so that both arms of the benchmark share identical reset
        # semantics; set rcg.reset_dof_targets = False to recover the upstream behaviour.
        if self.cfg.rcg.reset_dof_targets:
            self.robot_dof_targets[env_ids] = joint_pos
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # cabinet state
        zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

    # reverse curriculum generation hooks

    def _rcg_capture_state(self, env_ids: torch.Tensor) -> StatePool:
        """Capture the start state.

        By default, joint **positions** only: deliberately the same 13 numbers the reset-pose
        curriculum stores (9 arm/gripper joint positions + 4 cabinet joint positions, see its
        ``success_buffer``), so that neither curriculum carries state the other does not and the
        benchmark comparison is not confounded. Velocities and the action-target buffer are then not
        stored, and are set to a canonical rest condition on restore, exactly as the reset-pose
        curriculum does.

        That default is a documented deviation from Florensa et al., whose start states are genuine
        visited states including velocity. Restoring a position-only state is still physically valid
        -- it is the same class of state the task's own reset produces -- but it is not a replay of
        the captured moment, which is why the gate test's dynamics-equivalence check does not gate at
        this setting. ``rcg.capture_full_state = True`` stores the rest and makes it a real gate; see
        that flag's docstring.

        All fields are joint coordinates, so no conversion between world and environment-local
        frames is needed. Both articulations are fixed-base and their root poses are never
        written, so the root state is not part of the task's state either.
        """
        state = {
            # arm and gripper
            "robot_joint_pos": self._robot.data.joint_pos[env_ids].clone(),
            # all four cabinet joints, not just the top drawer
            "cabinet_joint_pos": self._cabinet.data.joint_pos[env_ids].clone(),
        }
        if self.cfg.rcg.capture_full_state:
            state["robot_joint_vel"] = self._robot.data.joint_vel[env_ids].clone()
            state["cabinet_joint_vel"] = self._cabinet.data.joint_vel[env_ids].clone()
            # the environment's own integrator state: _pre_physics_step integrates from this buffer,
            # not from the articulation's commanded target, so a continuation that does not restore
            # it is not the same continuation
            state["robot_dof_targets"] = self.robot_dof_targets[env_ids].clone()
        return state

    def _rcg_restore_state(self, env_ids: torch.Tensor, state: StatePool) -> None:
        """Restore a start state. Inverse of :meth:`_rcg_capture_state`.

        Fields the pool does not carry are restored to a canonical rest, which is what makes the
        position-only default a valid state rather than an arbitrary one.
        """
        robot_joint_pos = state["robot_joint_pos"]
        cabinet_joint_pos = state["cabinet_joint_pos"]

        # at rest, matching the task's own reset (`joint_vel = torch.zeros_like(robot_joint_pos)`)
        robot_joint_vel = state.get("robot_joint_vel")
        if robot_joint_vel is None:
            robot_joint_vel = torch.zeros_like(robot_joint_pos)
        cabinet_joint_vel = state.get("cabinet_joint_vel")
        if cabinet_joint_vel is None:
            cabinet_joint_vel = torch.zeros_like(cabinet_joint_pos)

        # Re-anchor the action integrator on the pose being restored, rather than replaying the
        # captured target. This matches what the reset-pose curriculum does
        # (`self.robot_dof_targets[env_ids] = robot_joint_pos`), so both curricula carry the same
        # information in their start states and the comparison is not confounded by RCG having
        # extra state. Note: not literally zeroed -- commanding every joint to 0 would slam the
        # arm across its workspace on every reset.
        dof_targets = state.get("robot_dof_targets")
        self.robot_dof_targets[env_ids] = robot_joint_pos if dof_targets is None else dof_targets
        self._robot.set_joint_position_target(self.robot_dof_targets[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(robot_joint_pos, robot_joint_vel, env_ids=env_ids)

        # the cabinet's joint targets are never commanded away from their defaults, so the
        # drawer's restoring spring is part of the task dynamics and nothing to restore here
        self._cabinet.write_joint_state_to_sim(cabinet_joint_pos, cabinet_joint_vel, env_ids=env_ids)

    def _rcg_is_solved(self) -> torch.Tensor:
        """Binary task success: the top drawer is open past the threshold.

        This is also the task's termination condition, so the two cannot drift apart.
        """
        return self._cabinet.data.joint_pos[:, self.drawer_joint_idx] > self.cfg.drawer_open_threshold

    def _rcg_apply_state_noise(self, env_ids: torch.Tensor, std: float) -> None:
        """Optional Brownian noise on the cabinet joints (documented deviation from the paper).

        The paper's random walk acts purely through the action space. For this task the drawer
        is under-actuated from the arm's point of view, so if the diagnostics ever show that
        action-space noise alone does not move the drawer away from the goal, this provides a
        state-space component. Off by default.
        """
        joint_pos = self._cabinet.data.joint_pos[env_ids]
        noise = torch.randn_like(joint_pos) * std
        lower = self._cabinet.data.soft_joint_pos_limits[env_ids, :, 0]
        upper = self._cabinet.data.soft_joint_pos_limits[env_ids, :, 1]
        joint_pos = torch.clamp(joint_pos + noise, lower, upper)
        self._cabinet.write_joint_state_to_sim(joint_pos, self._cabinet.data.joint_vel[env_ids], env_ids=env_ids)

    def _rcg_pool_diagnostics(self, pool: StatePool) -> dict[str, torch.Tensor | float]:
        """How far along the task the current start-state pool sits.

        ``rcg/pool_drawer_mean`` falling across stages is the direct evidence that the
        curriculum is expanding backwards away from the goal.
        """
        if self._pool_size(pool) == 0:
            return {}
        drawer = pool["cabinet_joint_pos"][:, self.drawer_joint_idx]
        return {
            "rcg/pool_drawer_mean": drawer.mean(),
            "rcg/pool_drawer_max": drawer.max(),
            "rcg/pool_drawer_min": drawer.min(),
        }

    def _rcg_pool_progress(self, pool: StatePool) -> torch.Tensor:
        """Drawer opening as a fraction of the success threshold, clipped to ``[0, 1]``.

        The same quantity as the ``dones/success_rate_margin`` logged during training, so the
        gate test's histogram and the training curve are measured on one scale.
        """
        drawer = pool["cabinet_joint_pos"][:, self.drawer_joint_idx]
        return torch.clamp(drawer / self.cfg.drawer_open_threshold, 0.0, 1.0)

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        drawer_pos = self._cabinet.data.body_pos_w[env_ids, self.drawer_link_idx]
        drawer_rot = self._cabinet.data.body_quat_w[env_ids, self.drawer_link_idx]
        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
            self.drawer_grasp_rot[env_ids],
            self.drawer_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
            drawer_rot,
            drawer_pos,
            self.drawer_local_grasp_rot[env_ids],
            self.drawer_local_grasp_pos[env_ids],
        )

    def _compute_rewards(
        self,
        actions,
        cabinet_dof_pos,
        franka_grasp_pos,
        drawer_grasp_pos,
        franka_grasp_rot,
        drawer_grasp_rot,
        franka_lfinger_pos,
        franka_rfinger_pos,
        gripper_forward_axis,
        drawer_inward_axis,
        gripper_up_axis,
        drawer_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        open_reward_scale,
        action_penalty_scale,
        finger_reward_scale,
        joint_positions,
    ):
        # distance from hand to the drawer
        d = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
        axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

        dot1 = (
            torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of forward axis for gripper
        dot2 = (
            torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the cabinet has been opened out
        open_reward = cabinet_dof_pos[:, self.drawer_joint_idx]  # drawer_top_joint

        # penalty for distance of each finger from the drawer handle
        lfinger_dist = franka_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2]
        rfinger_dist = drawer_grasp_pos[:, 2] - franka_rfinger_pos[:, 2]
        finger_dist_penalty = torch.zeros_like(lfinger_dist)
        finger_dist_penalty += torch.where(lfinger_dist < 0, lfinger_dist, torch.zeros_like(lfinger_dist))
        finger_dist_penalty += torch.where(rfinger_dist < 0, rfinger_dist, torch.zeros_like(rfinger_dist))

        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + open_reward_scale * open_reward
            + finger_reward_scale * finger_dist_penalty
            - action_penalty_scale * action_penalty
        )

        self.extras["log"] = {
            "dist_reward": (dist_reward_scale * dist_reward).mean(),
            "rot_reward": (rot_reward_scale * rot_reward).mean(),
            "open_reward": (open_reward_scale * open_reward).mean(),
            "action_penalty": (-action_penalty_scale * action_penalty).mean(),
            "left_finger_distance_reward": (finger_reward_scale * lfinger_dist).mean(),
            "right_finger_distance_reward": (finger_reward_scale * rfinger_dist).mean(),
            "finger_dist_penalty": (finger_reward_scale * finger_dist_penalty).mean(),
        }

        # bonus for opening drawer properly
        drawer_pos = cabinet_dof_pos[:, self.drawer_joint_idx]
        rewards = torch.where(drawer_pos > 0.01, rewards + 0.25, rewards)
        rewards = torch.where(drawer_pos > 0.2, rewards + 0.25, rewards)
        rewards = torch.where(drawer_pos > 0.35, rewards + 0.25, rewards)

        return rewards

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos
