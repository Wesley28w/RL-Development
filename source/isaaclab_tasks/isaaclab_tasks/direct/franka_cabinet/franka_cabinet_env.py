# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

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

from isaaclab_tasks.utils.reverse_curriculum import ReverseCurriculum


@configclass
class FrankaCabinetEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 9
    observation_space = 23
    state_space = 0

    # reverse curriculum generation (Florensa et al., CoRL 2017)
    reverse_curriculum_enabled = False
    rcg_goal_state_path = ""  # .pt file from scripts/tools/capture_rcg_goal_state.py
    rcg_pool_size = 512
    # R_min/R_max and the 2:1 new:old pool ratio (see reverse_curriculum.py) match the reference
    # implementation's arm-manipulation experiments exactly (curriculum/experiments/starts/arm3d/
    # {arm3d_key,arm3d_disc}_brownian.py both use min_reward=0.1, max_reward=0.9).
    rcg_r_min = 0.1
    rcg_r_max = 0.9
    rcg_n_new = None  # defaults to 2/3 * rcg_pool_size, see reverse_curriculum.py
    rcg_n_old = None  # defaults to rcg_pool_size - rcg_n_new
    rcg_min_episodes_per_state = 5
    rcg_replay_history_size = 2000
    # 50 matches arm3d_key_brownian.py's fixed value (arm3d_disc's sibling experiment sweeps
    # {20, 50, 100} -- 50 is the shared "standard" choice across both).
    rcg_brownian_horizon = 50
    # fraction of the full [-1, 1] normalized action range sampled uniformly during expansion.
    # 1.0 matches the reference exactly: curriculum/envs/start_env.py's brownian() samples
    # np.random.uniform(*env.action_space.bounds) -- i.e. full-range uniform noise, not a small
    # Gaussian perturbation (its `variance` parameter is accepted but never actually used in that
    # function -- confirmed by reading it -- so "brownian_variance" in the reference configs is
    # vestigial and full-range-uniform is what the original experiments actually ran).
    rcg_action_noise_scale = 1.0
    rcg_snapshot_interval = 1  # collect a candidate every N brownian steps
    # Derived, not guessed: this task trains for num_steps_per_env(16) * max_iterations(1500) =
    # 24000 common_step_counter ticks total (see agents/rsl_rl_ppo_cfg.py). A generation update
    # costs brownian_horizon * decimation physics ticks, i.e. the GPU-time equivalent of
    # brownian_horizon=50 ordinary steps. Capping expansion overhead at ~10% of total training
    # compute -> at most 24000*0.10/50 = 48 generations over the whole run -> interval >=
    # 24000/48 = 500. The reference itself regenerates on essentially every outer iteration
    # (outer_iters=5000 == total generations for the whole run) -- we can't match that cadence at
    # this brownian_horizon without expansion dominating wall-clock time, since our per-step
    # sample count (num_envs=4096) is far larger than the reference's CPU-rollout-worker setup.
    rcg_update_interval = 500
    rcg_expansion_cohort_size = 256  # number of envs used as expansion workers per phase
    rcg_eval_fraction = 0.1  # fraction of envs permanently held out on the default reset for eval

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


class FrankaCabinetEnv(DirectRLEnv):
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

        self.is_curriculum_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._done_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._success_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if self.cfg.reverse_curriculum_enabled:
            if not self.cfg.rcg_goal_state_path:
                raise ValueError(
                    "reverse_curriculum_enabled=True but cfg.rcg_goal_state_path is empty. Run"
                    " scripts/tools/capture_rcg_goal_state.py against a converged baseline checkpoint"
                    " to produce a goal-state file first, then point this cfg field at it."
                )
            goal_states = torch.load(self.cfg.rcg_goal_state_path, map_location=self.device)
            goal_states = torch.as_tensor(goal_states, device=self.device, dtype=torch.float32)

            self.rcg = ReverseCurriculum(
                state_dim=13,  # 9 robot joint pos + 4 cabinet joint pos, see _get_world()
                pool_size=self.cfg.rcg_pool_size,
                num_envs=self.num_envs,
                device=self.device,
                r_min=self.cfg.rcg_r_min,
                r_max=self.cfg.rcg_r_max,
                n_new=self.cfg.rcg_n_new,
                n_old=self.cfg.rcg_n_old,
                min_episodes_per_state=self.cfg.rcg_min_episodes_per_state,
                replay_history_size=self.cfg.rcg_replay_history_size,
            )
            self.rcg.seed(goal_states)
            self._rcg_expansion_sim_steps = 0  # cumulative extra physics ticks spent on expansion
            self._rcg_last_candidates_collected = 0

            # a fixed, permanent slice of envs never receives a curriculum reset --
            # this is the "evaluate from the original reset distribution" sleeve the guide
            # calls for, kept vectorized/continuous instead of a separate periodic eval pass.
            n_eval = max(1, int(self.num_envs * self.cfg.rcg_eval_fraction))
            self._rcg_eval_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._rcg_eval_env_mask[:n_eval] = True

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
        if (
            self.cfg.reverse_curriculum_enabled
            and self.common_step_counter > 0
            and self.common_step_counter % self.cfg.rcg_update_interval == 0
        ):
            # runs entirely before this step's real actions are processed, and fully restores
            # the scene before falling through below -- rollout collection never sees it.
            self._run_rcg_expansion_phase()

        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        drawer_pos = self._cabinet.data.joint_pos[:, self.drawer_joint_idx]
        terminated = drawer_pos > 0.39
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        self._done_mask = terminated | truncated
        self._success_mask = terminated  # the only early-termination condition here is success
        return terminated, truncated

    def _get_world(self) -> torch.Tensor:
        """Full resettable scene state: 9 robot joint positions + 4 cabinet joint positions."""
        return torch.cat([self._robot.data.joint_pos, self._cabinet.data.joint_pos], dim=1)

    def _is_valid_world(self, world: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(world).all(dim=1)
        robot_pos = world[:, 0:9]
        within_limits = ((robot_pos >= self.robot_dof_lower_limits) & (robot_pos <= self.robot_dof_upper_limits)).all(
            dim=1
        )
        cabinet_pos = world[:, 9:13]
        cabinet_ok = (cabinet_pos >= -0.01).all(dim=1) & (cabinet_pos <= 0.42).all(dim=1)
        return finite & within_limits & cabinet_ok

    def _run_rcg_expansion_phase(self) -> None:
        """Generate this generation's RCG candidate states without touching rollout-visible
        state: advances physics directly (bypassing _get_dones/_get_rewards/_reset_idx/
        _get_observations and the episode/common step counters), then restores every env to
        its pre-phase state so on-policy collection resumes exactly where it left off.
        """
        # 1. snapshot everything, not just the expansion cohort -- sim.step() advances the
        # whole shared scene each tick, so non-cohort envs would drift too if left unsnapshotted.
        snap_robot_pos = self._robot.data.joint_pos.clone()
        snap_robot_vel = self._robot.data.joint_vel.clone()
        snap_robot_targets = self.robot_dof_targets.clone()
        snap_cabinet_pos = self._cabinet.data.joint_pos.clone()
        snap_cabinet_vel = self._cabinet.data.joint_vel.clone()

        # 2. pick a cohort and seed it from the current good-start frontier
        cohort_size = min(self.cfg.rcg_expansion_cohort_size, self.num_envs)
        cohort = torch.randperm(self.num_envs, device=self.device)[:cohort_size]

        seeds = self.rcg.select_good_starts()
        worlds = seeds[torch.randint(0, seeds.shape[0], (cohort_size,), device=self.device)]

        seed_robot_pos = snap_robot_pos.clone()
        seed_robot_pos[cohort] = worlds[:, 0:9]
        seed_robot_vel = snap_robot_vel.clone()
        seed_robot_vel[cohort] = 0.0
        seed_cabinet_pos = snap_cabinet_pos.clone()
        seed_cabinet_pos[cohort] = worlds[:, 9:13]
        seed_cabinet_vel = snap_cabinet_vel.clone()
        seed_cabinet_vel[cohort] = 0.0

        self._robot.write_joint_state_to_sim(seed_robot_pos, seed_robot_vel)
        self._cabinet.write_joint_state_to_sim(seed_cabinet_pos, seed_cabinet_vel)
        self.robot_dof_targets[:] = seed_robot_pos
        self._robot.set_joint_position_target(self.robot_dof_targets)

        # 3. Brownian rollout: perturb only the cohort's targets, replay raw physics ticks --
        # the same four calls DirectRLEnv.step() makes per decimation substep, called directly.
        candidates: list[torch.Tensor] = []
        for step in range(self.cfg.rcg_brownian_horizon):
            # uniform over the full normalized action range, matching the reference
            # implementation's brownian() -- see the rcg_action_noise_scale cfg comment.
            noise = (torch.rand((cohort_size, self._robot.num_joints), device=self.device) * 2.0 - 1.0)
            noise *= self.cfg.rcg_action_noise_scale
            cohort_targets = (
                self.robot_dof_targets[cohort]
                + self.robot_dof_speed_scales[cohort] * self.dt * noise * self.cfg.action_scale
            )
            self.robot_dof_targets[cohort] = torch.clamp(
                cohort_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits
            )

            for _ in range(self.cfg.decimation):
                self._apply_action()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=self.physics_dt)

            if (step + 1) % self.cfg.rcg_snapshot_interval == 0:
                world = self._get_world()
                valid = self._is_valid_world(world)[cohort]
                if valid.any():
                    candidates.append(world[cohort][valid].detach().clone())

        # 4. restore every env to its pre-phase state -- nothing about this phase is visible
        # once real rollout collection resumes below.
        self._robot.write_joint_state_to_sim(snap_robot_pos, snap_robot_vel)
        self._cabinet.write_joint_state_to_sim(snap_cabinet_pos, snap_cabinet_vel)
        self.robot_dof_targets[:] = snap_robot_targets
        self._robot.set_joint_position_target(self.robot_dof_targets)
        self._compute_intermediate_values()

        # 5. rebuild the pool from whatever this phase collected
        all_candidates = torch.cat(candidates, dim=0) if candidates else torch.zeros((0, 13), device=self.device)
        self.rcg.rebuild_pool(all_candidates)
        self._rcg_expansion_sim_steps += self.cfg.rcg_brownian_horizon * self.cfg.decimation
        self._rcg_last_candidates_collected = all_candidates.shape[0]

    def _update_reverse_curriculum(self) -> None:
        done_ids = self._done_mask.nonzero(as_tuple=True)[0]
        if len(done_ids) > 0:
            self.rcg.record_episode_results(done_ids, self._success_mask[done_ids])

        if hasattr(self, "extras") and "log" in self.extras:
            self.extras["log"].update(self.rcg.stats())
            self.extras["log"]["rcg/expansion_sim_steps_cumulative"] = float(self._rcg_expansion_sim_steps)
            self.extras["log"]["rcg/last_expansion_candidates_collected"] = float(
                self._rcg_last_candidates_collected
            )

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

        # note: _compute_rewards() above just (re)created self.extras["log"] as a fresh dict for
        # this step, so success-rate logging has to happen after it, not in _get_dones() -- _get_dones()
        # runs before _get_rewards() each step and anything it wrote to extras["log"] would be
        # discarded by _compute_rewards()'s dict replacement.
        drawer_pos = self._cabinet.data.joint_pos[:, self.drawer_joint_idx]
        overall_success = torch.clamp(drawer_pos / 0.39, 0.0, 1.0)
        if self.cfg.reverse_curriculum_enabled:
            # success must be measured on the default (non-curriculum) reset distribution,
            # never on curriculum resets that started partway to the goal.
            eval_mask = self._rcg_eval_env_mask
            success_for_metric = overall_success[eval_mask] if eval_mask.any() else overall_success
        else:
            success_for_metric = overall_success
        self.extras["log"]["dones/success_rate"] = success_for_metric.mean().item()

        if self.cfg.reverse_curriculum_enabled:
            self._update_reverse_curriculum()

        return rewards

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        n = len(env_ids)

        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (n, self._robot.num_joints),
            self.device,
        )
        # cabinet state
        cabinet_state = torch.zeros((n, self._cabinet.num_joints), device=self.device)

        self.is_curriculum_episode[env_ids] = False

        if self.cfg.reverse_curriculum_enabled:
            # every trainable reset draws from the curriculum pool -- unconditional, matching
            # the paper's own reset_source=curriculum.active_training_starts. Expansion is no
            # longer reset-triggered at all; see _run_rcg_expansion_phase.
            local_idx = torch.arange(n, device=self.device)
            trainable_local = local_idx[~self._rcg_eval_env_mask[env_ids]]
            if len(trainable_local) > 0:
                worlds = self.rcg.sample_for_resets(env_ids[trainable_local])
                joint_pos[trainable_local] = worlds[:, 0:9]
                cabinet_state[trainable_local] = worlds[:, 9:13]
                self.is_curriculum_episode[env_ids[trainable_local]] = True

        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        cabinet_vel = torch.zeros_like(cabinet_state)
        self._cabinet.write_joint_state_to_sim(cabinet_state, cabinet_vel, env_ids=env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

        if self.cfg.reverse_curriculum_enabled and hasattr(self, "extras") and "log" in self.extras:
            self.extras["log"]["rcg/reset_curriculum_fraction"] = (
                self.is_curriculum_episode[env_ids].float().mean().item()
            )

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
