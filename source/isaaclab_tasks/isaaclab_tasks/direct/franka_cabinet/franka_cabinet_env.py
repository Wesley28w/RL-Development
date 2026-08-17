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

@configclass
class FrankaCabinetEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 9
    observation_space = 23
    state_space = 0

    # reset state curriculum
    reset_state_curriculum_enabled = True # True

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

    # robot (15 dim pose)
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

    # cabinet (11 dim pose)
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

    # custom hyperparamters
    success_buffer_size = 64
    prob_exp = 2 # how much we sharpen the probability distribution (1 = No sharpening)
    sampling_ratio = 0.3 # what fraction of resets go to the sample distribution
    curriculum_dr = 0.02 # how much domain randomization to apply to robot joints
    success_rate_alpha = 0.05 # momentum control of success rate movement (pre-calculations)
    greedy_margin = 0.10 # controls the margin between top and second distribution value that enables softmax
    
    # policy params
    curriculum_total_iterations = 2500
    controller_enabled = False
    window_analysis_size = 0.02 # percent to look at
    window_analysis_start = 0.01 # percent to start at
    slope_threshold = 2.0 # what threshold slope will disable curriculum

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

        # added variables for curriculum ---

        # progression: completion, and poses
        self.progression = torch.zeros([self.num_envs, 4, 14], device=self.device) # 4 for the num_subtasks, 13 for (compelted, poses)
        self.success_rate = torch.zeros(4, device=self.device) # 4 is number of subtasks

        # distribution: probabilites for each subtask to sample from
        self.distribution = torch.softmax(torch.ones([4], device=self.device), dim=0) # [0.25, 0.25, 0.25, 0.25]

        # success buffer
        self.success_buffer = torch.zeros([4, self.cfg.success_buffer_size, 13], device=self.device) # 4 subtasks, buffer size of 64, and 13 joint attributes to save 
    
        self.pose_buffer_idx = torch.zeros(
            4,
            dtype=torch.long,
            device=self.device
        )

        # We only want to compute times using episodes that are not biased by the curriculum
        self.is_curriculum_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # for logging sr of replay environments
        self.curriculum_subtask = torch.full(
            (self.num_envs,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        
        # Controller
        self.progress = 0.0 # for tracking progression (0.0-1.0)
        self.curriculum_enabled = self.cfg.reset_state_curriculum_enabled # set to whatever cfg (mutable)
        self.overall_success = 0.0 # final task success (0.0-1.0)
        self.controller_snapshot = None
        self.controller_snapshot_two = None
        self.controller_checked = False

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
        self.actions = actions.clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        drawer_pos = self._cabinet.data.joint_pos[:, self.drawer_joint_idx]
        terminated = drawer_pos > 0.39
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        done = terminated | truncated

        # should log success rate episodically
        if hasattr(self, "extras") and "log" in self.extras:
            L = self.extras["log"]

            # 0.0 = closed, 1.0 = fully open (39 cm)
            self.overall_success = torch.clamp(drawer_pos / 0.39, 0.0, 1.0)
            # exclude environments currently replaying a reset-pose-curriculum state: they were teleported
            # into an already-partially-open drawer, so counting them would inflate the reported success rate.
            # There is no separate eval phase here, so this metric has to reflect only genuinely-earned
            # progress. Fall back to the unfiltered mean on the (very unlikely, given sampling_ratio < 1) step
            # where every single environment happens to be a curriculum replay, so the key is never dropped.
            natural_mask = ~self.is_curriculum_episode
            success_for_metric = self.overall_success[natural_mask] if natural_mask.any() else self.overall_success
            L["dones/success_rate_margin"] = success_for_metric.mean().item()

        return terminated, truncated

    # returns each environment completion of the subtasks [N, 4]
    def _get_subtasks(self) -> torch.Tensor:
        # 20 cm
        sub_task_1 = torch.norm(self.robot_grasp_pos - self.drawer_grasp_pos, p=2, dim=-1) < 0.2
        # 10 cm
        sub_task_2 = (torch.norm(self.robot_grasp_pos - self.drawer_grasp_pos, p=2, dim=-1) < 0.10) # couple with 1
        # 20 cm open
        sub_task_3 = (self._cabinet.data.joint_pos[:, self.drawer_joint_idx] > 0.20) # doesn't need bounding because doesn't always need to touch
        # Fully open (40 cm)
        sub_task_4 = (self._cabinet.data.joint_pos[:, self.drawer_joint_idx] > 0.38) # couple with 3

        # return each environments subtask completion in the form of [N, [0/1, 0/1, 0/1, 0/1]]
        return torch.stack([sub_task_1, sub_task_2, sub_task_3, sub_task_4], dim=1)

    # provides functionality to fetch whole environment poses: [N, 13].
    # Note: We obtain 13 through adding joints for every object. Some situtations will use pos, vel, and quat
    def _get_world(self) -> torch.Tensor:
        return torch.cat([
            self._robot.data.joint_pos, # (9)
            self._cabinet.data.joint_pos, # (4)
        ], dim=1) # (N, 13)
    
    def _update_progression(self):
        completions = self._get_subtasks() # which are completed 
        world = self._get_world() # get the current poses of all envs

        completed_before = self.progression[:,:,0].bool() # [N, 4]
        # which haven't been completed until now
        new_completion = completions & (~completed_before)
        
        # store completion forver this episode
        self.progression[:, :, 0] = torch.maximum(
            self.progression[:, :, 0],
            completions.float(),
        )

        world_expanded = world[:,None,:].expand(-1,4,-1)
        # insert the worlds to where there was a new completion
        self.progression[:,:,1:] = torch.where(
            new_completion[:,:,None],
            world_expanded,
            self.progression[:,:,1:]
        )
        
        # add successful worlds to buffer (sliding)
        completed_envs, completed_tasks = torch.where(new_completion)

        if len(completed_envs) > 0:
            completed_worlds = world[completed_envs]

            for task in range(4):
                task_mask = completed_tasks == task

                if task_mask.any():
                    worlds = completed_worlds[task_mask]

                    start = self.pose_buffer_idx[task]
                    count = worlds.shape[0]

                    indices = (
                        torch.arange(count, device=self.device)
                        + start
                    ) % self.cfg.success_buffer_size

                    self.success_buffer[task, indices] = worlds

                    self.pose_buffer_idx[task] = (
                        start + count
                    ) % self.cfg.success_buffer_size

        if hasattr(self, "extras") and "log" in self.extras:
            L = self.extras["log"]
            success = self.progression[:, :, 0]
            eval_success = torch.zeros([4], dtype=torch.float32)
            replay_success = torch.zeros([4], dtype=torch.float32)

            highest = success.sum(dim=1)

            curriculum_mask = self.is_curriculum_episode
            eval_mask = ~curriculum_mask

            # overall progression
            for i in range(4):
                L[f"subtasks/success_{i+1}"] = success[:, i].mean().item()
            L["env_compare/highest_subtask"] = highest.mean().item()

            # eval for computing difficulty
            if eval_mask.any():
                eval_success = success[eval_mask].mean(dim=0)
                L["env_compare/eval_success_mean"] = eval_success.mean().item()

                for i in range(4):
                    L[f"env_compare/eval_success_{i+1}"] = (eval_success[i].item())

            # replay environment overall
            if curriculum_mask.any():
                replay_success = success[curriculum_mask].mean(dim=0)
                L["env_compare/replay_success_mean"] = (replay_success.mean().item())

                for i in range(4):
                    L[f"env_compare/replay_success_{i+1}"] = (replay_success[i].item())

            # replay competance per subtask
            for task in range(4):
                mask = (curriculum_mask & (self.curriculum_subtask == task))

                if mask.any():
                    replay_task_success = success[mask, task].mean()
                    L[f"env_compare/replay_task_success_{task+1}"] = (replay_task_success.item())

            if curriculum_mask.any() and eval_mask.any():
                gap = eval_success - replay_success

                L["env_compare/replay_task_success_gap_mean"] = (gap.mean().item())
                for i in range(4):
                    L[f"env_compare/replay_task_success_gap_{i+1}"] = (gap[i].item())
            # sample counting
            L[f"env_compare/replay_count_{task+1}"] = mask.sum().item()

    def _update_distribution(self):
        # mask to remove curriculum episodes from compute
        batch_success = (self.progression[:, :, 0].float().mean(dim=0))

        # ema on the success rate to filter noise
        alpha = self.cfg.success_rate_alpha
        self.success_rate = ((1.0 - alpha) * self.success_rate + alpha * batch_success)

        # difficulty
        difficulty = 1.0 - self.success_rate # turns success rate (sr) into failure rate (fr)
        # subtract the previous index from itself: [a, b, c, d] - [0, a, b, c]
        previous = torch.cat([torch.zeros(1, device=self.device), difficulty[:-1]])
        gaps = difficulty - previous # THIS is the distribution
        # alternative to clamping because we want to avoid losing info 
        gaps = gaps - gaps.min() # ensure all values are postitive
        gaps = gaps + 1e-8 # if all gaps are equal we don't want all 0s so add tiny value
        
        # softmax distribution 
        soft = gaps.pow(
            self.cfg.prob_exp
        ).softmax(dim=0)

        # confidence
        confidence = (
            gaps / gaps.sum().clamp(min=1e-8) # linear norm
        )

        top2 = torch.topk(confidence, k=2)
        winner = top2.indices[0] # biggest fr
        margin = (top2.values[0] - top2.values[1]) # difference between biggest and second biggest fr

        # greedy
        hard = torch.zeros_like(gaps)
        hard[winner] = 1.0

        # blend between the two
        blend = torch.clamp(margin / self.cfg.greedy_margin, 0.0, 1.0) # elegant: if margin is great than 0.1 then it will be clamped to 1.0. 
        self.distribution = ((1.0 - blend) * soft + blend * hard)
        self.distribution /= self.distribution.sum()

        # for logging
        if hasattr(self, "extras") and "log" in self.extras:
            L = self.extras["log"]
            L["curriculum/blend"] = blend.item()
            L["curriculum/margin"] = margin.item()
            L["curriculum/selected"] = winner.item()
            for i in range(4):
                L[f"curriculum/success_rate_{i+1}"] = (self.success_rate[i].item())
                L[f"curriculum/difficulty_{i+1}"] = (difficulty[i].item())
                L[f"curriculum/distribution_{i+1}"] = (self.distribution[i].item())
            for i in range(4):
                L[f"curriculum/gap_{i+1}"] = gaps[i].item()
    
    def _run_curriculum_controller(self):
        # only run while curriculum is enabled
        if not self.curriculum_enabled:
            return

        # same natural-only exclusion as the dones/success_rate_margin log: replayed curriculum episodes
        # would otherwise bias the controller's enable/disable decision.
        natural_mask = ~self.is_curriculum_episode
        success_tensor = self.overall_success[natural_mask] if natural_mask.any() else self.overall_success
        success = success_tensor.mean().item() # TODO: switch to ema smoothed success_rate var

        # Take snapshot once
        if (self.progress >= self.cfg.window_analysis_start and self.controller_snapshot is None):
            self.controller_snapshot = success

        # Evaluate once
        if (
            self.progress >= self.cfg.window_analysis_start + self.cfg.window_analysis_size
            and not self.controller_checked
        ):
            self.controller_checked = True
            self.controller_snapshot_two = success
            delta_success = success - self.controller_snapshot
            slope = (delta_success / self.cfg.window_analysis_size)
            self.curriculum_enabled = (slope > self.cfg.slope_threshold)

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        # update progress
        self.progress = min(
            self.common_step_counter /
            (self.cfg.curriculum_total_iterations * 16),
            1.0,
        )
        # run controller for choosing enable/disable
        if self.cfg.reset_state_curriculum_enabled and self.cfg.controller_enabled:
            self._run_curriculum_controller()

        # # custom curriclum work
        self._update_progression() # update data each step
        # uses the updated progressions
        if self.cfg.reset_state_curriculum_enabled:
            if torch.rand((), device=self.device) < 0.10:
                self._update_distribution()
        else: # keep determinisitc by not messing with the rand generator
            if self.common_step_counter % 10 == 0:
                self._update_distribution()

        robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        rewards =  self._compute_rewards(
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
    
        if hasattr(self, "extras") and "log" in self.extras:
            L = self.extras["log"]
            L["reward/total"] = rewards.mean().item()
            L["controller/curriculum_enabled"] = int(self.curriculum_enabled)
            L["controller/first_snapshot_taken"] = 0 if self.controller_snapshot is None else self.controller_snapshot
            L["controller/second_snapshot_taken"] = 0 if self.controller_snapshot_two is None else self.controller_snapshot_two
            if self.controller_checked:
                L["controller/slope"] = ((self.controller_snapshot_two - self.controller_snapshot) / self.cfg.window_analysis_size)

        return rewards

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # robot state
        robot_joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )

        # cabinet state
        cabinet = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)

        # apply curriculum
        if (self.cfg.reset_state_curriculum_enabled and self.curriculum_enabled):
        #if (self.cfg.reset_state_curriculum_enabled and self.curriculum_enabled):
            # force X% of envrionments to be non curriculum (evals instead)
            sample_ratio = self.cfg.sampling_ratio

            # if self.progress > 0.9:
            #     sample_ratio = min(sample_ratio + 0.35, 1.0)
            # elif self.progress > 0.5:
            #     sample_ratio = min(sample_ratio + 0.2, 1.0)
            # elif self.progress < 0.1:
            #     sample_ratio = 0.0
            
            num_curriculum = int(len(env_ids) * sample_ratio)

            perm = torch.randperm(len(env_ids), device=self.device)

            picked = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
            picked[perm[:num_curriculum]] = True

            # update what episodes are actively using the curriculum
            self.is_curriculum_episode[env_ids] = False
            self.is_curriculum_episode[env_ids[picked]] = True

            self.curriculum_subtask[env_ids] = -1

            if picked.any():
                # sample subtasks
                subtasks = torch.multinomial(
                    self.distribution,
                    int(picked.sum().item()), # change value to 0 for reset always to subtask 1, value to 1 for reset always to subtask 2, etc
                    replacement=True,
                )

                self.curriculum_subtask[env_ids[picked]] = subtasks

                # sample stored worlds
                world_ids = torch.randint(
                    0,
                    self.cfg.success_buffer_size,
                    (int(picked.sum().item()),),
                    device=self.device,
                )

                worlds = self.success_buffer[subtasks, world_ids]

                # overwrite default reset with curriculum reset
                robot_joint_pos[picked] = worlds[:, 0:9]
                cabinet[picked] = worlds[:, 9:13]

                # optional domain randomization
                robot_joint_pos[picked] += sample_uniform(
                    -self.cfg.curriculum_dr,
                    self.cfg.curriculum_dr,
                    robot_joint_pos[picked].shape,
                    self.device,
                )

        # reset progression buffer of all environments reset
        self.progression[env_ids] = 0 # set back to incomplete
        
        # robot reset
        robot_joint_pos = torch.clamp(robot_joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(robot_joint_pos)
        self._robot.set_joint_position_target(robot_joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(robot_joint_pos, joint_vel, env_ids=env_ids)

        # cabinet reset
        self._cabinet.write_joint_state_to_sim(cabinet, cabinet, env_ids=env_ids)

        # refresh observations
        self._compute_intermediate_values(env_ids)

        if hasattr(self, "extras") and "log" in self.extras:
            variance = (robot_joint_pos - self._robot.data.default_joint_pos[env_ids]).pow(2).mean() # mean squared difference
            distance = torch.norm(robot_joint_pos - self._robot.data.default_joint_pos[env_ids], dim=1) # distance from the actual joint positions
            
            # logging for variance on reset world
            L = self.extras["log"]
            L["curriculum/reset_distance"] = distance.mean().item()
            L["curriculum/reset_variance"] = variance.item()
            L["curriculum/natural"] = self.is_curriculum_episode.float().mean()
            if self.curriculum_enabled:
                L["curriculum/sample_rate"] = picked.float().mean().item() # make sure we are sampling correct ratio
                L["curriculum/sample_ratio_target"] = sample_ratio

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

        if hasattr(self, "extras") and "log" in self.extras:
            L = self.extras["log"]
            L["reward/dist_reward"] = (dist_reward_scale * dist_reward).mean().item()
            L["reward/rot_reward"] = (rot_reward_scale * rot_reward).mean().item()
            L["reward/open_reward"] = (open_reward_scale * open_reward).mean().item()
            L["reward/action_penalty"] = (-action_penalty_scale * action_penalty).mean().item()
            L["reward/left_finger_distance_reward"] = (finger_reward_scale * lfinger_dist).mean().item()
            L["reward/right_finger_distance_reward"] = (finger_reward_scale * rfinger_dist).mean().item()
            L["reward/finger_dist_penalty"] = (finger_reward_scale * finger_dist_penalty).mean().item()

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
