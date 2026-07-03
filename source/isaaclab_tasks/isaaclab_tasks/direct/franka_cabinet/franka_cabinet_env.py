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
    reset_state_curriculum_enabled = False

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
    prob_exp = 2 # how much we sharpen the probability disturbtion (1 = No sharpening)
    sampling_ratio = 0.3 # what fraction of resets go to the sample distrubtion
    curriculum_dr = 0.02 # how much domain randomization to apply to robot joints

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

        # progression: completion, times, and poses
        self.progression = torch.zeros([self.num_envs, 5, 14], device=self.device) # 5 for the num_subtasks, 13 for (compelted time, poses)
        self.progression[:, :, 0] = self.max_episode_length # first value of world is now max episode length

        # distribution: probabilites for each subtask to sample from
        self.distribution = torch.softmax(torch.ones([5], device=self.device), dim=0) # [0.2, 0.2, 0.2, 0.2, 0.2]

        # success buffer
        self.success_buffer = torch.zeros([5, self.cfg.success_buffer_size, 13], device=self.device) # 5 subtasks, buffer size of 64, and 13 joint attributes to save 
    
        self.pose_buffer_idx = torch.zeros(
            5,
            dtype=torch.long,
            device=self.device
        )

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
        terminated = self._cabinet.data.joint_pos[:, self.drawer_joint_idx] > 0.39
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # returns each environment completion of the subtasks [N, 5]
    def _get_subtasks(self) -> torch.Tensor:
        # 20 cm
        sub_task_1 = torch.norm(self.robot_grasp_pos - self.drawer_grasp_pos, p=2, dim=-1) < 0.20
        # 10 cm
        sub_task_2 = (torch.norm(self.robot_grasp_pos - self.drawer_grasp_pos, p=2, dim=-1) < 0.10) & sub_task_1 # couple with 1
        # 2cm (Touch)
        sub_task_3 = (torch.norm(self.robot_grasp_pos - self.drawer_grasp_pos, p=2, dim=-1) < 0.02) & sub_task_2 # couple with 1 & 2
        # 20 cm open
        sub_task_4 = (self._cabinet.data.joint_pos[:, self.drawer_joint_idx] > 0.20) # doesn't need bounding because doesn't always need to touch
        # Fully open (40 cm)
        sub_task_5 = (self._cabinet.data.joint_pos[:, self.drawer_joint_idx] > 0.39) & sub_task_4 # couple with 4

        # return each environments subtask completion in the form of [N, [0/1, 0/1, 0/1, 0/1, 0/1]]
        return torch.stack([sub_task_1, sub_task_2, sub_task_3, sub_task_4, sub_task_5], dim=1)

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

        previous_times = self.progression[:,:,0] # [N, 5]

        # which haven't been completed until now
        new_completion = (
            completions &
            (previous_times == self.max_episode_length)
        )
        # grab the timestamp of each env
        current_time = self.episode_length_buf.float()

        # insert the timing where their were new completions only
        self.progression[:,:,0] = torch.where(
            new_completion,
            current_time[:,None],
            previous_times
        )

        world_expanded = world[:,None,:].expand(-1,5,-1)
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

            for task in range(5):
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

        # for logging
        if hasattr(self, "extras") and "log" in self.extras:
            times = self.progression[:, :, 0]
            completed = times < self.max_episode_length
            highest = completed.sum(dim=1)

            L = self.extras["log"]
            L["subtasks/sr_subtask_one"] = completions[:, 0].float().mean().item() # success rate
            L["subtasks/sr_subtask_two"] = completions[:, 1].float().mean().item() # success rate
            L["subtasks/sr_subtask_three"] = completions[:, 2].float().mean().item() # success rate
            L["subtasks/sr_subtask_four"] = completions[:, 3].float().mean().item() # success rate
            L["subtasks/sr_subtask_five"] = completions[:, 4].float().mean().item() # success rate
            L["curriculum/highest_subtask"] = highest.float().mean().item() # the task with the highest completion rate
            # the time completion average for each subtask
            for i in range(5):
                if completed[:, i].any():
                    L[f"subtasks/time_subtask_{i+1}"] = (
                        times[completed[:, i], i].mean() /
                        self.max_episode_length
                    ).item()


    def _update_distribution(self):
        times = self.progression[:, :, 0] # [N, 5]
        # average the times for each subtask across all environments
        times_avg = times.mean(dim=0) # [5, 1]
        # divide by episode length to make uncompleted = 1
        times_norm = times_avg / self.max_episode_length # [5, 1]

        # subtract the previous index from itself: [a, b, c, d, e] - [0, a, b, c, d]
        previous = torch.cat([torch.zeros(1, device=self.device),times_norm[:-1]])
        gaps = times_norm - previous # THIS is the distrubtion
        gaps = torch.clamp(gaps, min=0) # make sure its +

        # sharpen values
        gaps = gaps.pow(self.cfg.prob_exp) # Hyperparameter prob_exp is the exponential scaler
        self.distribution = gaps.softmax(dim=0) # update

        # for logging
        if hasattr(self, "extras") and "log" in self.extras:
            L = self.extras["log"]
            for i in range(5):
                L[f"subtasks/distribution_subtask_{i+1}"] = self.distribution[i].item()

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        # custom curriclum work
        self._update_progression() # update data each step
        # uses the updated progressions
        if self.common_step_counter % 10 == 0: # save compute
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
        if self.cfg.reset_state_curriculum_enabled:
            picked = torch.rand(len(env_ids), device=self.device) < self.cfg.sampling_ratio

            if picked.any():
                # sample subtasks
                subtasks = torch.multinomial(
                    self.distribution,
                    int(picked.sum().item()),
                    replacement=True,
                )

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
        self.progression[env_ids] = 0
        self.progression[env_ids, :, 0] = self.max_episode_length

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
            if self.cfg.reset_state_curriculum_enabled:
                L["curriculum/sample_rate"] = picked.float().mean().item() # make sure we are sampling correct ratio

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
