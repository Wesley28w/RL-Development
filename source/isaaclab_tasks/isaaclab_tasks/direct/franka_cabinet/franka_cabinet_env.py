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

from isaaclab_tasks.utils.reset_buffer import available_reset_targets, sample_reset_worlds
from isaaclab_tasks.utils.tscl_window import TSCLWindowTeacher


@configclass
class FrankaCabinetEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 9
    observation_space = 23
    state_space = 0

    # reset state curriculum
    reset_state_curriculum_enabled = False # True
    # ``aces`` preserves the existing empirical-difficulty scheduler. The other modes change only the
    # reset-category probabilities while reusing the same state banks and restoration path.
    reset_curriculum_mode = "aces"
    # Milestone definitions used only by the empirical reset curriculum. The control keeps all four
    # milestones; the simplified-decomposition ablation retains only handle reach and half-open.
    curriculum_subtask_decomposition = "four_stage"

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
    curriculum_store_velocities = False  # control stores joint positions only and resets velocities to zero
    sampling_ratio = 0.3 # what fraction of resets go to the sample distribution
    curriculum_delay_fraction = 0.0  # fraction of training completed before curriculum resets may begin
    success_rate_alpha = 0.05 # momentum control of success rate movement (pre-calculations)
    # softmax temperature on the simplex-normalised confidence. lower = sharper soft branch. replaces the
    # old `prob_exp`, which flattened the distribution instead of sharpening it (see _update_distribution)
    softmax_temperature = 0.25
    # blend band, in units of the uninformative margin 1/(num_subtasks-1): below `lo` the scheduler runs
    # fully soft, above `hi` fully greedy, in between it interpolates. replaces the old `greedy_margin`,
    # which saturated at 1.0 on ~85% of updates
    greedy_margin_lo = 0.5
    greedy_margin_hi = 1.5
    min_subtask_prob = 0.05 # floor on each subtask's sampling probability so none can be starved

    # Shared TSCL-style Window-teacher parameters; 15 iterations gives 100 update opportunities.
    tscl_history_size = 10
    tscl_min_history = 5
    tscl_alpha = 0.1
    tscl_temperature = 0.0004
    tscl_min_samples = 8
    tscl_update_interval_iterations = 15
    tscl_exploration_fraction = 0.20
    # Must match the unchanged RSL-RL num_steps_per_env for this direct Cabinet benchmark.
    tscl_rollout_steps_per_iteration = 16
    
    # policy params
    curriculum_total_iterations = 1500


@configclass
class FrankaCabinetUniformRandomEnvCfg(FrankaCabinetEnvCfg):
    """Cabinet uniform-reset ablation with the environment and PPO settings held fixed."""

    reset_state_curriculum_enabled = True
    reset_curriculum_mode = "uniform_random"


@configclass
class FrankaCabinetAbsoluteDifficultyEnvCfg(FrankaCabinetEnvCfg):
    """Cabinet absolute-difficulty ablation with the environment and PPO settings held fixed."""

    reset_state_curriculum_enabled = True
    reset_curriculum_mode = "absolute_difficulty"


@configclass
class FrankaCabinetFixed5050BlendEnvCfg(FrankaCabinetEnvCfg):
    """Cabinet fixed 50/50 soft-greedy blend with all other settings held fixed."""

    reset_state_curriculum_enabled = True
    reset_curriculum_mode = "fixed_50_50_blend"


@configclass
class FrankaCabinetArgmaxOnlyEnvCfg(FrankaCabinetEnvCfg):
    """Cabinet ACES argmax-only ablation with all other settings held fixed."""

    reset_state_curriculum_enabled = True
    reset_curriculum_mode = "argmax_only"


@configclass
class FrankaCabinetSoftmaxOnlyEnvCfg(FrankaCabinetEnvCfg):
    """Cabinet ACES softmax-only ablation with all other settings held fixed."""

    reset_state_curriculum_enabled = True
    reset_curriculum_mode = "softmax_only"


@configclass
class FrankaCabinetNoEMAEnvCfg(FrankaCabinetEnvCfg):
    """Cabinet ACES no-EMA ablation with all other settings held fixed."""

    reset_state_curriculum_enabled = True
    reset_curriculum_mode = "no_ema"
    # Alpha=1 makes the existing EMA assignment exactly equal to the current measurement.
    success_rate_alpha = 1.0


@configclass
class FrankaCabinetSamplingRatio05EnvCfg(FrankaCabinetEnvCfg):
    """Cabinet ACES 0.50 curriculum-reset-ratio ablation with all other settings held fixed."""

    reset_state_curriculum_enabled = True
    sampling_ratio = 0.5


@configclass
class FrankaCabinetDelay15EnvCfg(FrankaCabinetEnvCfg):
    """Cabinet ACES 15%-delay ablation with all other settings held fixed."""

    reset_state_curriculum_enabled = True
    curriculum_delay_fraction = 0.15


@configclass
class FrankaCabinetBufferSize1EnvCfg(FrankaCabinetEnvCfg):
    """Cabinet ACES one-state-per-subtask-buffer ablation with all other settings held fixed."""

    reset_state_curriculum_enabled = True
    success_buffer_size = 1


@configclass
class FrankaCabinetVelocityWorldEnvCfg(FrankaCabinetEnvCfg):
    """Cabinet ACES position-and-velocity empirical-state ablation with all other settings held fixed."""

    reset_state_curriculum_enabled = True
    curriculum_store_velocities = True


@configclass
class FrankaCabinetSimplifiedSubtasksEnvCfg(FrankaCabinetEnvCfg):
    """Cabinet ACES two-milestone decomposition ablation with all other settings held fixed."""

    reset_state_curriculum_enabled = True
    curriculum_subtask_decomposition = "two_stage"


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

        subtask_counts = {"four_stage": 4, "two_stage": 2}
        if self.cfg.curriculum_subtask_decomposition not in subtask_counts:
            raise ValueError(
                f"Unknown curriculum_subtask_decomposition={self.cfg.curriculum_subtask_decomposition!r}; "
                "expected 'four_stage' or 'two_stage'."
            )
        self.num_curriculum_subtasks = subtask_counts[self.cfg.curriculum_subtask_decomposition]

        # The control stores 13 joint positions. The velocity-world ablation appends the corresponding
        # 13 joint velocities without changing milestone detection, buffer collection, or ACES sampling.
        self.curriculum_world_dim = self._robot.num_joints + self._cabinet.num_joints
        if self.cfg.curriculum_store_velocities:
            self.curriculum_world_dim *= 2

        # progression: completion flag followed by the captured empirical world
        self.progression = torch.zeros(
            [self.num_envs, self.num_curriculum_subtasks, 1 + self.curriculum_world_dim], device=self.device
        )
        self.success_rate = torch.zeros(self.num_curriculum_subtasks, device=self.device)

        # distribution: probabilites for each subtask to sample from
        self.distribution = torch.softmax(
            torch.ones([self.num_curriculum_subtasks], device=self.device), dim=0
        )

        # success buffer
        self.success_buffer = torch.zeros(
            [self.num_curriculum_subtasks, self.cfg.success_buffer_size, self.curriculum_world_dim],
            device=self.device,
        )
    
        self.pose_buffer_idx = torch.zeros(
            self.num_curriculum_subtasks,
            dtype=torch.long,
            device=self.device
        )

        # how many slots of each subtask's buffer have actually been written. sampling an unwritten slot
        # replays all-zero joint positions (~3.4 rad from the home pose for this franka), which is a state no
        # rollout ever visits. the difficulty rule concentrates on whichever subtask the policy reaches least
        # -- exactly the one most likely to still be empty -- so this gates both which subtasks can be
        # sampled and which slots within them. same as factory_env's pose_buffer_count.
        self.pose_buffer_count = torch.zeros(
            self.num_curriculum_subtasks,
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

        # The empirical reset category is the TSCL task. The teacher changes only the distribution over
        # these existing banks; it never changes rewards, observations, dynamics, or PPO data.
        self.tscl_teacher: TSCLWindowTeacher | None = None
        if self.cfg.reset_curriculum_mode not in {
            "aces",
            "tscl_window",
            "uniform_random",
            "absolute_difficulty",
            "fixed_50_50_blend",
            "argmax_only",
            "softmax_only",
            "no_ema",
        }:
            raise ValueError(
                f"Unknown reset_curriculum_mode={self.cfg.reset_curriculum_mode!r}; "
                "expected 'aces', 'tscl_window', 'uniform_random', 'absolute_difficulty', "
                "'fixed_50_50_blend', 'argmax_only', 'softmax_only', or 'no_ema'."
            )
        if self.cfg.reset_state_curriculum_enabled and self.cfg.reset_curriculum_mode == "tscl_window":
            self.tscl_teacher = TSCLWindowTeacher(
                self.num_curriculum_subtasks,
                self.device,
                history_size=self.cfg.tscl_history_size,
                min_history=self.cfg.tscl_min_history,
                alpha=self.cfg.tscl_alpha,
                temperature=self.cfg.tscl_temperature,
                min_samples=self.cfg.tscl_min_samples,
                update_interval_iterations=self.cfg.tscl_update_interval_iterations,
                exploration_fraction=self.cfg.tscl_exploration_fraction,
            )
        
        # Controller
        self.progress = 0.0 # for tracking progression (0.0-1.0)
        self.curriculum_enabled = self._curriculum_reset_is_active(
            self.cfg.reset_state_curriculum_enabled,
            self.progress,
            self.cfg.curriculum_delay_fraction,
        )
        self.overall_success = 0.0 # final task success (0.0-1.0)

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

    # returns each environment completion of the configured curriculum milestones
    def _get_subtasks(self) -> torch.Tensor:
        grasp_distance = torch.norm(self.robot_grasp_pos - self.drawer_grasp_pos, p=2, dim=-1)
        reach_20_cm = grasp_distance < 0.20
        reach_handle = grasp_distance < 0.10
        drawer_half_open = self._cabinet.data.joint_pos[:, self.drawer_joint_idx] > 0.20
        drawer_fully_open = self._cabinet.data.joint_pos[:, self.drawer_joint_idx] > 0.38

        if self.cfg.curriculum_subtask_decomposition == "two_stage":
            return torch.stack([reach_handle, drawer_half_open], dim=1)
        return torch.stack([reach_20_cm, reach_handle, drawer_half_open, drawer_fully_open], dim=1)

    # provides functionality to fetch whole environment poses: [N, 13].
    # Note: We obtain 13 through adding joints for every object. Some situtations will use pos, vel, and quat
    def _get_world(self) -> torch.Tensor:
        components = [
            self._robot.data.joint_pos,
            self._cabinet.data.joint_pos,
        ]
        if self.cfg.curriculum_store_velocities:
            components.extend(
                [
                    self._robot.data.joint_vel,
                    self._cabinet.data.joint_vel,
                ]
            )
        return torch.cat(components, dim=1)
    
    def _update_progression(self):
        completions = self._get_subtasks() # which are completed 
        world = self._get_world() # get the current poses of all envs

        completed_before = self.progression[:,:,0].bool()
        # which haven't been completed until now
        new_completion = completions & (~completed_before)
        
        # store completion forver this episode
        self.progression[:, :, 0] = torch.maximum(
            self.progression[:, :, 0],
            completions.float(),
        )

        world_expanded = world[:,None,:].expand(-1, self.num_curriculum_subtasks, -1)
        # insert the worlds to where there was a new completion
        self.progression[:,:,1:] = torch.where(
            new_completion[:,:,None],
            world_expanded,
            self.progression[:,:,1:]
        )
        
        # add successful worlds to buffer (sliding).
        # ONLY natural episodes may contribute. self.progression is zeroed for every env at reset, so a
        # curriculum env satisfies the rung it was teleported into on the very first evaluation after the
        # teleport -- without this mask it re-deposits its own start state
        # one physics step after drawing it, and the buffer becomes a self-replicating population of its
        # own output. A rung-4 teleport also satisfies rung 3 (drawer > 0.20) and usually rungs 1-2 (the
        # gripper is still on the handle), so one teleport would seed up to four buffers with the same
        # state. Measured consequence in the runs this was found in: seeds whose natural success at a rung
        # had reached exactly 0.000 still reported that rung's buffer as completely full.
        buffer_completion = new_completion & (~self.is_curriculum_episode).unsqueeze(1)
        completed_envs, completed_tasks = torch.where(buffer_completion)

        if len(completed_envs) > 0:
            completed_worlds = world[completed_envs]

            for task in range(self.num_curriculum_subtasks):
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

                    self.pose_buffer_count[task] = torch.clamp(
                        self.pose_buffer_count[task] + count,
                        max=self.cfg.success_buffer_size,
                    )

        if self.tscl_teacher is not None:
            ppo_iteration = self.common_step_counter // self.cfg.tscl_rollout_steps_per_iteration
            global_env_steps = self.common_step_counter * self.num_envs
            available_tasks = self.pose_buffer_count > 0
            self.tscl_teacher.update_if_due(ppo_iteration, global_env_steps, available_tasks)
            self.distribution = self.tscl_teacher.get_distribution(available_tasks)

        if hasattr(self, "extras") and "log" in self.extras:
            L = self.extras["log"]
            success = self.progression[:, :, 0]
            eval_success = torch.zeros([self.num_curriculum_subtasks], dtype=torch.float32)
            replay_success = torch.zeros([self.num_curriculum_subtasks], dtype=torch.float32)

            highest = success.sum(dim=1)

            curriculum_mask = self.is_curriculum_episode
            eval_mask = ~curriculum_mask

            # overall progression
            for i in range(self.num_curriculum_subtasks):
                L[f"subtasks/success_{i+1}"] = success[:, i].mean().item()
            L["env_compare/highest_subtask"] = highest.mean().item()

            # eval for computing difficulty
            if eval_mask.any():
                eval_success = success[eval_mask].mean(dim=0)
                L["env_compare/eval_success_mean"] = eval_success.mean().item()

                for i in range(self.num_curriculum_subtasks):
                    L[f"env_compare/eval_success_{i+1}"] = (eval_success[i].item())

            # replay environment overall
            if curriculum_mask.any():
                replay_success = success[curriculum_mask].mean(dim=0)
                L["env_compare/replay_success_mean"] = (replay_success.mean().item())

                for i in range(self.num_curriculum_subtasks):
                    L[f"env_compare/replay_success_{i+1}"] = (replay_success[i].item())

            # replay competance per subtask
            for task in range(self.num_curriculum_subtasks):
                mask = (curriculum_mask & (self.curriculum_subtask == task))
                L[f"env_compare/replay_count_{task+1}"] = mask.sum().item()

                if mask.any():
                    replay_task_success = success[mask, task].mean()
                    L[f"env_compare/replay_task_success_{task+1}"] = (replay_task_success.item())

            if curriculum_mask.any() and eval_mask.any():
                gap = eval_success - replay_success

                L["env_compare/replay_task_success_gap_mean"] = (gap.mean().item())
                for i in range(self.num_curriculum_subtasks):
                    L[f"env_compare/replay_task_success_gap_{i+1}"] = (gap[i].item())
            if self.tscl_teacher is not None:
                ppo_iteration = self.common_step_counter // self.cfg.tscl_rollout_steps_per_iteration
                global_env_steps = self.common_step_counter * self.num_envs
                available_tasks = self.pose_buffer_count > 0
                L.update(self.tscl_teacher.metrics(ppo_iteration, global_env_steps, available_tasks))
                L["tscl/rollout_steps_per_iteration"] = self.cfg.tscl_rollout_steps_per_iteration
                L["tscl/standard_reset_fraction"] = 1.0 - self.cfg.sampling_ratio
                L["tscl/curriculum_reset_fraction"] = self.cfg.sampling_ratio

    def _update_distribution(self):
        # mask to remove curriculum episodes from compute
        mask = ~self.is_curriculum_episode
        # edge case where every env is curriculum
        if mask.sum() == 0:
            return
        
        batch_success = (self.progression[mask, :, 0].float().mean(dim=0))

        # EMA on the success rate to filter noise. The no-EMA ablation sets alpha=1.0, making this
        # assignment exactly equal to the current natural-episode measurement without another code path.
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
        
        num_subtasks = self.success_rate.numel()

        # confidence: put the gaps on the simplex *before* shaping them
        confidence = (
            gaps / gaps.sum().clamp(min=1e-8) # linear norm
        )

        # soft branch: a real temperature on the normalised confidence.
        # the old form was gaps.pow(prob_exp).softmax(). gaps live in [0, 1], so raising them to a power
        # pushes them toward zero, and a softmax over values spanning ~0.04 is flat to within 4%. measured
        # on the paper runs the soft branch sat within 0.05 of uniform for entire runs -- it was not a soft
        # preference, it was uniform sampling, which left the greedy branch as the only thing that ever
        # shaped the distribution.
        soft = (confidence / self.cfg.softmax_temperature).softmax(dim=0)

        top2 = torch.topk(confidence, k=2)
        winner = top2.indices[0] # biggest fr
        margin = (top2.values[0] - top2.values[1]) # difference between biggest and second biggest fr

        # `gaps - gaps.min()` forces one entry to zero, so confidence is a distribution over at most
        # num_subtasks-1 non-zero entries. under an uninformative difficulty signal (flat Dirichlet over
        # those entries) E[margin] is exactly 1/(num_subtasks-1) -- verified numerically at K=3,4,5.
        # dividing by that reference makes the margin mean the same thing regardless of how many subtasks
        # the task has: ~1 is "no more informative than noise", >1 is "one subtask genuinely stands out".
        margin_ref = 1.0 / max(num_subtasks - 1, 1)
        margin_norm = margin / margin_ref

        # greedy
        hard = torch.zeros_like(gaps)
        hard[winner] = 1.0

        if self.cfg.reset_curriculum_mode == "fixed_50_50_blend":
            # Ablate only ACES's adaptive mixing coefficient. Soft/hard construction, EMA, update timing,
            # and the post-blend starvation floor remain identical to the ACES control.
            blend = confidence.new_tensor(0.5)
        elif self.cfg.reset_curriculum_mode == "argmax_only":
            # Remove the soft branch while retaining ACES's post-blend starvation floor.
            blend = confidence.new_tensor(1.0)
        elif self.cfg.reset_curriculum_mode == "softmax_only":
            # Remove the hard/argmax branch while retaining ACES's post-blend starvation floor.
            blend = confidence.new_tensor(0.0)
        else:
            # traverse soft -> greedy across a band instead of saturating. the old clamp(margin / 0.10) hit its
            # ceiling on ~85% of updates, so blend was pinned at exactly 1 and the controller was bang-bang
            # between one-hot and uniform with nothing in between. the band below leaves ~70% of updates
            # strictly inside (0, 1) on the logged margins from the paper runs.
            blend = torch.clamp(
                (margin_norm - self.cfg.greedy_margin_lo)
                / max(self.cfg.greedy_margin_hi - self.cfg.greedy_margin_lo, 1e-8),
                0.0,
                1.0,
            )
        distribution = ((1.0 - blend) * soft + blend * hard)

        # floor every subtask so none can be starved to exactly zero. a subtask that stops being sampled
        # stops generating buffer entries and stops being re-evaluated, which makes the collapse permanent.
        eps = self.cfg.min_subtask_prob
        distribution = (1.0 - eps * num_subtasks) * distribution + eps
        self.distribution = distribution / distribution.sum()

        # for logging
        if hasattr(self, "extras") and "log" in self.extras:
            L = self.extras["log"]
            ablation_modes = {"fixed_50_50_blend", "argmax_only", "softmax_only", "no_ema"}
            log_prefix = (
                self.cfg.reset_curriculum_mode if self.cfg.reset_curriculum_mode in ablation_modes else "curriculum"
            )
            L[f"{log_prefix}/blend"] = blend.item()
            L[f"{log_prefix}/margin"] = margin.item()
            L[f"{log_prefix}/margin_norm"] = margin_norm.item()
            L[f"{log_prefix}/selected"] = winner.item()
            distribution_metric = (
                "raw_distribution" if self.cfg.reset_curriculum_mode in ablation_modes else "distribution"
            )
            if self.cfg.reset_curriculum_mode == "no_ema":
                L["no_ema/ema_alpha_effective"] = self.cfg.success_rate_alpha
            for i in range(num_subtasks):
                if self.cfg.reset_curriculum_mode == "no_ema":
                    L[f"no_ema/batch_success_rate_{i+1}"] = batch_success[i].item()
                L[f"{log_prefix}/success_rate_{i+1}"] = (self.success_rate[i].item())
                L[f"{log_prefix}/difficulty_{i+1}"] = (difficulty[i].item())
                L[f"{log_prefix}/{distribution_metric}_{i+1}"] = (self.distribution[i].item())
            for i in range(num_subtasks):
                L[f"{log_prefix}/gap_{i+1}"] = gaps[i].item()

    def _available_curriculum_targets(self) -> torch.Tensor:
        return available_reset_targets(self.pose_buffer_count, use_predecessor=self.tscl_teacher is None)

    @staticmethod
    def _uniform_available_distribution(available_subtasks: torch.Tensor) -> torch.Tensor:
        """Return equal probability for each available reset target."""
        distribution = available_subtasks.to(dtype=torch.float32)
        return distribution / distribution.sum().clamp(min=1.0)

    @staticmethod
    def _absolute_difficulty_distribution(
        success_rates: torch.Tensor, available_subtasks: torch.Tensor
    ) -> torch.Tensor:
        """Normalize failure rates over available targets, without nonlinear shaping."""
        difficulty = (1.0 - success_rates).clamp(0.0, 1.0)
        weighted_difficulty = torch.where(available_subtasks, difficulty, torch.zeros_like(difficulty))
        total_difficulty = weighted_difficulty.sum()
        normalized_difficulty = weighted_difficulty / total_difficulty.clamp(min=1e-8)
        uniform_fallback = FrankaCabinetEnv._uniform_available_distribution(available_subtasks)
        return torch.where(total_difficulty > 0.0, normalized_difficulty, uniform_fallback)

    def _update_absolute_difficulty_distribution(self):
        """Update the EMA success measurement, then sample in direct proportion to absolute difficulty."""
        natural_mask = ~self.is_curriculum_episode
        if natural_mask.sum() == 0:
            return

        batch_success = self.progression[natural_mask, :, 0].float().mean(dim=0)
        alpha = self.cfg.success_rate_alpha
        self.success_rate = (1.0 - alpha) * self.success_rate + alpha * batch_success

        available_subtasks = self._available_curriculum_targets()
        self.distribution = self._absolute_difficulty_distribution(self.success_rate, available_subtasks)

        if hasattr(self, "extras") and "log" in self.extras:
            L = self.extras["log"]
            difficulty = 1.0 - self.success_rate
            for i in range(self.success_rate.numel()):
                L[f"absolute_difficulty/batch_success_rate_{i+1}"] = batch_success[i].item()
                L[f"absolute_difficulty/success_rate_{i+1}"] = self.success_rate[i].item()
                L[f"absolute_difficulty/difficulty_{i+1}"] = difficulty[i].item()
                L[f"absolute_difficulty/distribution_{i+1}"] = self.distribution[i].item()

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        # update progress
        self.progress = min(
            self.common_step_counter /
            (self.cfg.curriculum_total_iterations * 16),
            1.0,
        )
        # Delay gates only reset-state replay. Progress tracking, empirical buffer collection, ACES success
        # estimation, distribution updates, logging, rewards, and PPO training all remain active from step 0.
        self.curriculum_enabled = self._curriculum_reset_is_active(
            self.cfg.reset_state_curriculum_enabled,
            self.progress,
            self.cfg.curriculum_delay_fraction,
        )
        # # custom curriclum work
        self._update_progression() # update data each step
        # uses the updated progressions
        if self.cfg.reset_state_curriculum_enabled and self.cfg.reset_curriculum_mode == "aces":
            if torch.rand((), device=self.device) < 0.10:
                self._update_distribution()
        elif self.cfg.reset_state_curriculum_enabled and self.cfg.reset_curriculum_mode == "tscl_window":
            if self.tscl_teacher is None:
                raise RuntimeError("TSCL mode is active but its Window teacher was not initialized.")
            self.distribution = self.tscl_teacher.get_distribution(self.pose_buffer_count > 0)
        elif self.cfg.reset_state_curriculum_enabled and self.cfg.reset_curriculum_mode == "uniform_random":
            self.distribution = self._uniform_available_distribution(self._available_curriculum_targets())
        elif self.cfg.reset_state_curriculum_enabled and self.cfg.reset_curriculum_mode == "absolute_difficulty":
            # Preserve ACES's success-rate EMA and update cadence; replace only its bottleneck scheduler.
            if torch.rand((), device=self.device) < 0.10:
                self._update_absolute_difficulty_distribution()
            # Keep the maintained/logged vector masked when bank availability changes between EMA updates.
            self.distribution = self._absolute_difficulty_distribution(
                self.success_rate, self._available_curriculum_targets()
            )
        elif self.cfg.reset_state_curriculum_enabled and self.cfg.reset_curriculum_mode == "fixed_50_50_blend":
            if torch.rand((), device=self.device) < 0.10:
                self._update_distribution()
        elif self.cfg.reset_state_curriculum_enabled and self.cfg.reset_curriculum_mode == "argmax_only":
            if torch.rand((), device=self.device) < 0.10:
                self._update_distribution()
        elif self.cfg.reset_state_curriculum_enabled and self.cfg.reset_curriculum_mode == "softmax_only":
            if torch.rand((), device=self.device) < 0.10:
                self._update_distribution()
        elif self.cfg.reset_state_curriculum_enabled and self.cfg.reset_curriculum_mode == "no_ema":
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

            if self.cfg.curriculum_delay_fraction > 0.0:
                L["delay_15/training_fraction"] = self.progress
                L["delay_15/delay_fraction"] = self.cfg.curriculum_delay_fraction
                L["delay_15/reset_replay_enabled"] = int(self.curriculum_enabled)
                L["delay_15/effective_curriculum_reset_fraction"] = (
                    self.cfg.sampling_ratio if self.curriculum_enabled else 0.0
                )

            if self.cfg.reset_state_curriculum_enabled and self.cfg.reset_curriculum_mode == "uniform_random":
                L["uniform_random/standard_reset_fraction"] = 1.0 - self.cfg.sampling_ratio
                L["uniform_random/curriculum_reset_fraction"] = self.cfg.sampling_ratio
                for i in range(self.num_curriculum_subtasks):
                    L[f"uniform_random/distribution_{i+1}"] = self.distribution[i].item()
                    L[f"uniform_random/buffer_count_{i+1}"] = self.pose_buffer_count[i].item()
            elif self.cfg.reset_state_curriculum_enabled and self.cfg.reset_curriculum_mode == "absolute_difficulty":
                L["absolute_difficulty/standard_reset_fraction"] = 1.0 - self.cfg.sampling_ratio
                L["absolute_difficulty/curriculum_reset_fraction"] = self.cfg.sampling_ratio
                for i in range(self.num_curriculum_subtasks):
                    L[f"absolute_difficulty/distribution_{i+1}"] = self.distribution[i].item()
                    L[f"absolute_difficulty/buffer_count_{i+1}"] = self.pose_buffer_count[i].item()
            elif self.cfg.reset_state_curriculum_enabled and self.cfg.reset_curriculum_mode == "fixed_50_50_blend":
                L["fixed_50_50_blend/standard_reset_fraction"] = 1.0 - self.cfg.sampling_ratio
                L["fixed_50_50_blend/curriculum_reset_fraction"] = self.cfg.sampling_ratio
                L["fixed_50_50_blend/blend"] = 0.5
                valid_subtasks = self._available_curriculum_targets()
                effective_distribution = torch.where(
                    valid_subtasks, self.distribution, torch.zeros_like(self.distribution)
                )
                effective_distribution /= effective_distribution.sum().clamp(min=1e-8)
                for i in range(self.num_curriculum_subtasks):
                    L[f"fixed_50_50_blend/raw_distribution_{i+1}"] = self.distribution[i].item()
                    L[f"fixed_50_50_blend/distribution_{i+1}"] = effective_distribution[i].item()
                    L[f"fixed_50_50_blend/buffer_count_{i+1}"] = self.pose_buffer_count[i].item()
            elif self.cfg.reset_state_curriculum_enabled and self.cfg.reset_curriculum_mode == "argmax_only":
                L["argmax_only/standard_reset_fraction"] = 1.0 - self.cfg.sampling_ratio
                L["argmax_only/curriculum_reset_fraction"] = self.cfg.sampling_ratio
                L["argmax_only/blend"] = 1.0
                valid_subtasks = self._available_curriculum_targets()
                effective_distribution = torch.where(
                    valid_subtasks, self.distribution, torch.zeros_like(self.distribution)
                )
                effective_distribution /= effective_distribution.sum().clamp(min=1e-8)
                for i in range(self.num_curriculum_subtasks):
                    L[f"argmax_only/raw_distribution_{i+1}"] = self.distribution[i].item()
                    L[f"argmax_only/distribution_{i+1}"] = effective_distribution[i].item()
                    L[f"argmax_only/buffer_count_{i+1}"] = self.pose_buffer_count[i].item()
            elif self.cfg.reset_state_curriculum_enabled and self.cfg.reset_curriculum_mode == "softmax_only":
                L["softmax_only/standard_reset_fraction"] = 1.0 - self.cfg.sampling_ratio
                L["softmax_only/curriculum_reset_fraction"] = self.cfg.sampling_ratio
                L["softmax_only/blend"] = 0.0
                valid_subtasks = self._available_curriculum_targets()
                effective_distribution = torch.where(
                    valid_subtasks, self.distribution, torch.zeros_like(self.distribution)
                )
                effective_distribution /= effective_distribution.sum().clamp(min=1e-8)
                for i in range(self.num_curriculum_subtasks):
                    L[f"softmax_only/raw_distribution_{i+1}"] = self.distribution[i].item()
                    L[f"softmax_only/distribution_{i+1}"] = effective_distribution[i].item()
                    L[f"softmax_only/buffer_count_{i+1}"] = self.pose_buffer_count[i].item()
            elif self.cfg.reset_state_curriculum_enabled and self.cfg.reset_curriculum_mode == "no_ema":
                L["no_ema/standard_reset_fraction"] = 1.0 - self.cfg.sampling_ratio
                L["no_ema/curriculum_reset_fraction"] = self.cfg.sampling_ratio
                L["no_ema/ema_alpha_effective"] = self.cfg.success_rate_alpha
                valid_subtasks = self._available_curriculum_targets()
                effective_distribution = torch.where(
                    valid_subtasks, self.distribution, torch.zeros_like(self.distribution)
                )
                effective_distribution /= effective_distribution.sum().clamp(min=1e-8)
                for i in range(self.num_curriculum_subtasks):
                    L[f"no_ema/raw_distribution_{i+1}"] = self.distribution[i].item()
                    L[f"no_ema/distribution_{i+1}"] = effective_distribution[i].item()
                    L[f"no_ema/buffer_count_{i+1}"] = self.pose_buffer_count[i].item()

        return rewards

    @staticmethod
    def _curriculum_reset_is_active(enabled: bool, progress: float, delay_fraction: float) -> bool:
        """Return whether empirical reset-state replay may be applied at the current training fraction."""
        return bool(enabled and progress >= delay_fraction)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Record the just-finished, category-labelled episodes before reset clears their labels. The success
        # test is exactly the environment's existing termination criterion; teacher statistics never enter
        # the reward or PPO rollout buffer.
        if self.tscl_teacher is not None:
            successes = self._cabinet.data.joint_pos[env_ids, self.drawer_joint_idx] > 0.39
            self.tscl_teacher.record_episode_outcomes(self.curriculum_subtask[env_ids], successes)

        super()._reset_idx(env_ids)

        # robot state
        robot_joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        self.is_curriculum_episode[env_ids] = False
        self.curriculum_subtask[env_ids] = -1

        # cabinet state
        cabinet = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        joint_vel = torch.zeros_like(robot_joint_pos)
        cabinet_joint_vel = torch.zeros_like(cabinet)

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

            # ACES targets milestone i from bank i-1; target zero keeps the normal reset.
            # TSCL categories already identify reset banks directly.
            valid_subtasks = self._available_curriculum_targets()
            if not valid_subtasks.any():
                picked[:] = False

            # update what episodes are actively using the curriculum
            self.is_curriculum_episode[env_ids] = False
            self.is_curriculum_episode[env_ids[picked]] = True

            self.curriculum_subtask[env_ids] = -1

            if picked.any():
                # sample subtasks, restricted to ones that actually have recorded poses. if the distribution
                # has committed all of its mass to subtasks with no data, fall back to uniform over the ones
                # that do have data.
                if self.cfg.reset_curriculum_mode == "uniform_random":
                    masked_distribution = self._uniform_available_distribution(valid_subtasks)
                    self.distribution = masked_distribution
                elif self.tscl_teacher is not None:
                    masked_distribution = self.tscl_teacher.get_distribution(valid_subtasks)
                    self.distribution = masked_distribution
                else:
                    masked_distribution = torch.where(
                        valid_subtasks, self.distribution, torch.zeros_like(self.distribution)
                    )
                if masked_distribution.sum() <= 0:
                    masked_distribution = valid_subtasks.float()
                masked_distribution = masked_distribution / masked_distribution.sum()

                subtasks = torch.multinomial(
                    masked_distribution,
                    int(picked.sum().item()), # change value to 0 for reset always to subtask 1, value to 1 for reset always to subtask 2, etc
                    replacement=True,
                )

                self.curriculum_subtask[env_ids[picked]] = subtasks
                if self.tscl_teacher is not None:
                    self.tscl_teacher.record_reset_draws(subtasks)

                replay, worlds = sample_reset_worlds(
                    self.success_buffer, self.pose_buffer_count, subtasks,
                    use_predecessor=self.tscl_teacher is None,
                )
                picked_ids = env_ids[picked]
                self.is_curriculum_episode[picked_ids] = replay
                # A first-milestone target is a natural reset, not a replayed episode.
                picked[picked.clone()] = replay

                # overwrite default reset with curriculum reset
                robot_pos_end = self._robot.num_joints
                cabinet_pos_end = robot_pos_end + self._cabinet.num_joints
                robot_joint_pos[picked] = worlds[:, :robot_pos_end]
                cabinet[picked] = worlds[:, robot_pos_end:cabinet_pos_end]

                if self.cfg.curriculum_store_velocities:
                    robot_vel_end = cabinet_pos_end + self._robot.num_joints
                    cabinet_vel_end = robot_vel_end + self._cabinet.num_joints
                    joint_vel[picked] = worlds[:, cabinet_pos_end:robot_vel_end]
                    cabinet_joint_vel[picked] = worlds[:, robot_vel_end:cabinet_vel_end]

        # Preserve the original baseline joint-reset noise only for natural episodes.
        natural = ~self.is_curriculum_episode[env_ids]
        if natural.any():
            robot_joint_pos[natural] += sample_uniform(
                -0.125, 0.125, (int(natural.sum().item()), self._robot.num_joints), self.device
            )

        # reset progression buffer of all environments reset
        self.progression[env_ids] = 0 # set back to incomplete
        
        # robot reset
        robot_joint_pos = torch.clamp(robot_joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        # re-anchor the action integrator on the pose we are resetting to. actions are deltas accumulated onto
        # robot_dof_targets (see _pre_physics_step) and _apply_action writes that whole buffer every step for
        # all envs, so without this the target below is immediately overwritten by the *previous* episode's
        # final commanded target -- hidden state leaking across the episode boundary. factory_env does the same.
        self.robot_dof_targets[env_ids] = robot_joint_pos
        self._robot.set_joint_position_target(robot_joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(robot_joint_pos, joint_vel, env_ids=env_ids)

        # cabinet reset. `cabinet` holds joint *positions* (zeros normally, a replayed drawer state when the
        # curriculum picked this env), so velocity needs its own zero tensor -- passing `cabinet` twice would
        # launch a replayed drawer outward at drawer_pos m/s.
        self._cabinet.write_joint_state_to_sim(cabinet, cabinet_joint_vel, env_ids=env_ids)

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
            if self.cfg.curriculum_store_velocities:
                L["velocity_world/state_dimension"] = self.curriculum_world_dim
                L["velocity_world/robot_joint_speed_mean"] = joint_vel.norm(dim=1).mean().item()
                L["velocity_world/cabinet_joint_speed_mean"] = cabinet_joint_vel.norm(dim=1).mean().item()
            if self.cfg.reset_state_curriculum_enabled and self.curriculum_enabled:
                L["curriculum/sample_rate"] = picked.float().mean().item() # make sure we are sampling correct ratio
                L["curriculum/sample_ratio_target"] = sample_ratio
                L["curriculum/valid_subtasks"] = valid_subtasks.float().sum().item()
                for i in range(self.num_curriculum_subtasks):
                    L[f"curriculum/buffer_fill_{i+1}"] = self.pose_buffer_count[i].item()

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
