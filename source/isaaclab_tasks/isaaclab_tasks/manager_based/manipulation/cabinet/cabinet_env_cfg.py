# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


##
# Scene definition
##


@configclass
class CabinetSceneCfg(InteractiveSceneCfg):
    """Configuration for the cabinet scene with a robot and a cabinet.

    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames
    """

    # robots, Will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # End-effector, Will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, 0, 0.4),
            rot=(0.0, 0.0, 0.0, 1.0),
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

    # Frame definitions for the cabinet.
    cabinet_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet/sektion",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/CabinetFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_top",
                name="drawer_handle_top",
                offset=OffsetCfg(
                    pos=(0.305, 0.0, 0.01),
                    rot=(0.5, 0.5, -0.5, -0.5),  # align with end-effector frame
                ),
            ),
        ],
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cabinet_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
        )
        cabinet_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
        )
        rel_ee_drawer_distance = ObsTerm(func=mdp.rel_ee_drawer_distance)

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    cabinet_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cabinet", body_names="drawer_handle_top"),
            "static_friction_range": (1.0, 1.25),
            "dynamic_friction_range": (1.25, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Reset-pose curriculum: for a sampled fraction of resets, overwrites the default reset above with a
    # replayed subtask state (see mdp/events.py). Must be registered *last* in this class so its writes are
    # the ones that stick. Gated by `reset_state_curriculum_enabled` (True by default below); setting that
    # to False makes this a complete no-op -- it doesn't even draw from the random generator.
    sample_curriculum_reset_state = EventTerm(
        func=mdp.sample_curriculum_reset_state,
        mode="reset",
        params={"tracker_term_name": "subtask_progression_tracker"},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 1. Approach the handle
    approach_ee_handle = RewTerm(func=mdp.approach_ee_handle, weight=2.0, params={"threshold": 0.2})
    align_ee_handle = RewTerm(func=mdp.align_ee_handle, weight=0.5)

    # 2. Grasp the handle
    approach_gripper_handle = RewTerm(func=mdp.approach_gripper_handle, weight=5.0, params={"offset": MISSING})
    align_grasp_around_handle = RewTerm(func=mdp.align_grasp_around_handle, weight=0.125)
    grasp_handle = RewTerm(
        func=mdp.grasp_handle,
        weight=0.5,
        params={
            "threshold": 0.03,
            "open_joint_pos": MISSING,
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
        },
    )

    # 3. Open the drawer
    open_drawer_bonus = RewTerm(
        func=mdp.open_drawer_bonus,
        weight=7.5,
        params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
    )
    multi_stage_open_drawer = RewTerm(
        func=mdp.multi_stage_open_drawer,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
    )

    # 4. Penalize actions for cosmetic reasons
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)

    # Reset-pose curriculum bookkeeping (see mdp/events.py). Always contributes exactly 0 to the reward,
    # enabled or not -- registered here (rather than as an "interval" event) so a disabled run performs zero
    # extra RNG draws relative to the true baseline. Gated by `reset_state_curriculum_enabled`; a complete
    # no-op (no scene reads, no random draws) when that is False.
    #
    # Reproduces the direct-workflow Franka Cabinet environment's original four subtasks exactly
    # (isaaclab_tasks.direct.franka_cabinet.franka_cabinet_env.FrankaCabinetEnv._get_subtasks): 20cm/10cm
    # end-effector-to-handle proximity (subtask 2 is a pure distance check -- `require_graspable_pose=False`
    # -- not a grasp-pose gate), then the drawer open 20cm/38cm. Absolute meter thresholds are used for
    # subtasks 3/4 (rather than the default fraction-of-runtime-range) because this cabinet spawns unscaled,
    # so the original's literal values still apply -- see mdp/events.py's class docstring.
    subtask_progression_tracker = RewTerm(
        func=mdp.subtask_progression_tracker,
        weight=1.0,
        params={
            "proximity_threshold": 0.20,
            "touch_distance_threshold": 0.10,
            "require_graspable_pose": False,
            "slightly_open_threshold": 0.20,
            "almost_open_threshold": 0.38,
            "drawer_joint_name": "drawer_top_joint",
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # read-only: logs the raw (unsmoothed) drawer-opening success rate to TensorBoard, computed only from
    # natural (non-curriculum) episodes to avoid the replayed episodes inflating the number. See
    # mdp/curriculums.py's cabinet_success_rate docstring. Logged under "Curriculum/cabinet_success_rate".
    # Uses the same absolute 0.38 m bar as subtask 4 above (this cabinet is unscaled, so the direct-workflow
    # environment's original value applies directly). Read-only: it does not modify any environment
    # parameter, reward, observation, or termination, so it has no effect on training.
    cabinet_success_rate = CurrTerm(
        func=mdp.cabinet_success_rate, params={"success_threshold": 0.38, "drawer_joint_name": "drawer_top_joint"}
    )

    # surfaces the reset-pose curriculum's bookkeeping (see mdp/events.py) to TensorBoard under
    # "Curriculum/reset_pose_curriculum_metrics/...". Logging-only: it does not modify any environment
    # parameter, reward, observation, or termination, so it has no effect on training. Logs nothing (empty
    # dict) while `reset_state_curriculum_enabled` is False.
    reset_pose_curriculum_metrics = CurrTerm(
        func=mdp.reset_pose_curriculum_metrics, params={"tracker_term_name": "subtask_progression_tracker"}
    )


##
# Environment configuration
##


@configclass
class CabinetEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cabinet environment."""

    # Scene settings
    scene: CabinetSceneCfg = CabinetSceneCfg(num_envs=4096, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg | None = CurriculumCfg()

    # reset state curriculum (see mdp/events.py)
    reset_state_curriculum_enabled = True  # master switch; everything below is a full no-op while this is False
    success_buffer_size = 64
    prob_exp = 2  # how much we sharpen the probability distribution (1 = no sharpening)
    sampling_ratio = 0.3  # what fraction of resets go to the sampled (curriculum) distribution
    curriculum_dr = 0.02  # how much domain randomization to apply to replayed robot joints
    success_rate_alpha = 0.05  # momentum control of success rate movement (pre-calculations)
    greedy_margin = 0.10  # controls the margin between top and second distribution value that enables softmax

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 8.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
