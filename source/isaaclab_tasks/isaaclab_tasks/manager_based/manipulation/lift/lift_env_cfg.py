# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.utils.rcg import RCGCfg

from . import mdp

DEFAULT_GOAL_STATE_PATH = os.path.join(os.path.dirname(__file__), "data", "goal_states_franka_lift.pt")
"""Default location of the recorded RCG goal states, written by ``scripts/rcg/record_goal_states.py``."""

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
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
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    ##
    # Task success, and reverse curriculum generation over start states.
    #
    # Inert on this configuration: nothing here is read unless the environment class is
    # `LiftRCGEnv`, and `rcg.enabled` is False, so `Isaac-Lift-Cube-Franka-v0` and the IK variants
    # are unaffected. See `config/franka/lift_rcg_env_cfg.py` for the two benchmark arms.
    ##

    success_threshold = 0.02
    """How close the object must get to the commanded goal position to count as success, in metres.

    Not a termination condition: Franka Lift ends an episode only on time-out or on dropping the
    object, and that is left alone. Success here is something *measured*. The value matches the
    threshold the reset-pose curriculum's ``lift_success_rate`` metric uses, so the two curricula's
    curves are on one scale, and it is well inside the ``0.05`` ``std`` of the
    ``object_goal_tracking_fine_grained`` reward term. No separate height test is needed: the
    commanded goal sits at ``z`` in ``(0.25, 0.5)`` and the cube rests at ``z ~ 0.055``, so the
    object cannot be within 2 cm of the goal without having been lifted.
    """

    success_rate_ema_alpha = 0.01
    """Smoothing for the episodic success-rate metrics, applied once per completed episode."""

    rcg: RCGCfg = RCGCfg(
        goal_state_path=DEFAULT_GOAL_STATE_PATH,
        # Franka Lift's episode is 250 steps, half of Franka Cabinet's, so the same "300 starts x
        # 8 episodes" stage budget is half the environment steps. See RCGCfg.policy_steps_per_stage.
        policy_steps_per_stage=600_000,
        max_policy_steps_per_stage=2_000_000,
        # this task does not terminate on success, so a start state one step from the goal would
        # otherwise be scored on whether the policy can *hold* the goal for 250 steps rather than
        # on whether it can reach it. See RCGCfg.episode_success_mode.
        episode_success_mode="ever",
        # The paper's T_B = 50 does not transfer to this task, and using it is the difference
        # between a curriculum that works and one that does nothing. 50 steps at 50 Hz is a full
        # second of random joint targets applied to an arm holding a cube: the cube is long gone and
        # the arm has wandered anywhere. Measured with `--dry_run_expand` over 256 environments, the
        # progress of the generated starts (1.0 = in the goal set, goal states measure 0.99):
        #
        #   T_B = 50   mean 0.20, spread flat across [0, 1]   <- indistinguishable from random
        #   T_B = 10   mean 0.75, concentrated in [0.5, 1.0]   <- a real difficulty gradient
        #   T_B =  3   mean 0.90, all inside [0.8, 1.0]        <- too close, mostly mastered
        #
        # At 50 the new starts are unsolvable and the replayed archive starts are trivial, so every
        # start's success rate is exactly 0 or exactly 1, `select()` finds nothing in (r_min, r_max)
        # and the curriculum never advances on its own merits. Franka Cabinet tolerates T_B = 50
        # because its drawer is spring-loaded and the arm stays near the handle; a free cube has no
        # such restoring force. Re-run `--dry_run_expand` if the object, the gripper or the episode
        # rate changes -- the right value depends on all three.
        brownian_horizon=10,
    )
    """Reverse curriculum generation. Disabled by default; enabled by ``FrankaCubeLiftRCGEnvCfg``."""

    progress_reference_distance = 0.35
    """Object-to-goal distance treated as "the start of the task" when reporting pool progress.

    Only used by ``_rcg_pool_progress``, which is a reporting quantity -- nothing in the curriculum
    reads it. ``0.35`` m is roughly the object-to-goal distance at a fresh ``rho_0`` start: the cube
    rests at ``z ~ 0.055`` and the goal is sampled in ``z`` in ``(0.25, 0.5)`` with up to 0.25 m of
    lateral offset.
    """

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
