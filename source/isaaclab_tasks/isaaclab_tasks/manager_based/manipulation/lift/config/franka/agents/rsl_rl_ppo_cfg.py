# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "franka_lift"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class LiftCubeBenchmarkPPORunnerCfg(LiftCubePPORunnerCfg):
    """Shared by both arms of the reverse-curriculum benchmark, so they cannot drift apart.

    One deviation from :class:`LiftCubePPORunnerCfg`, applied to *both* arms:
    ``policy.noise_std_type = "log"``.

    rsl-rl's default, ``"scalar"``, makes the action standard deviation a raw ``nn.Parameter`` that
    nothing constrains to be positive. A single bad update can drive one of its eight components to
    zero or below -- or to ``NaN``, which also fails the check -- and training then dies inside
    ``alg.update()`` with ``RuntimeError: normal expects all elements of std >= 0.0``, several
    frames away from whatever actually went wrong. ``"log"`` stores ``log(std)`` and exponentiates,
    so the parameter cannot express a negative std at all.

    This is insurance, not a cure: it removes one confusing failure mode and it cannot turn a
    genuinely diverged run into a converged one. It is applied to both arms because a PPO
    hyperparameter that differs between them would confound the comparison.
    """

    clip_actions = 10.0
    """Hard bound on the action passed to the environment, applied by ``RslRlVecEnvWrapper``.

    A circuit breaker, not a tuning knob. Franka Lift's ``JointPositionAction`` maps an action to
    ``action * 0.5 + default_joint_pos``, and the arm's joints span roughly +/-2.9 rad, so any
    action beyond about +/-6 is already meaningless -- ``10.0`` is far outside the range a working
    policy ever uses and leaves the baseline untouched.

    What it stops is a feedback loop that the RCG arm reliably falls into and the baseline does not.
    ``last_action`` and ``joint_vel_rel`` are both observation terms and neither is bounded; the
    ``action_rate_l2`` and ``joint_vel_l2`` reward terms are unbounded quadratics whose weights the
    task's own ``CurriculumCfg`` multiplies by 1000 after 10k steps. So a policy whose mean drifts
    outward drives joint targets far outside their limits, which produces joint velocities in the
    hundreds of rad/s, which enter the observation, which drive the mean further out -- and the
    quadratic penalties turn that into per-step rewards of -26 against a normal ceiling of +0.74.
    Measured without this bound, observations and actions reached 122 and the value loss overflowed
    to ``inf`` within a few hundred iterations.

    Clipping bounds every link in that chain at once: the observation, the joint target, and both
    penalties. Applied to *both* arms, because a wrapper setting that differed between them would
    confound the comparison.
    """

    def __post_init__(self):
        super().__post_init__()
        self.policy.noise_std_type = "log"


@configclass
class LiftCubeBaselinePPORunnerCfg(LiftCubeBenchmarkPPORunnerCfg):
    """PPO baseline arm of the reverse-curriculum benchmark.

    Identical to :class:`LiftCubeBenchmarkPPORunnerCfg` in every PPO hyperparameter. Only the log
    directory differs, so that the baseline's runs sit beside the RCG arm's instead of mixing into
    the upstream task's.
    """

    experiment_name = "franka_lift_baseline"


@configclass
class LiftCubeRCGPPORunnerCfg(LiftCubeBenchmarkPPORunnerCfg):
    """PPO with reverse curriculum generation over start states.

    Identical to :class:`LiftCubeBenchmarkPPORunnerCfg` in every PPO hyperparameter, so that a
    comparison between the arms isolates the effect of the curriculum. Only the runner class -- which
    decides when a curriculum stage ends, and nothing else -- and the log directory differ.
    """

    class_name = "RCGOnPolicyRunner"
    experiment_name = "franka_lift_rcg"
