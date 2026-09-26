# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""On-policy runner with a Reverse Curriculum Generation hook.

Reference:
    C. Florensa, D. Held, M. Wulfmeier, M. Zhang, P. Abbeel,
    "Reverse Curriculum Generation for Reinforcement Learning", CoRL 2017.
    https://arxiv.org/abs/1707.05300

The only thing the runner adds to :class:`~rsl_rl.runners.OnPolicyRunner` is *when* a
curriculum stage ends. Everything else -- start states, physics, success bookkeeping, Brownian
expansion -- lives in the environment, behind three methods:

* ``env.unwrapped.initialize_rcg()``
* ``env.unwrapped.rcg_stage_ready(policy_steps_in_stage)``
* ``env.unwrapped.advance_rcg_stage()``

so that the same runner works for any RCG task and the baseline never has to be modified.

The stage boundary is placed after the policy update and before the next rollout::

    24 rollout steps -> compute returns -> PPO update -> [RCG stage boundary] -> next rollout

which keeps the rollout storage filled with genuine on-policy transitions collected under a
single start-state distribution.
"""

from __future__ import annotations

import os
import time

import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan


class RCGOnPolicyRunner(OnPolicyRunner):
    """PPO runner that regenerates the environment's start-state curriculum between rollouts."""

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop for the specified number of iterations.

        A copy of :meth:`rsl_rl.runners.OnPolicyRunner.learn` with the curriculum hook added
        after the policy update. Kept as a copy rather than composed, because rsl-rl offers no
        per-iteration callback.
        """
        # -- RCG: resolve the curriculum configuration from the underlying environment
        base_env = self.env.unwrapped
        rcg_cfg = getattr(base_env.cfg, "rcg", None)
        rcg_enabled = bool(rcg_cfg is not None and rcg_cfg.enabled)

        if rcg_enabled:
            if self.alg.actor.is_recurrent or self.alg.critic.is_recurrent:
                raise NotImplementedError(
                    "[RCG] Recurrent policies are not supported: advance_rcg_stage() teleports every environment,"
                    " which would require the policy's hidden states to be reset at the same time."
                )
            # build the first pool before the first observation is read, so that the very first
            # rollout already starts from the curriculum distribution
            base_env.initialize_rcg()

        rcg_policy_steps = 0
        rcg_num_stages = 0

        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initialize the logging writer
        self.logger.init_logging_writer()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Check for NaN values from the environment
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards if RND is used (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    # Book keeping
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            # -- RCG: the safe curriculum boundary. The rollout storage has been consumed, and
            #    the next rollout has not started, so every environment may be teleported here.
            if rcg_enabled:
                rcg_policy_steps += self.env.num_envs * self.cfg["num_steps_per_env"]
                if base_env.rcg_stage_ready(rcg_policy_steps):
                    if base_env.advance_rcg_stage():
                        rcg_num_stages += 1
                        # every environment was just teleported, so the observations cached
                        # above describe states that no longer exist. Without this refresh the
                        # first transition of the next rollout would be corrupt.
                        obs = self.env.get_observations().to(self.device)
                    # reset the budget either way: if no good starts were found, the stage is
                    # retried after another full budget rather than on every iteration
                    rcg_policy_steps = 0

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            # Save model
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        # Save the final model after training and stop the logging writer
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()

        if rcg_enabled:
            print(f"[RCG] Training finished after {rcg_num_stages} curriculum stage advance(s).")
