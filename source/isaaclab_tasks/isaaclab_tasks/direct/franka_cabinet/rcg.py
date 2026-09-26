import time
import math
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils import configclass
from rsl_rl.utils import check_nan

# TODO: tune
@configclass
class RCGCfg:
    enabled: bool = True

    # Good-start criterion
    r_min: float = 0.1
    r_max: float = 0.9

    # original RCG cocepts
    n_new: int = 200
    n_old: int = 100

    reset_state_size = 9 + 9 + 1 # 9 for Franka Pos, 9 for Franka Target, 1 for Cabinet

    candidate_count: int = 10_000
    brownian_horizon: int = 50
    brownian_action_std: float = 1.0

    # how long PPO trains before regenerating curriculum
    policy_steps_per_stage: int = 250_000

class RCGOnPolicyRunner(OnPolicyRunner):
    
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
            """Run the learning loop for the specified number of iterations."""
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

                policy_steps_this_iter = (
                    self.env.num_envs
                    * self.cfg["num_steps_per_env"]
                )

                rcg_policy_steps += policy_steps_this_iter

                if rcg_policy_steps > rcg_steps_per_stage:
                    base_env = self.env.unwrapped

                    updated = base_env.advance_rcg_stage()

                    if updated:
                        # Environment state changed outside normal env.step()
                        # Runner's old 'obs' is now stale.
                        obs = self.env.get_observations().to(self.device)

                    rcg_policy_steps = 9

    
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
    