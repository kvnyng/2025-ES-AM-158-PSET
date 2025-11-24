#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for Upkie Pendulum environment using PPO.

Supports:
- Training from scratch
- Resuming from checkpoints
- Periodic checkpointing during training
"""

import argparse
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import upkie.envs
import upkie.logging
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

upkie.envs.register()
# Disable upkie warnings during training (termination conditions are expected)
upkie.logging.disable_warnings()

# Training configuration
ENV_ID = "Upkie-PyBullet-Pendulum"  # Use PyBullet for training (faster than Spine)
ENV_KWARGS = dict(
    frequency=200.0,
    gui=False,  # Headless training
    regulate_frequency=False,
)
N_ENVS = 10  # Number of parallel environments
# Using DummyVecEnv (same process) to avoid PyBullet multiprocessing issues on macOS
# Each environment has its own PyBullet connection, so multiple envs work fine
# SubprocVecEnv (separate processes) has issues on macOS due to process spawning
TOTAL_TIMESTEPS = 1_000_000  # Total training steps
EVAL_FREQ = 50000  # Evaluate every N steps (less frequent to avoid issues)
N_EVAL_EPISODES = 3  # Number of episodes for evaluation

# Model save paths
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(exist_ok=True)
MODEL_NAME_BASE = "ppo_pendulum"
TENSORBOARD_LOG = "./logs/tensorboard/"

# Checkpoint configuration
CHECKPOINT_FREQ = 50000  # Save checkpoint every N steps
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"


class LossCurveCallback(BaseCallback):
    """
    Custom callback to explicitly log loss curves to TensorBoard.
    
    This callback ensures that loss metrics (policy loss, value loss, total loss)
    are clearly visible in TensorBoard for monitoring training progress.
    The loss curves will appear under the "loss_curve" tag in TensorBoard.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.loss_history = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'steps': []
        }
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        return True
    
    def _on_step(self) -> bool:
        """
        Called at each training step.
        We check for loss values after training updates occur.
        """
        # Access training statistics from the logger
        # Stable-baselines3 automatically logs these during training updates
        # We'll also log them under a dedicated "loss_curve" tag for easy viewing
        if self.logger is not None:
            try:
                # Get the latest logged values from the logger
                # These are populated by Stable-baselines3 during the training update
                name_to_value = getattr(self.logger, 'name_to_value', {})
                
                # Extract loss values if available
                # Stable-baselines3 logs these under 'train/policy_loss', 'train/value_loss', etc.
                policy_loss = name_to_value.get('train/policy_loss', None)
                value_loss = name_to_value.get('train/value_loss', None)
                total_loss = name_to_value.get('train/loss', None)
                
                # If total loss is not directly available, compute it from components
                if total_loss is None and policy_loss is not None and value_loss is not None:
                    total_loss = policy_loss + value_loss
                
                # Log to TensorBoard with explicit tags for easy viewing
                # Only log if we have new values to avoid duplicate logging
                current_step = self.num_timesteps
                
                if policy_loss is not None:
                    self.logger.record('loss_curve/policy_loss', policy_loss)
                    if not self.loss_history['policy_loss'] or self.loss_history['policy_loss'][-1] != policy_loss:
                        self.loss_history['policy_loss'].append(policy_loss)
                
                if value_loss is not None:
                    self.logger.record('loss_curve/value_loss', value_loss)
                    if not self.loss_history['value_loss'] or self.loss_history['value_loss'][-1] != value_loss:
                        self.loss_history['value_loss'].append(value_loss)
                
                if total_loss is not None:
                    self.logger.record('loss_curve/total_loss', total_loss)
                    if not self.loss_history['total_loss'] or self.loss_history['total_loss'][-1] != total_loss:
                        self.loss_history['total_loss'].append(total_loss)
                        self.loss_history['steps'].append(current_step)
                        
                        # Also log a smoothed version of the loss for better visualization
                        if len(self.loss_history['total_loss']) > 1:
                            # Compute moving average over last 10 values
                            window_size = min(10, len(self.loss_history['total_loss']))
                            recent_losses = self.loss_history['total_loss'][-window_size:]
                            smoothed_loss = np.mean(recent_losses)
                            self.logger.record('loss_curve/total_loss_smoothed', smoothed_loss)
            except Exception as e:
                # Silently handle any errors to avoid disrupting training
                # The loss values are already logged by Stable-baselines3, so this is just for convenience
                pass
        
        return True


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find the latest checkpoint in the checkpoint directory."""
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("ppo_pendulum_*_steps.zip"))
    if not checkpoints:
        return None
    
    # Sort by step number (extract from filename)
    def get_step_number(path: Path) -> int:
        # Extract number from "ppo_pendulum_XXXXXX_steps.zip"
        parts = path.stem.split("_")
        if len(parts) >= 3:
            try:
                return int(parts[2])
            except ValueError:
                return 0
        return 0
    
    checkpoints.sort(key=get_step_number, reverse=True)
    return checkpoints[0]


def main():
    parser = argparse.ArgumentParser(description="Train Upkie Pendulum Policy with PPO")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (or 'latest' to resume from most recent checkpoint)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {TOTAL_TIMESTEPS:,})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides default)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides default)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=None,
        help="Number of epochs per update (overrides default)",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=None,
        help="Value function coefficient (overrides default)",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=None,
        help="PPO clip range (overrides default)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Number of parallel environments (overrides default)",
    )
    args = parser.parse_args()
    
    # Determine if we're resuming from a checkpoint
    resume_from = None
    if args.resume:
        if args.resume.lower() == "latest":
            resume_from = find_latest_checkpoint(CHECKPOINT_DIR)
            if resume_from:
                print(f"Found latest checkpoint: {resume_from}")
            else:
                print("No checkpoints found. Starting training from scratch.")
        else:
            resume_from = Path(args.resume)
            if not resume_from.exists():
                raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
    
    total_timesteps = args.total_timesteps
    
    # Override N_ENVS if specified
    n_envs = args.n_envs if args.n_envs is not None else N_ENVS
    
    print("=" * 60)
    print("Training Upkie Pendulum Policy with PPO")
    print("=" * 60)
    print(f"Environment: {ENV_ID}")
    print(f"Parallel environments: {n_envs}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Model will be saved with timestamp when training completes")
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
    else:
        print("Starting training from scratch")
    print("=" * 60)
    
    # Create vectorized environment
    # Using DummyVecEnv (same process) to avoid PyBullet multiprocessing issues on macOS
    # Each environment has its own PyBullet connection, so multiple envs work fine
    print("\nCreating training environment...")
    print(f"Creating {n_envs} parallel environments using DummyVecEnv...")
    
    # Create multiple environments in the same process
    # Each will have its own PyBullet connection
    def make_env(rank: int):
        """Create a single environment with a unique seed."""
        def _init():
            env = gym.make(ENV_ID, **ENV_KWARGS)
            # Wrap with Monitor to track statistics
            env = Monitor(env, filename=None, allow_early_resets=True)
            return env
        return _init
    
    # Create list of environment factory functions
    env_fns = [make_env(i) for i in range(n_envs)]
    train_env = DummyVecEnv(env_fns)
    train_env.seed(0)
    
    # CRITICAL: Normalize rewards and observations to stabilize value function training
    # This prevents value loss from diverging by normalizing returns
    train_env = VecNormalize(
        train_env,
        norm_obs=True,  # Normalize observations
        norm_reward=True,  # Normalize rewards - CRITICAL for preventing value loss divergence
        clip_obs=10.0,  # Clip normalized observations
        clip_reward=10.0,  # Clip normalized rewards
        gamma=0.99,  # Match discount factor for reward normalization
    )

    # Create evaluation environment (single env, wrapped with Monitor)
    print("Creating evaluation environment...")
    def make_eval_env():
        env = gym.make(ENV_ID, **ENV_KWARGS)
        # Wrap with Monitor to track evaluation statistics
        eval_log_dir = str(MODELS_DIR / "pendulum_best" / "eval_logs")
        env = Monitor(env, eval_log_dir)
        return env
    eval_env = DummyVecEnv([make_eval_env])
    eval_env.seed(42)
    
    # Evaluation environment should use the same normalization as training
    # We'll sync it with training env normalization after model creation

    # Create or load PPO model
    if resume_from:
        print(f"\nLoading model from checkpoint: {resume_from}")
        model = PPO.load(str(resume_from), env=train_env)
        print(f"Resumed from step: {model.num_timesteps:,}")
        
        # Try to load normalization statistics if they exist
        checkpoint_path = Path(resume_from)
        norm_stats_path = checkpoint_path.parent / f"{MODEL_NAME_BASE}_norm_stats.pkl"
        if norm_stats_path.exists() and isinstance(train_env, VecNormalize):
            train_env.load(str(norm_stats_path))
            print(f"Loaded normalization statistics from: {norm_stats_path}")
        
        remaining_steps = max(0, total_timesteps - model.num_timesteps)
        if remaining_steps == 0:
            print("Model has already completed training. Exiting.")
            return
        print(f"Remaining steps to train: {remaining_steps:,}")
    else:
        print("\nInitializing new PPO model...")
        # Learning rate schedule: slower decay to prevent premature convergence
        # Keep learning rate higher for longer to allow continued improvement
        def linear_schedule(initial_value: float, final_value: float = 5e-5):
            """Linear learning rate schedule with slower decay."""
            def func(progress_remaining: float) -> float:
                # Decay more slowly - keep 50% of initial LR until 50% progress
                # Then decay to final value
                if progress_remaining > 0.5:
                    # First half: keep at higher rate
                    return initial_value * 0.5 + (initial_value * 0.5 - final_value) * (1.0 - progress_remaining) * 2.0
                else:
                    # Second half: decay to final value
                    return final_value + (initial_value * 0.5 - final_value) * progress_remaining * 2.0
            return func
        
        # Use command-line arguments if provided, otherwise use defaults
        # Increased default learning rate to help with loss plateau
        learning_rate_init = args.learning_rate if args.learning_rate is not None else 2e-4
        batch_size = args.batch_size if args.batch_size is not None else 128
        n_epochs = args.n_epochs if args.n_epochs is not None else 5
        vf_coef = args.vf_coef if args.vf_coef is not None else 0.5
        clip_range = args.clip_range if args.clip_range is not None else 0.2
        
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=linear_schedule(learning_rate_init, 1e-5),
            # Keep total rollout size constant: 2048 total steps across all envs
            # This ensures consistent batch sizes regardless of n_envs
            n_steps=2048 // n_envs if n_envs > 0 else 2048,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=clip_range,
            clip_range_vf=0.2,  # Value function clipping to prevent value function from diverging
            ent_coef=0.0,
            vf_coef=vf_coef,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=[64, 64],  # Two hidden layers with 64 units each
            ),
            tensorboard_log=TENSORBOARD_LOG,
            verbose=1,
            device="auto",
        )

    # Setup callbacks
    print("\nSetting up callbacks...")
    
    # Evaluation callback - disable to avoid robot deletion issues during training
    # You can evaluate manually after training using rollout_policy.py
    eval_callback = None
    # Uncomment below to enable evaluation (may cause issues if robot gets deleted)
    # try:
    #     eval_callback = EvalCallback(
    #         eval_env,
    #         best_model_save_path=str(MODELS_DIR / "pendulum_best"),
    #         log_path=str(MODELS_DIR / "pendulum_best" / "logs"),
    #         eval_freq=EVAL_FREQ,
    #         n_eval_episodes=N_EVAL_EPISODES,
    #         deterministic=True,
    #         render=False,
    #         warn=False,
    #     )
    #     print("Evaluation callback enabled")
    # except Exception as e:
    #     print(f"Warning: Could not create evaluation callback: {e}")
    print("Evaluation callback disabled to avoid robot deletion issues")
    print("You can evaluate the model manually after training using rollout_policy.py")

    # Checkpoint callback (save periodically)
    # Save checkpoints every CHECKPOINT_FREQ steps for recovery and analysis
    # Checkpoints include: model weights, optimizer state, training metadata
    CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_pendulum",
        save_replay_buffer=False,  # Don't save replay buffer (PPO doesn't use one)
        save_vecnormalize=True,  # Save VecNormalize statistics for proper resuming
    )
    print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")
    print(f"Checkpoint frequency: Every {CHECKPOINT_FREQ:,} steps")
    
    # Loss curve callback for TensorBoard visualization
    loss_curve_callback = LossCurveCallback(verbose=1)
    print("Loss curve callback enabled - loss metrics will be logged to TensorBoard")

    # Train the model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    print(f"Monitor training progress with: tensorboard --logdir {TENSORBOARD_LOG}")
    print("=" * 60 + "\n")

    # Prepare callbacks list (filter out None)
    # Include loss curve callback to ensure loss metrics are visible in TensorBoard
    callbacks = [cb for cb in [eval_callback, checkpoint_callback, loss_curve_callback] if cb is not None]
    
    # Calculate remaining timesteps if resuming
    if resume_from:
        remaining_steps = max(0, total_timesteps - model.num_timesteps)
        learn_timesteps = remaining_steps
        reset_num_timesteps = False  # Don't reset timestep counter when resuming
    else:
        learn_timesteps = total_timesteps
        reset_num_timesteps = True
    
    model.learn(
        total_timesteps=learn_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
    )

    # Generate timestamp when training finishes
    finish_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_NAME = f"{MODEL_NAME_BASE}_{finish_timestamp}"
    MODEL_PATH = MODELS_DIR / MODEL_NAME
    
    # Save final model with timestamp
    print(f"\nSaving final model to {MODEL_PATH}...")
    model.save(str(MODEL_PATH))
    
    # Save normalization statistics if using VecNormalize
    if isinstance(train_env, VecNormalize):
        norm_stats_path = MODELS_DIR / f"{MODEL_NAME_BASE}_norm_stats.pkl"
        train_env.save(str(norm_stats_path))
        print(f"Saved normalization statistics to: {norm_stats_path}")
    
    # Also create a symlink or copy to the latest model for convenience
    latest_model_path = MODELS_DIR / f"{MODEL_NAME_BASE}_latest"
    try:
        # Remove old symlink if it exists
        if latest_model_path.exists() or latest_model_path.is_symlink():
            latest_model_path.unlink()
        # Create symlink to the timestamped model
        latest_model_path.symlink_to(MODEL_PATH.name)
        print(f"Created symlink: {latest_model_path} -> {MODEL_PATH.name}")
    except Exception as e:
        # If symlink fails (e.g., on Windows), just print a message
        print(f"Note: Could not create symlink (this is OK): {e}")

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Final model saved to: {MODEL_PATH}.zip")
    print(f"Model name: {MODEL_NAME}")
    print(f"Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}/")
    print(f"TensorBoard logs: {TENSORBOARD_LOG}")
    print("=" * 60)
    print("\nTo resume training from a checkpoint:")
    print(f"  model = PPO.load('{CHECKPOINT_DIR}/ppo_pendulum_XXXXXX_steps.zip')")
    print("  model.learn(total_timesteps=remaining_steps, reset_num_timesteps=False)")
    print("=" * 60)

    # Cleanup
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

