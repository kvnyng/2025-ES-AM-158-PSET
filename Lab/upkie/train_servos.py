#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for Upkie Servos environment using PPO.

Supports:
- Training from scratch
- Resuming from checkpoints
- Periodic checkpointing during training
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import upkie.envs
import upkie.logging
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import the wrappers from rollout_policy_servos
from rollout_policy_servos import ServoObsFlattenWrapper, ServoVelActionWrapper
from upkie.envs.servos_reward_wrapper import ServosRewardWrapper

upkie.envs.register()
# Disable upkie warnings during training (termination conditions are expected)
upkie.logging.disable_warnings()

# Training configuration
ENV_ID = "Upkie-PyBullet-Servos"  # Use PyBullet for training (faster than Spine)
DEFAULT_FREQUENCY = 200.0  # Control loop frequency in Hz
ENV_KWARGS = dict(
    gui=False,  # Headless training
    regulate_frequency=False,
)
N_ENVS = 8  # Number of parallel environments (reduced from 10 for servos)
TOTAL_TIMESTEPS = 2_000_000  # Total training steps (more for servos)
EVAL_FREQ = 50000  # Evaluate every N steps
N_EVAL_EPISODES = 3  # Number of episodes for evaluation

# Model save paths
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(exist_ok=True)
MODEL_NAME_BASE = "ppo_servos"
TENSORBOARD_LOG = "./logs/tensorboard/"

# Checkpoint configuration
CHECKPOINT_FREQ = 50000  # Save checkpoint every N steps
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"

# Default gains for action wrapper
DEFAULT_GAINS = dict(kp_wheel=0.0, kd_wheel=1.7, kp_leg=2.0, kd_leg=1.7)


class LossCurveCallback(BaseCallback):
    """
    Custom callback to explicitly log loss curves to TensorBoard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.loss_history = {
            "policy_loss": [],
            "value_loss": [],
            "total_loss": [],
            "steps": [],
        }

    def _on_step(self) -> bool:
        """Called at each training step."""
        if self.logger is not None:
            try:
                name_to_value = getattr(self.logger, "name_to_value", {})

                policy_loss = name_to_value.get("train/policy_loss", None)
                value_loss = name_to_value.get("train/value_loss", None)
                total_loss = name_to_value.get("train/loss", None)

                if total_loss is None and policy_loss is not None and value_loss is not None:
                    total_loss = policy_loss + value_loss

                current_step = self.num_timesteps

                if policy_loss is not None:
                    self.logger.record("loss_curve/policy_loss", policy_loss)
                    if (
                        not self.loss_history["policy_loss"]
                        or self.loss_history["policy_loss"][-1] != policy_loss
                    ):
                        self.loss_history["policy_loss"].append(policy_loss)

                if value_loss is not None:
                    self.logger.record("loss_curve/value_loss", value_loss)
                    if (
                        not self.loss_history["value_loss"]
                        or self.loss_history["value_loss"][-1] != value_loss
                    ):
                        self.loss_history["value_loss"].append(value_loss)

                if total_loss is not None:
                    self.logger.record("loss_curve/total_loss", total_loss)
                    if (
                        not self.loss_history["total_loss"]
                        or self.loss_history["total_loss"][-1] != total_loss
                    ):
                        self.loss_history["total_loss"].append(total_loss)
                        self.loss_history["steps"].append(current_step)

                        if len(self.loss_history["total_loss"]) > 1:
                            window_size = min(10, len(self.loss_history["total_loss"]))
                            recent_losses = self.loss_history["total_loss"][-window_size:]
                            smoothed_loss = np.mean(recent_losses)
                            self.logger.record("loss_curve/total_loss_smoothed", smoothed_loss)
            except Exception:
                pass

        return True


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find the latest checkpoint in the checkpoint directory."""
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("ppo_servos_*_steps.zip"))
    if not checkpoints:
        return None

    def get_step_number(path: Path) -> int:
        parts = path.stem.split("_")
        if len(parts) >= 3:
            try:
                return int(parts[2])
            except ValueError:
                return 0
        return 0

    checkpoints.sort(key=get_step_number, reverse=True)
    return checkpoints[0]


def make_wrapped_env(
    *,
    frequency_hz: float = 200.0,
    max_steps: int = 300,
    fixed_order: Optional[Dict[str, list]] = None,
    gains: Optional[Dict[str, float]] = None,
) -> gym.Env:
    """Create a wrapped servos environment with reward shaping."""
    # Create base environment
    env = gym.make(ENV_ID, frequency=frequency_hz, **ENV_KWARGS)
    
    # Add reward wrapper (must be before action/obs wrappers to access full spine obs)
    env = ServosRewardWrapper(
        env,
        fall_pitch=1.0,
        max_position_drift=5.0,
        max_angular_velocity=10.0,
        max_linear_velocity=2.0,
    )
    
    # Add action wrapper (converts Box to Dict actions)
    env = ServoVelActionWrapper(env, fixed_order=fixed_order, gains=gains or DEFAULT_GAINS)
    
    # Add observation wrapper (flattens Dict to Box)
    env = ServoObsFlattenWrapper(env)
    
    # Add time limit
    env = TimeLimit(env, max_episode_steps=max_steps)
    
    return env


def main():
    parser = argparse.ArgumentParser(description="Train Upkie Servos Policy with PPO")
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
    n_envs = args.n_envs if args.n_envs is not None else N_ENVS

    print("=" * 60)
    print("Training Upkie Servos Policy with PPO")
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
    print("\nCreating training environment...")
    print(f"Creating {n_envs} parallel environments using DummyVecEnv...")

    def make_env(rank: int):
        """Create a single environment with a unique seed."""

        def _init():
            env = make_wrapped_env(
                frequency_hz=DEFAULT_FREQUENCY,
                max_steps=300,
                fixed_order=None,
                gains=DEFAULT_GAINS,
            )
            # Wrap with Monitor to track statistics
            env = Monitor(env, filename=None, allow_early_resets=True)
            return env

        return _init

    # Create list of environment factory functions
    env_fns = [make_env(i) for i in range(n_envs)]
    train_env = DummyVecEnv(env_fns)
    train_env.seed(0)

    # Normalize rewards and observations to stabilize training
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    # Create evaluation environment
    print("Creating evaluation environment...")

    def make_eval_env():
        env = make_wrapped_env(
            frequency_hz=DEFAULT_FREQUENCY,  # Match training frequency
            max_steps=300,
            fixed_order=None,
            gains=DEFAULT_GAINS,
        )
        eval_log_dir = str(MODELS_DIR / "servos_best" / "eval_logs")
        env = Monitor(env, eval_log_dir)
        return env

    eval_env = DummyVecEnv([make_eval_env])
    eval_env.seed(42)

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

        def linear_schedule(initial_value: float, final_value: float = 5e-5):
            """Linear learning rate schedule with slower decay."""

            def func(progress_remaining: float) -> float:
                if progress_remaining > 0.5:
                    return initial_value * 0.5 + (initial_value * 0.5 - final_value) * (
                        1.0 - progress_remaining
                    ) * 2.0
                else:
                    return final_value + (initial_value * 0.5 - final_value) * progress_remaining * 2.0

            return func

        learning_rate_init = args.learning_rate if args.learning_rate is not None else 2e-4
        batch_size = args.batch_size if args.batch_size is not None else 128
        n_epochs = args.n_epochs if args.n_epochs is not None else 5
        vf_coef = args.vf_coef if args.vf_coef is not None else 0.5
        clip_range = args.clip_range if args.clip_range is not None else 0.2

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=linear_schedule(learning_rate_init, 1e-5),
            n_steps=2048 // n_envs if n_envs > 0 else 2048,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=clip_range,
            clip_range_vf=0.2,
            ent_coef=0.0,
            vf_coef=vf_coef,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=[128, 128],  # Larger network for servos (6 joints vs 1 action)
            ),
            tensorboard_log=TENSORBOARD_LOG,
            verbose=1,
            device="auto",
        )

    # Setup callbacks
    print("\nSetting up callbacks...")
    print("Evaluation callback disabled to avoid robot deletion issues")
    print("You can evaluate the model manually after training using rollout_policy_servos.py")

    CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_servos",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")
    print(f"Checkpoint frequency: Every {CHECKPOINT_FREQ:,} steps")

    loss_curve_callback = LossCurveCallback(verbose=1)
    print("Loss curve callback enabled - loss metrics will be logged to TensorBoard")

    # Train the model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    print(f"Monitor training progress with: tensorboard --logdir {TENSORBOARD_LOG}")
    print("=" * 60 + "\n")

    callbacks = [checkpoint_callback, loss_curve_callback]

    if resume_from:
        remaining_steps = max(0, total_timesteps - model.num_timesteps)
        learn_timesteps = remaining_steps
        reset_num_timesteps = False
    else:
        learn_timesteps = total_timesteps
        reset_num_timesteps = True

    try:
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
            if latest_model_path.exists() or latest_model_path.is_symlink():
                latest_model_path.unlink()
            latest_model_path.symlink_to(MODEL_PATH.name)
            print(f"Created symlink: {latest_model_path} -> {MODEL_PATH.name}")
        except Exception as e:
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
        print(f"  python train_servos.py --resume {CHECKPOINT_DIR}/ppo_servos_XXXXXX_steps.zip")
        print("=" * 60)
    finally:
        # Cleanup environments before Python shutdown to avoid tqdm/rich cleanup errors
        try:
            train_env.close()
        except Exception:
            pass
        try:
            eval_env.close()
        except Exception:
            pass
        # Small delay to allow progress bar to finish cleanup
        time.sleep(0.1)


if __name__ == "__main__":
    main()

