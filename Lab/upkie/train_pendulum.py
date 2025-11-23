#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for Upkie Pendulum environment using PPO.
"""

from pathlib import Path

import gymnasium as gym
import upkie.envs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

upkie.envs.register()

# Training configuration
ENV_ID = "Upkie-PyBullet-Pendulum"  # Use PyBullet for training (faster than Spine)
ENV_KWARGS = dict(
    frequency=200.0,
    gui=False,  # Headless training
    regulate_frequency=False,
)
N_ENVS = 4  # Number of parallel environments
# Using DummyVecEnv (same process) to avoid PyBullet multiprocessing issues on macOS
# Each environment has its own PyBullet connection, so multiple envs work fine
# SubprocVecEnv (separate processes) has issues on macOS due to process spawning
TOTAL_TIMESTEPS = 1_000_000  # Total training steps
EVAL_FREQ = 50000  # Evaluate every N steps (less frequent to avoid issues)
N_EVAL_EPISODES = 3  # Number of episodes for evaluation

# Model save paths
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(exist_ok=True)
MODEL_NAME = "ppo_pendulum"
MODEL_PATH = MODELS_DIR / MODEL_NAME
TENSORBOARD_LOG = "./logs/tensorboard/"


def main():
    print("=" * 60)
    print("Training Upkie Pendulum Policy with PPO")
    print("=" * 60)
    print(f"Environment: {ENV_ID}")
    print(f"Parallel environments: {N_ENVS}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Model save path: {MODEL_PATH}")
    print("=" * 60)

    # Create vectorized environment
    # Using DummyVecEnv (same process) to avoid PyBullet multiprocessing issues on macOS
    # Each environment has its own PyBullet connection, so multiple envs work fine
    print("\nCreating training environment...")
    print(f"Creating {N_ENVS} parallel environments using DummyVecEnv...")
    
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
    env_fns = [make_env(i) for i in range(N_ENVS)]
    train_env = DummyVecEnv(env_fns)
    train_env.seed(0)

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

    # Create PPO model
    print("\nInitializing PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048 // N_ENVS,  # Adjust steps per env based on number of envs
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
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
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=str(MODELS_DIR / "checkpoints"),
        name_prefix="ppo_pendulum",
    )

    # Train the model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    print(f"Monitor training progress with: tensorboard --logdir {TENSORBOARD_LOG}")
    print("=" * 60 + "\n")

    # Prepare callbacks list (filter out None)
    callbacks = [cb for cb in [eval_callback, checkpoint_callback] if cb is not None]
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    print(f"\nSaving final model to {MODEL_PATH}...")
    model.save(str(MODEL_PATH))

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best model saved to: {MODELS_DIR / 'pendulum_best' / 'best_model.zip'}")
    print(f"Final model saved to: {MODEL_PATH}.zip")
    print("=" * 60)

    # Cleanup
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

