#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rollout script to test trained Upkie Pendulum policy.
"""

import sys
from pathlib import Path

import upkie.envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

upkie.envs.register()

# Configuration
MODEL_PATH = "./models/pendulum_best/best_model.zip"
ENV_ID = "Upkie-PyBullet-Pendulum"  # Use PyBullet for testing (can switch to Spine if available)
ENV_KWARGS = dict(
    frequency=200.0,
    gui=True,  # Show GUI for visualization
    regulate_frequency=False,
)
SEED = 0
N_EPISODES = 5  # Number of episodes to run


def main():
    # Check if model exists
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train a model first using train_pendulum.py")
        sys.exit(1)

    print("=" * 60)
    print("Testing Upkie Pendulum Policy")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Environment: {ENV_ID}")
    print(f"Number of episodes: {N_EPISODES}")
    print("=" * 60)

    # Create environment
    env = make_vec_env(ENV_ID, n_envs=1, env_kwargs=ENV_KWARGS, seed=SEED)

    # Load trained model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = PPO.load(str(model_path), env=env)

    # Run episodes
    print("\nRunning episodes...\n")
    episode_returns = []
    episode_lengths = []

    for episode in range(1, N_EPISODES + 1):
        obs, _ = env.reset()
        ep_return = 0.0
        ep_len = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward[0])
            ep_len += 1
            done = bool(terminated[0] or truncated[0])

        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        print(f"Episode {episode}: return={ep_return:.3f}, length={ep_len} steps")

    # Print statistics
    print("\n" + "=" * 60)
    print("Evaluation Statistics")
    print("=" * 60)
    print(f"Mean return: {sum(episode_returns) / len(episode_returns):.3f}")
    print(f"Std return: {(sum((x - sum(episode_returns)/len(episode_returns))**2 for x in episode_returns) / len(episode_returns))**0.5:.3f}")
    print(f"Mean length: {sum(episode_lengths) / len(episode_lengths):.1f} steps")
    print(f"Std length: {(sum((x - sum(episode_lengths)/len(episode_lengths))**2 for x in episode_lengths) / len(episode_lengths))**0.5:.1f} steps")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
