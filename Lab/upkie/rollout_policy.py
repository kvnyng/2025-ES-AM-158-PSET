#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rollout script to test trained Upkie Pendulum policy.

Supports:
- Fast headless mode for quick evaluation
- GUI mode for visualization
- Configurable number of episodes
"""

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import upkie.envs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    import pybullet
    HAS_PYBULLET = True
except ImportError:
    pybullet = None
    HAS_PYBULLET = False

upkie.envs.register()

# Default configuration
MODEL_PATH_BEST = "./models/pendulum_best/best_model.zip"
MODEL_PATH_FINAL = "./models/ppo_pendulum.zip"
ENV_ID = "Upkie-PyBullet-Pendulum"  # Use PyBullet for testing (can switch to Spine if available)
SEED = 0
N_EPISODES = 5  # Number of episodes to run


def main():
    parser = argparse.ArgumentParser(description="Test trained Upkie Pendulum policy")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run in fast headless mode (no GUI, faster evaluation)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Force GUI mode (overrides --fast)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=N_EPISODES,
        help=f"Number of episodes to run (default: {N_EPISODES})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model file (default: auto-detect best or final model)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Run in real-time mode (slower, but matches real robot timing)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Save video to file (e.g., rollout.mp4 or just 'rollout' for auto-naming). "
             "Videos are saved to ./videos/ folder. Requires GUI mode.",
    )
    args = parser.parse_args()
    
    # Determine model path
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Error: Model not found at {model_path}")
            sys.exit(1)
    else:
        # Try best model first, fall back to final model
        if Path(MODEL_PATH_BEST).exists():
            model_path = Path(MODEL_PATH_BEST)
            print(f"Using best model: {MODEL_PATH_BEST}")
        elif Path(MODEL_PATH_FINAL).exists():
            model_path = Path(MODEL_PATH_FINAL)
            print(f"Using final model: {MODEL_PATH_FINAL}")
        else:
            print(f"Error: No model found at {MODEL_PATH_BEST} or {MODEL_PATH_FINAL}")
            print("Please train a model first using train_pendulum.py")
            sys.exit(1)

    # Determine GUI setting
    use_gui = args.gui or (not args.fast) or (args.video is not None)  # GUI needed for video
    
    # Video export requires GUI
    if args.video and not use_gui:
        print("Warning: --video requires GUI mode. Enabling GUI...")
        use_gui = True
    
    # Setup video path early if video recording is requested
    video_path = None
    if args.video:
        # Create videos directory in current working directory
        # Note: Path.cwd() is the directory where the script is RUN from, not where it's located
        videos_dir = Path.cwd() / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        # Determine video filename
        video_name = args.video
        # If no extension provided, add .mp4
        if not video_name.endswith(('.mp4', '.avi', '.mov', '.gif')):
            video_name += '.mp4'
        
        # If path doesn't contain directory separators, save to videos folder
        if '/' not in video_name and '\\' not in video_name:
            video_path = videos_dir / video_name
        else:
            # User provided a path - check if it's absolute or relative
            video_path = Path(video_name)
            if not video_path.is_absolute():
                # Relative path - save to videos folder
                video_path = videos_dir / video_path.name
        
        # Print the full path early so user knows where it will be saved
        print(f"\n{'='*60}")
        print(f"VIDEO RECORDING")
        print(f"{'='*60}")
        print(f"Video will be saved to:")
        print(f"  Full path: {video_path.absolute()}")
        print(f"  Relative:  {video_path}")
        print(f"  Directory: {videos_dir.absolute()}")
        print(f"  Filename:  {video_path.name}")
        print(f"\nCurrent working directory: {Path.cwd()}")
        print(f"{'='*60}\n")
    
    print("=" * 60)
    print("Testing Upkie Pendulum Policy")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Environment: {ENV_ID}")
    print(f"Number of episodes: {args.episodes}")
    print(f"Mode: {'GUI (visualization)' if use_gui else 'Headless (fast)'}")
    print(f"Timing: {'Real-time' if args.realtime else 'As fast as possible'}")
    if args.video and video_path:
        print(f"Video export: {video_path}")
    print("=" * 60)

    # Create environment configuration
    ENV_KWARGS = dict(
        frequency=200.0,
        gui=use_gui,
        regulate_frequency=args.realtime,  # Real-time if requested, otherwise fast
    )

    # Create environment directly (avoid render_mode issue with make_vec_env)
    # Note: Upkie environments don't support render_mode parameter
    def make_env():
        try:
            env = gym.make(ENV_ID, **ENV_KWARGS)
            return env
        except Exception as e:
            if "GUI" in str(e) or "connection" in str(e).lower():
                # Fall back to headless mode if GUI connection fails
                print(f"\nWarning: Could not create GUI environment: {e}")
                print("Falling back to headless (DIRECT) mode...")
                env_kwargs_no_gui = ENV_KWARGS.copy()
                env_kwargs_no_gui['gui'] = False
                env = gym.make(ENV_ID, **env_kwargs_no_gui)
                return env
            else:
                raise
    
    env = DummyVecEnv([make_env])
    env.seed(SEED)

    # Load trained model
    print(f"\nLoading model from {model_path}...")
    model = PPO.load(str(model_path), env=env)

    # Validate video recording setup if requested
    if args.video:
        if not HAS_IMAGEIO:
            print("Error: imageio is required for video export. Install with: pip install imageio")
            sys.exit(1)
        if not use_gui:
            print("Error: Video export requires GUI mode. Use --gui or remove --fast")
            sys.exit(1)
        if not HAS_PYBULLET:
            print("Error: pybullet is required for video export")
            sys.exit(1)
        
        print(f"\nRecording video to: {video_path}")
        print("Note: Video will be recorded at real-time speed (200 Hz)")
    
    # Get PyBullet connection ID for video capture if needed
    bullet_client_id = None
    if args.video and use_gui:
        # The environment structure: DummyVecEnv -> envs[0] -> UpkiePendulum -> env -> UpkieEnv -> backend -> PyBulletBackend
        try:
            # Get the wrapped environment from DummyVecEnv
            if hasattr(env, 'envs') and len(env.envs) > 0:
                pendulum_wrapper = env.envs[0]  # This is UpkiePendulum (gym.Wrapper)
                # UpkiePendulum wraps the base UpkieEnv
                if hasattr(pendulum_wrapper, 'env'):
                    base_env = pendulum_wrapper.env  # This is UpkieEnv
                    # UpkieEnv has the backend
                    if hasattr(base_env, 'backend'):
                        backend = base_env.backend
                        # PyBulletBackend has _bullet attribute (the client ID)
                        if hasattr(backend, '_bullet'):
                            bullet_client_id = backend._bullet
                            print(f"✓ Found PyBullet client ID: {bullet_client_id}")
                        else:
                            print(f"Warning: Backend does not have _bullet attribute. Type: {type(backend)}")
                            # Debug: print backend attributes
                            print(f"  Backend attributes: {[attr for attr in dir(backend) if not attr.startswith('__')]}")
                    else:
                        print(f"Warning: Base environment does not have backend attribute")
                        print(f"  Base env type: {type(base_env)}")
                        print(f"  Base env attributes: {[attr for attr in dir(base_env) if not attr.startswith('__')]}")
                else:
                    print(f"Warning: Pendulum wrapper does not have 'env' attribute")
                    print(f"  Wrapper type: {type(pendulum_wrapper)}")
            else:
                print(f"Warning: DummyVecEnv does not have envs or envs is empty")
        except Exception as e:
            print(f"Warning: Exception accessing PyBullet client ID: {e}")
            import traceback
            traceback.print_exc()
        
        # Fallback: If we're in GUI mode and couldn't get the client ID from backend,
        # try using the default client ID (0) which is typically used for GUI connections
        if bullet_client_id is None and use_gui:
            try:
                # Test if client 0 is active by trying a simple query
                test_result = pybullet.getNumBodies(physicsClientId=0)
                bullet_client_id = 0
                print(f"✓ Using default PyBullet client ID (0) - verified active")
            except Exception as e:
                print(f"Warning: Default client ID (0) is not active: {e}")
        
        if bullet_client_id is None:
            print("✗ Error: Could not find PyBullet client ID. Video recording disabled.")
    
    # Initialize video frames list if we have a valid bullet client
    video_frames = [] if (args.video and bullet_client_id is not None) else None
    
    # Frame capture settings for video (only capture at video FPS, not simulation FPS)
    video_fps = 30.0  # Target FPS for video
    sim_frequency = 200.0  # Simulation frequency (Hz)
    frame_skip = max(1, int(sim_frequency / video_fps))  # Capture every Nth frame
    frame_counter = 0  # Counter for frame skipping
    
    # Run episodes
    print("\nRunning episodes...\n")
    episode_returns = []
    episode_lengths = []

    for episode in range(1, args.episodes + 1):
        # DummyVecEnv.reset() returns (obs, info) tuple in Gymnasium API
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
        else:
            # Fallback: if reset only returns obs, create empty info
            obs = reset_result
            info = {}
        
        ep_return = 0.0
        ep_len = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # DummyVecEnv.step() returns (obs, reward, terminated, truncated, info)
            step_result = env.step(action)
            if isinstance(step_result, tuple) and len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
            else:
                # Fallback for older API versions
                obs, reward, done_vec, info = step_result[:4]
                terminated = done_vec
                truncated = [False] * len(done_vec) if hasattr(done_vec, '__len__') else [False]
            
            # Capture frame for video if recording (only every Nth frame to match video FPS)
            if args.video and bullet_client_id is not None and video_frames is not None:
                frame_counter += 1
                # Only capture frame if we've skipped enough frames
                if frame_counter >= frame_skip:
                    frame_counter = 0  # Reset counter
                    try:
                        # Get camera image from PyBullet
                        # Parameters: width, height, viewMatrix, projectionMatrix
                        width, height = 640, 480
                        view_matrix = pybullet.computeViewMatrix(
                            cameraEyePosition=[0, -2, 1],
                            cameraTargetPosition=[0, 0, 0.6],
                            cameraUpVector=[0, 0, 1],
                            physicsClientId=bullet_client_id
                        )
                        projection_matrix = pybullet.computeProjectionMatrixFOV(
                            fov=60, aspect=width/height, nearVal=0.1, farVal=10.0,
                            physicsClientId=bullet_client_id
                        )
                        _, _, rgb, _, _ = pybullet.getCameraImage(
                            width, height,
                            viewMatrix=view_matrix,
                            projectionMatrix=projection_matrix,
                            physicsClientId=bullet_client_id
                        )
                        # Convert RGB array to uint8 image
                        frame = np.array(rgb, dtype=np.uint8)
                        frame = frame.reshape((height, width, 4))[:, :, :3]  # Remove alpha channel
                        video_frames.append(frame)
                    except Exception as e:
                        if ep_len == 0 and episode == 1:  # Only warn once at start
                            print(f"Warning: Could not capture frame: {e}")
                            print("This may indicate a PyBullet connection issue.")
            
            ep_return += float(reward[0])
            ep_len += 1
            done = bool(terminated[0] or truncated[0])

        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        print(f"Episode {episode}: return={ep_return:.3f}, length={ep_len} steps")
        
        # Reset frame counter for next episode
        if args.video:
            frame_counter = 0

    # Print statistics
    print("\n" + "=" * 60)
    print("Evaluation Statistics")
    print("=" * 60)
    print(f"Mean return: {sum(episode_returns) / len(episode_returns):.3f}")
    print(f"Std return: {(sum((x - sum(episode_returns)/len(episode_returns))**2 for x in episode_returns) / len(episode_returns))**0.5:.3f}")
    print(f"Mean length: {sum(episode_lengths) / len(episode_lengths):.1f} steps")
    print(f"Std length: {(sum((x - sum(episode_lengths)/len(episode_lengths))**2 for x in episode_lengths) / len(episode_lengths))**0.5:.1f} steps")
    print("=" * 60)
    
    # Save video if frames were captured
    if args.video:
        if video_frames is None:
            print("\nWarning: Video recording was requested but video_frames is None.")
            print("This may indicate that PyBullet client ID was not found.")
        elif len(video_frames) == 0:
            print("\nWarning: No video frames were captured.")
            print("This may indicate that frame capture failed or episodes were too short.")
        else:
            print(f"\nSaving video with {len(video_frames)} frames to {video_path}...")
            try:
                # Use the target video FPS we set earlier
                fps = video_fps  # Use 30 fps for smooth playback
                imageio.mimwrite(str(video_path), video_frames, fps=fps, codec='libx264', quality=8)
                
                # Verify the file was actually created
                if video_path.exists():
                    file_size = video_path.stat().st_size / (1024 * 1024)  # Size in MB
                    print(f"\n{'='*60}")
                    print(f"✓ VIDEO SAVED SUCCESSFULLY!")
                    print(f"{'='*60}")
                    print(f"File: {video_path.name}")
                    print(f"Location: {video_path.absolute()}")
                    print(f"Size: {file_size:.2f} MB")
                    print(f"Frames: {len(video_frames)}")
                    print(f"FPS: {fps}")
                    print(f"{'='*60}\n")
                else:
                    print(f"✗ Warning: Video file was not created at {video_path.absolute()}")
            except Exception as e:
                print(f"Error saving video: {e}")
                print("Trying alternative codec...")
                try:
                    imageio.mimwrite(str(video_path), video_frames, fps=fps)
                    
                    # Verify the file was actually created
                    if video_path.exists():
                        file_size = video_path.stat().st_size / (1024 * 1024)  # Size in MB
                        print(f"\n{'='*60}")
                        print(f"✓ VIDEO SAVED WITH DEFAULT CODEC!")
                        print(f"{'='*60}")
                        print(f"File: {video_path.name}")
                        print(f"Location: {video_path.absolute()}")
                        print(f"Size: {file_size:.2f} MB")
                        print(f"Frames: {len(video_frames)}")
                        print(f"FPS: {fps}")
                        print(f"{'='*60}\n")
                    else:
                        print(f"✗ Warning: Video file was not created at {video_path.absolute()}")
                except Exception as e2:
                    print(f"✗ Failed to save video: {e2}")
                    print(f"  Tried to save {len(video_frames)} frames to {video_path.absolute()}")
                    import traceback
                    traceback.print_exc()

    env.close()


if __name__ == "__main__":
    main()
