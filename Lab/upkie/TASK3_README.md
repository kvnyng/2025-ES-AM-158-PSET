# Task 3: Train a Stabilizing Policy on the Full Model (Upkie-Servos)

This directory contains the implementation for training a stabilizing policy on the full Upkie model with 6 servos.

## Files Created

1. **`upkie/envs/servos_reward_wrapper.py`**: Reward and termination wrapper for the Upkie-Servos environment
   - Adds reward shaping based on balance state (pitch, velocities, actions)
   - Implements termination conditions (fall detection, position drift, velocity limits)
   - Similar structure to `UpkiePendulum` wrapper

2. **`train_servos.py`**: Training script for Upkie-Servos using PPO
   - Supports training from scratch or resuming from checkpoints
   - Uses parallel environments (DummyVecEnv) for efficient training
   - Includes VecNormalize for observation and reward normalization
   - Saves checkpoints periodically and final model with timestamp

3. **Updated `rollout_policy_servos.py`**: Added reward wrapper integration
   - Now includes `ServosRewardWrapper` in the environment chain
   - Maintains compatibility with existing action/observation wrappers

## Environment Structure

The wrapped environment chain is:
1. Base `UpkieServos` environment (from upkie package)
2. `ServosRewardWrapper` - adds rewards and termination conditions
3. `ServoVelActionWrapper` - converts Box actions to Dict servo commands
4. `ServoObsFlattenWrapper` - flattens Dict observations to Box
5. `TimeLimit` - limits episode length

## Reward Function

The reward encourages stable balancing:

```
reward = 1.0 - (
    w_pitch * pitch² +
    w_pitch_velocity * pitch_velocity² +
    w_linear_velocity * linear_velocity² +
    w_action * action_magnitude² +
    smoothness_penalty
)
```

Default weights:
- `w_pitch = 0.5`
- `w_pitch_velocity = 0.1`
- `w_linear_velocity = 0.1`
- `w_action = 0.05`
- Smoothness penalty: `0.01 * (action_change² + 0.5 * action_accel²)`

## Termination Conditions

The environment terminates when:
1. **Fall detected**: `|pitch| > fall_pitch` (default: 1.0 rad)
2. **Position drift**: `|position - initial_position| > max_position_drift` (default: 5.0 m)
3. **Angular velocity**: `|pitch_velocity| > max_angular_velocity` (default: 10.0 rad/s)
4. **Linear velocity**: `|linear_velocity| > max_linear_velocity` (default: 2.0 m/s)

## Training

### Basic Training

```bash
cd upkie
python train_servos.py
```

### Training with Custom Parameters

```bash
python train_servos.py \
    --total-timesteps 2000000 \
    --learning-rate 2e-4 \
    --batch-size 128 \
    --n-epochs 5 \
    --n-envs 8
```

### Resuming from Checkpoint

```bash
python train_servos.py --resume latest
# or
python train_servos.py --resume ./models/checkpoints/ppo_servos_XXXXXX_steps.zip
```

## Evaluation

After training, evaluate the policy:

```bash
python rollout_policy_servos.py \
    --model ./models/ppo_servos_YYYYMMDD_HHMMSS.zip \
    --episodes 5 \
    --deterministic
```

## Model Outputs

- **Checkpoints**: Saved to `./models/checkpoints/ppo_servos_XXXXXX_steps.zip`
- **Final model**: Saved to `./models/ppo_servos_YYYYMMDD_HHMMSS.zip`
- **Normalization stats**: Saved to `./models/ppo_servos_norm_stats.pkl`
- **TensorBoard logs**: `./logs/tensorboard/`

## Training Configuration

Default hyperparameters:
- **Learning rate**: 2e-4 (with linear decay schedule)
- **Batch size**: 128
- **N epochs**: 5
- **Gamma**: 0.99
- **GAE lambda**: 0.95
- **Clip range**: 0.2
- **VF coefficient**: 0.5
- **Network architecture**: [128, 128] (two hidden layers)
- **Parallel environments**: 8
- **Total timesteps**: 2,000,000

## Notes

- Training can be tricky - the policy may exhibit diverse stabilization patterns
- The environment uses PyBullet for faster training (can be switched to Spine for real robot)
- VecNormalize is critical for stable training - it normalizes both observations and rewards
- Checkpoints are saved every 50,000 steps for recovery and analysis
- The reward wrapper extracts state from spine observations, so it must be applied before action/observation wrappers

## Differences from Pendulum Task

1. **Action space**: 6D (wheels: velocity commands, legs: position commands) vs 1D (ground velocity)
2. **Observation space**: 12D (6 joints × 2: position + velocity) vs 4D (pitch, position, velocities)
3. **Control**: Direct servo control with PID gain scaling vs simplified wheel velocity control
4. **Complexity**: Full 6-DOF robot vs simplified inverted pendulum model

