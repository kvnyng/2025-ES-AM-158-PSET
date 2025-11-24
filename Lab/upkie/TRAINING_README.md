# Training Upkie Pendulum Policy

This directory contains scripts for training and testing a policy to stabilize the Upkie robot in the pendulum configuration.

## Files

- `train_pendulum.py`: Training script using PPO
- `rollout_policy.py`: Script to test/evaluate a trained policy

## Training

To train a new policy:

```bash
cd upkie
python train_pendulum.py
```

This will:
- Train a PPO policy for 1,000,000 timesteps
- Use 4 parallel PyBullet environments
- Save the best model to `./models/pendulum_best/best_model.zip`
- Save periodic checkpoints to `./models/checkpoints/`
- Log training progress to TensorBoard in `./logs/tensorboard/`

Monitor training progress:
```bash
tensorboard --logdir ./logs/tensorboard/
```

### Plotting Training Metrics

To create plots of training loss and other metrics:

```bash
# Plot metrics from the latest training run
python plot_training_metrics.py

# Plot metrics from a specific run
python plot_training_metrics.py --run PPO_1

# Save plot to file
python plot_training_metrics.py --output training_metrics.png

# Export metrics to CSV for custom analysis
python plot_training_metrics.py --export-csv training_metrics.csv

# Watch training in real-time (updates every 5 seconds)
python plot_training_metrics.py --watch

# Watch with custom refresh interval (e.g., every 2 seconds)
python plot_training_metrics.py --watch --refresh 2.0
```

The script plots:
- Policy loss
- Value loss
- Total loss
- Learning rate
- Explained variance
- Approximate KL divergence
- Clip fraction
- Entropy loss (if applicable)

## Testing

To test a trained policy:

```bash
python rollout_policy.py
```

This will:
- Load the best model from `./models/pendulum_best/best_model.zip`
- Run 5 evaluation episodes
- Display statistics (mean return, episode lengths, etc.)
- Show the PyBullet GUI for visualization

## Requirements

- stable-baselines3
- upkie (with PyBullet backend)
- pybullet

## Notes

- Training takes approximately 1-2 hours depending on hardware
- The policy is trained in simulation (PyBullet) but can be tested on real hardware by changing `ENV_ID` in `rollout_policy.py` to `"Upkie-Spine-Pendulum"` (requires a running spine)

