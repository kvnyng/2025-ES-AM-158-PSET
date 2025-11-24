#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create graphs for loss and KL divergence with rolling smoothing.

This script reads TensorBoard event files and creates smoothed plots of:
- Loss (total loss)
- KL Divergence (approximate KL divergence)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def rolling_smooth(values, window_size=50):
    """
    Apply rolling average smoothing to a 1D array.
    
    Args:
        values: 1D numpy array of values to smooth
        window_size: Size of the rolling window (default: 50)
    
    Returns:
        Smoothed array with the same length as input
    """
    if len(values) == 0:
        return values
    
    # Use a smaller window if we don't have enough data points
    window_size = min(window_size, len(values))
    
    # Pad the beginning with the first value to handle edge cases
    padded = np.concatenate([[values[0]] * (window_size - 1), values])
    
    # Apply rolling average
    smoothed = np.convolve(padded, np.ones(window_size) / window_size, mode='valid')
    
    return smoothed


def extract_scalars(event_file_path: Path, scalar_name: str):
    """Extract scalar values from a TensorBoard event file."""
    ea = EventAccumulator(str(event_file_path))
    ea.Reload()
    
    if scalar_name not in ea.Tags()["scalars"]:
        return None, None
    
    scalar_events = ea.Scalars(scalar_name)
    steps = [s.step for s in scalar_events]
    values = [s.value for s in scalar_events]
    
    return np.array(steps), np.array(values)


def find_latest_run(log_dir: Path):
    """Find the most recent training run in the log directory."""
    if not log_dir.exists():
        return None
    
    # Find all subdirectories (PPO_1, PPO_2, etc.)
    runs = []
    for subdir in log_dir.iterdir():
        if subdir.is_dir():
            # Find event files in this directory
            event_files = list(subdir.glob("events.out.tfevents.*"))
            if event_files:
                # Use the most recent event file
                latest_event = max(event_files, key=lambda p: p.stat().st_mtime)
                runs.append((subdir, latest_event))
    
    if not runs:
        return None
    
    # Return the run with the most recent event file
    latest_run = max(runs, key=lambda x: x[1].stat().st_mtime)
    return latest_run[0], latest_run[1]


def plot_loss_and_kl(log_dir: Path, output_dir: Path, run_name: str = None, 
                     window_size: int = 50):
    """
    Plot loss and KL divergence with rolling smoothing.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        output_dir: Directory to save graphs
        run_name: Specific run to plot (None for latest)
        window_size: Size of rolling window for smoothing
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the run to plot
    if run_name:
        run_dir = log_dir / run_name
        event_files = list(run_dir.glob("events.out.tfevents.*"))
        if not event_files:
            print(f"Error: No event files found in {run_dir}")
            return
        event_file = max(event_files, key=lambda p: p.stat().st_mtime)
        run_identifier = run_name
    else:
        result = find_latest_run(log_dir)
        if result is None:
            print(f"Error: No training runs found in {log_dir}")
            return
        run_dir, event_file = result
        run_identifier = run_dir.name
        print(f"Using latest run: {run_identifier}")
    
    print(f"Reading event file: {event_file}")
    
    # Load event accumulator to see what metrics are available
    ea = EventAccumulator(str(event_file))
    ea.Reload()
    
    available_scalars = ea.Tags()["scalars"]
    print(f"\nAvailable metrics: {sorted(available_scalars)}")
    
    # Try different possible names for loss and KL divergence
    loss_names = ["train/loss", "train/total_loss", "loss_curve/total_loss"]
    kl_names = ["train/approx_kl", "train/kl", "train/kl_divergence"]
    
    # Find loss metric
    loss_steps = None
    loss_values = None
    loss_name = None
    for name in loss_names:
        if name in available_scalars:
            loss_steps, loss_values = extract_scalars(event_file, name)
            if loss_steps is not None and len(loss_steps) > 0:
                loss_name = name
                break
    
    # Find KL divergence metric
    kl_steps = None
    kl_values = None
    kl_name = None
    for name in kl_names:
        if name in available_scalars:
            kl_steps, kl_values = extract_scalars(event_file, name)
            if kl_steps is not None and len(kl_steps) > 0:
                kl_name = name
                break
    
    # Plot Loss
    if loss_steps is not None and len(loss_steps) > 0:
        print(f"\nPlotting Loss ({loss_name})...")
        print(f"  Data points: {len(loss_steps)}")
        print(f"  Steps range: {loss_steps[0]} - {loss_steps[-1]}")
        
        # Apply smoothing
        loss_smoothed = rolling_smooth(loss_values, window_size)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot raw and smoothed data
        ax.plot(loss_steps, loss_values, alpha=0.3, color='blue', 
                label='Raw', linewidth=0.5)
        ax.plot(loss_steps, loss_smoothed, color='blue', 
                label=f'Smoothed (window={window_size})', linewidth=2)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Loss Over Training Steps\n({run_identifier})', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_loss = np.mean(loss_values)
        final_loss = loss_values[-1]
        min_loss = np.min(loss_values)
        max_loss = np.max(loss_values)
        
        stats_text = (f"Mean: {mean_loss:.4f}\n"
                     f"Final: {final_loss:.4f}\n"
                     f"Min: {min_loss:.4f}\n"
                     f"Max: {max_loss:.4f}")
        
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9)
        
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"loss_{run_identifier}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {output_path}")
        plt.close()
    else:
        print(f"\nWarning: Loss metric not found. Tried: {loss_names}")
    
    # Plot KL Divergence
    if kl_steps is not None and len(kl_steps) > 0:
        print(f"\nPlotting KL Divergence ({kl_name})...")
        print(f"  Data points: {len(kl_steps)}")
        print(f"  Steps range: {kl_steps[0]} - {kl_steps[-1]}")
        
        # Apply smoothing
        kl_smoothed = rolling_smooth(kl_values, window_size)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot raw and smoothed data
        ax.plot(kl_steps, kl_values, alpha=0.3, color='red', 
                label='Raw', linewidth=0.5)
        ax.plot(kl_steps, kl_smoothed, color='red', 
                label=f'Smoothed (window={window_size})', linewidth=2)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('KL Divergence', fontsize=12)
        ax.set_title(f'KL Divergence Over Training Steps\n({run_identifier})', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_kl = np.mean(kl_values)
        final_kl = kl_values[-1]
        min_kl = np.min(kl_values)
        max_kl = np.max(kl_values)
        
        stats_text = (f"Mean: {mean_kl:.4f}\n"
                     f"Final: {final_kl:.4f}\n"
                     f"Min: {min_kl:.4f}\n"
                     f"Max: {max_kl:.4f}")
        
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9)
        
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"kl_divergence_{run_identifier}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {output_path}")
        plt.close()
    else:
        print(f"\nWarning: KL Divergence metric not found. Tried: {kl_names}")
    
    print(f"\nGraphs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot loss and KL divergence with rolling smoothing"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/tensorboard/",
        help="Directory containing TensorBoard logs (default: ./logs/tensorboard/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./graphs/",
        help="Directory to save graphs (default: ./graphs/)",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Specific run to plot (e.g., PPO_1). If not specified, uses latest run.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=50,
        help="Size of rolling window for smoothing (default: 50)",
    )
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    
    plot_loss_and_kl(log_dir, output_dir, args.run, args.window_size)


if __name__ == "__main__":
    main()

