#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract and plot training metrics from TensorBoard logs.

This script reads TensorBoard event files and creates plots of:
- Policy loss
- Value loss
- Total loss
- Learning rate
- Explained variance
- And other training metrics
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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


def plot_training_metrics(log_dir: Path, output_path: Path = None, run_name: str = None, 
                          watch: bool = False, refresh_interval: float = 5.0):
    """Plot training metrics from TensorBoard logs.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        output_path: Path to save plot (None to display)
        run_name: Specific run to plot (None for latest)
        watch: If True, continuously update plot while training
        refresh_interval: Seconds between updates when watching
    """
    
    if watch:
        # Continuous monitoring mode
        import time
        print("Watching training progress (press Ctrl+C to stop)...")
        while True:
            try:
                _plot_single_update(log_dir, output_path, run_name, watch=True)
                time.sleep(refresh_interval)
            except KeyboardInterrupt:
                print("\nStopped watching.")
                break
    else:
        # Single plot
        _plot_single_update(log_dir, output_path, run_name, watch=False)


def _plot_single_update(log_dir: Path, output_path: Path = None, run_name: str = None, 
                        watch: bool = False):
    """Plot a single update of training metrics."""
    
    # Find the run to plot
    if run_name:
        run_dir = log_dir / run_name
        event_files = list(run_dir.glob("events.out.tfevents.*"))
        if not event_files:
            if not watch:
                print(f"Error: No event files found in {run_dir}")
            return
        event_file = max(event_files, key=lambda p: p.stat().st_mtime)
    else:
        result = find_latest_run(log_dir)
        if result is None:
            if not watch:
                print(f"Error: No training runs found in {log_dir}")
            return
        run_dir, event_file = result
        if not watch:
            print(f"Using latest run: {run_dir.name}")
    
    if not watch:
        print(f"Reading event file: {event_file}")
    
    # Load event accumulator to see what metrics are available
    ea = EventAccumulator(str(event_file))
    ea.Reload()
    
    available_scalars = ea.Tags()["scalars"]
    if not watch:
        print(f"\nAvailable metrics: {sorted(available_scalars)}")
    
    # Define metrics to plot
    metrics_to_plot = {
        "train/policy_loss": "Policy Loss",
        "train/value_loss": "Value Loss",
        "train/loss": "Total Loss",
        "train/learning_rate": "Learning Rate",
        "train/explained_variance": "Explained Variance",
        "train/approx_kl": "Approximate KL Divergence",
        "train/clip_fraction": "Clip Fraction",
        "train/entropy_loss": "Entropy Loss",
    }
    
    # Filter to only plot metrics that exist
    metrics_to_plot = {
        k: v for k, v in metrics_to_plot.items() 
        if k in available_scalars
    }
    
    if not metrics_to_plot:
        print("Error: No expected metrics found in event file")
        return
    
    # Create subplots
    n_metrics = len(metrics_to_plot)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Use interactive mode for watching
    if watch:
        plt.ion()  # Turn on interactive mode
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each metric
    for idx, (metric_key, metric_label) in enumerate(metrics_to_plot.items()):
        steps, values = extract_scalars(event_file, metric_key)
        if steps is not None and len(steps) > 0:
            ax = axes[idx]
            ax.clear()  # Clear previous plot when watching
            ax.plot(steps, values, linewidth=1.5)
            ax.set_xlabel("Training Steps")
            ax.set_ylabel(metric_label)
            ax.set_title(metric_label)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(values)
            final_val = values[-1]
            min_val = np.min(values)
            max_val = np.max(values)
            
            stats_text = f"Mean: {mean_val:.4f}\nFinal: {final_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}"
            if watch:
                stats_text += f"\nSteps: {int(steps[-1]):,}"
            
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)
        else:
            axes[idx].clear()
            axes[idx].text(0.5, 0.5, f"No data for\n{metric_label}",
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(metric_label)
    
    # Hide unused subplots
    for idx in range(len(metrics_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if not watch:
            print(f"\nPlot saved to: {output_path}")
    elif watch:
        plt.draw()
        plt.pause(0.1)  # Brief pause for update
    else:
        plt.show()


def export_metrics_to_csv(log_dir: Path, output_csv: Path, run_name: str = None):
    """Export training metrics to CSV for further analysis."""
    
    # Find the run
    if run_name:
        run_dir = log_dir / run_name
        event_files = list(run_dir.glob("events.out.tfevents.*"))
        if not event_files:
            print(f"Error: No event files found in {run_dir}")
            return
        event_file = max(event_files, key=lambda p: p.stat().st_mtime)
    else:
        result = find_latest_run(log_dir)
        if result is None:
            print(f"Error: No training runs found in {log_dir}")
            return
        run_dir, event_file = result
    
    # Load event accumulator
    ea = EventAccumulator(str(event_file))
    ea.Reload()
    
    available_scalars = ea.Tags()["scalars"]
    
    # Extract all metrics
    import csv
    all_data = {}
    
    for scalar_name in available_scalars:
        steps, values = extract_scalars(event_file, scalar_name)
        if steps is not None:
            all_data[scalar_name] = (steps, values)
    
    # Find common step range
    if not all_data:
        print("Error: No metrics found")
        return
    
    # Write to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ["step"] + list(all_data.keys())
        writer.writerow(header)
        
        # Get all unique steps
        all_steps = set()
        for steps, _ in all_data.values():
            all_steps.update(steps)
        all_steps = sorted(all_steps)
        
        # Write data rows
        for step in all_steps:
            row = [step]
            for scalar_name in all_data.keys():
                steps, values = all_data[scalar_name]
                if step in steps:
                    idx = np.where(steps == step)[0][0]
                    row.append(values[idx])
                else:
                    row.append("")  # Missing value
            writer.writerow(row)
    
    print(f"Metrics exported to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training metrics from TensorBoard logs"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/tensorboard/",
        help="Directory containing TensorBoard logs (default: ./logs/tensorboard/)",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Specific run to plot (e.g., PPO_1). If not specified, uses latest run.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for plot (e.g., training_metrics.png). If not specified, displays plot.",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Export metrics to CSV file (e.g., training_metrics.csv)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously update plot while training (real-time monitoring)",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=5.0,
        help="Refresh interval in seconds when watching (default: 5.0)",
    )
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    
    if args.export_csv:
        export_metrics_to_csv(log_dir, Path(args.export_csv), args.run)
    
    plot_training_metrics(
        log_dir, 
        Path(args.output) if args.output else None, 
        args.run,
        watch=args.watch,
        refresh_interval=args.refresh
    )


if __name__ == "__main__":
    main()

