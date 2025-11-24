#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter sweep script for Upkie Pendulum training.

This script runs multiple training jobs with different hyperparameter configurations
and tracks the results for comparison.
"""

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def run_training(config: Dict, run_id: int, total_runs: int, base_dir: Path):
    """Run a single training job with given configuration."""
    print("\n" + "=" * 80)
    print(f"Run {run_id}/{total_runs}: {config.get('name', 'unnamed')}")
    print("=" * 80)
    print("Configuration:")
    for key, value in config.items():
        if key != "name":
            print(f"  {key}: {value}")
    print("=" * 80)
    
    # Build command
    cmd = [sys.executable, "train_pendulum.py"]
    
    # Add arguments based on config
    if "total_timesteps" in config:
        cmd.extend(["--total-timesteps", str(config["total_timesteps"])])
    if "learning_rate" in config:
        cmd.extend(["--learning-rate", str(config["learning_rate"])])
    if "batch_size" in config:
        cmd.extend(["--batch-size", str(config["batch_size"])])
    if "n_epochs" in config:
        cmd.extend(["--n-epochs", str(config["n_epochs"])])
    if "vf_coef" in config:
        cmd.extend(["--vf-coef", str(config["vf_coef"])])
    if "clip_range" in config:
        cmd.extend(["--clip-range", str(config["clip_range"])])
    if "n_envs" in config:
        cmd.extend(["--n-envs", str(config["n_envs"])])
    
    try:
        result = subprocess.run(
            cmd,
            cwd=base_dir,
            check=True,
            capture_output=False,  # Show output in real-time
        )
        return True, None
    except subprocess.CalledProcessError as e:
        return False, str(e)


def create_hyperparameter_configs(sweep_config: Dict) -> List[Dict]:
    """Generate all hyperparameter configurations from sweep definition."""
    configs = []
    
    # Extract parameter ranges
    param_names = []
    param_values = []
    
    for param_name, param_range in sweep_config.items():
        if param_name == "fixed":  # Fixed parameters for all runs
            continue
        param_names.append(param_name)
        if isinstance(param_range, (list, tuple)):
            param_values.append(param_range)
        elif isinstance(param_range, dict):
            # Support range definitions like {"start": 0.1, "end": 0.5, "num": 5}
            if "start" in param_range and "end" in param_range:
                num = param_range.get("num", 5)
                param_values.append(
                    np.linspace(param_range["start"], param_range["end"], num).tolist()
                )
            else:
                param_values.append([param_range])
        else:
            param_values.append([param_range])
    
    # Generate all combinations
    for i, combination in enumerate(itertools.product(*param_values)):
        config = dict(zip(param_names, combination))
        config["name"] = f"run_{i+1:03d}"
        if "fixed" in sweep_config:
            config.update(sweep_config["fixed"])
        configs.append(config)
    
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweep for Upkie Pendulum training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON/YAML config file (not implemented, using hardcoded config for now)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Number of random configurations to try (for random search)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./sweep_results",
        help="Directory to save sweep results",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted sweep (skip completed runs)",
    )
    args = parser.parse_args()
    
    # Define hyperparameter sweep configuration
    # You can modify these ranges based on what you want to explore
    # 
    # The script performs a grid search over all combinations of the parameters listed below.
    # For example, if you have 3 learning rates, 3 batch sizes, 3 n_epochs, and 3 vf_coef values,
    # you'll get 3 * 3 * 3 * 3 = 81 total configurations.
    #
    # To reduce the number of runs, use --runs N to randomly sample N configurations.
    sweep_config = {
        # Fixed parameters (same for all runs)
        "fixed": {
            "total_timesteps": 1_000_000,
            "n_envs": 8,
        },
        # Parameters to sweep (grid search)
        # All combinations of these will be tested
        "learning_rate": [1e-4, 2e-4, 3e-4],  # Learning rates to try
        "batch_size": [64, 128, 256],  # Batch sizes to try
        "n_epochs": [5, 10, 15],  # Epochs per update
        "vf_coef": [0.5, 1.0, 1.5],  # Value function coefficient
        # "clip_range": [0.1, 0.2, 0.3],  # PPO clip range (commented out to reduce combinations)
    }
    
    # Generate all configurations
    configs = create_hyperparameter_configs(sweep_config)
    
    # Limit to random subset if requested
    if args.runs and args.runs < len(configs):
        print(f"Randomly selecting {args.runs} configurations from {len(configs)} total...")
        np.random.seed(42)
        configs = np.random.choice(configs, size=args.runs, replace=False).tolist()
    
    print(f"\nTotal configurations to run: {len(configs)}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save sweep configuration
    sweep_info = {
        "timestamp": datetime.now().isoformat(),
        "total_configs": len(configs),
        "sweep_config": sweep_config,
    }
    
    with open(output_dir / "sweep_info.json", "w") as f:
        json.dump(sweep_info, f, indent=2)
    
    # Load existing results if resuming
    results = []
    results_file = output_dir / "results_summary.json"
    completed_runs = {}
    if args.resume and results_file.exists():
        with open(results_file, "r") as f:
            existing_results = json.load(f)
            # Create a map of completed runs
            completed_runs = {r["run_id"]: r for r in existing_results if r.get("status") == "completed"}
            print(f"Resuming sweep: {len(completed_runs)} runs already completed")
            results = existing_results
    
    # Run each configuration
    base_dir = Path(__file__).parent
    
    for i, config in enumerate(configs, 1):
        run_id = config["name"]
        
        # Skip if already completed and resuming
        if args.resume and run_id in completed_runs:
            print(f"\nSkipping {run_id} (already completed)")
            continue
        
        print(f"\n{'='*80}")
        print(f"Starting run {i}/{len(configs)}: {run_id}")
        print(f"{'='*80}")
        
        # Save config for this run
        run_dir = output_dir / run_id
        run_dir.mkdir(exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Run training with this configuration
        success, error = run_training(config, i, len(configs), base_dir)
        
        result = {
            "run_id": run_id,
            "config": config,
            "status": "completed" if success else "failed",
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Update or append result
        existing_idx = next((idx for idx, r in enumerate(results) if r["run_id"] == run_id), None)
        if existing_idx is not None:
            results[existing_idx] = result
        else:
            results.append(result)
        
        # Save individual result
        with open(run_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)
        
        # Save updated summary after each run
        with open(output_dir / "results_summary.json", "w") as f:
            json.dump(results, f, indent=2)
        
        if success:
            print(f"✓ Run {i}/{len(configs)} ({run_id}) completed successfully")
        else:
            print(f"✗ Run {i}/{len(configs)} ({run_id}) failed: {error}")
    
    # Print final summary
    completed = sum(1 for r in results if r.get("status") == "completed")
    failed = sum(1 for r in results if r.get("status") == "failed")
    pending = sum(1 for r in results if r.get("status") == "pending")
    
    print(f"\n{'='*80}")
    print(f"Sweep complete!")
    print(f"{'='*80}")
    print(f"Total runs: {len(results)}")
    print(f"  Completed: {completed}")
    print(f"  Failed: {failed}")
    print(f"  Pending: {pending}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Individual configs: {output_dir}/run_XXX/config.json")
    print(f"  - Individual results: {output_dir}/run_XXX/result.json")
    print(f"  - Summary: {output_dir}/results_summary.json")
    print(f"  - Sweep info: {output_dir}/sweep_info.json")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

