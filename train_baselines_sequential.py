#!/usr/bin/env python3
"""
Train All Baseline Models Sequentially

Trains each baseline combination (2 modalities × 3 environments) sequentially
to ensure all models complete properly with proper WandB logging.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_training(env_name, obs_type, mode="standard"):
    """Run training for a specific environment and observation type."""
    print(f"\n{'='*70}")
    print(f"TRAINING: {env_name.upper()} + {obs_type.upper()}")
    print(f"{'='*70}")

    cmd = [
        sys.executable, "train_all_baselines.py",
        "--mode", mode,
        "--env", env_name,
        "--obs", obs_type,
        "--device", "cuda"
    ]

    print(f"Command: {' '.join(cmd)}")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=False)
        end_time = time.time()

        if result.returncode == 0:
            duration = end_time - start_time
            print(".1f")
            return True
        else:
            print(f"[FAILED] Training failed with return code: {result.returncode}")
            return False

    except Exception as e:
        print(f"[ERROR] Training failed with exception: {e}")
        return False

def check_model_exists(env_name, obs_type):
    """Check if a trained model exists."""
    model_path = f"outputs/models/baseline/{env_name}_{obs_type.lower()}/metadata.json"
    return os.path.exists(model_path)

def main():
    """Train all baseline combinations sequentially."""

    # Define all combinations
    combinations = [
        ("highway-v0", "Lidar"),
        ("highway-v0", "GrayscaleObservation"),
        ("merge-v0", "Lidar"),
        ("merge-v0", "GrayscaleObservation"),
        ("intersection-v0", "Lidar"),
        ("intersection-v0", "GrayscaleObservation"),
    ]

    print("BASELINE MODEL TRAINING - SEQUENTIAL")
    print("="*70)
    print(f"Will train {len(combinations)} baseline models:")
    for i, (env, obs) in enumerate(combinations, 1):
        status = "[COMPLETED]" if check_model_exists(env, obs) else "[PENDING]"
        print(f"  {i}. {env} + {obs} - {status}")
    print("="*70)

    # Train each combination
    completed = 0
    total_start_time = time.time()

    for i, (env_name, obs_type) in enumerate(combinations, 1):
        if check_model_exists(env_name, obs_type):
            print(f"\n[SKIP] {env_name} + {obs_type} - Already trained")
            completed += 1
            continue

        print(f"\n[{i}/{len(combinations)}] Starting {env_name} + {obs_type}")

        if run_training(env_name, obs_type):
            completed += 1
            print(f"[OK] Completed {completed}/{len(combinations)} models")
        else:
            print(f"[FAILED] {env_name} + {obs_type}")
            # Continue with other models even if one fails

    total_duration = time.time() - total_start_time

    print("\n" + "="*70)
    print("BASELINE TRAINING COMPLETE")
    print("="*70)
    print(f"Completed: {completed}/{len(combinations)} models")
    print(f"Total time: {total_duration:.1f} minutes")
    print("\nNext steps:")
    print("1. Run evaluation: python evaluate_all.py --episodes 100")
    print("2. Check WandB dashboard for training metrics")
    print("3. Train generalist models for comparison")

if __name__ == "__main__":
    main()
