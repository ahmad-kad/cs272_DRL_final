#!/usr/bin/env python3
"""
Main Entry Point for Training Generalist Agents

Trains both Lidar and Grayscale generalist agents that can beat
Highway, Merge, and Intersection environments.
"""

import sys
import os
import argparse
import importlib.util

def import_training_module():
    """Import training module with proper path setup."""
    # Add the highway_distillation directory to sys.path
    highway_dist_path = os.path.join(os.path.dirname(__file__), "highway_distillation")
    if highway_dist_path not in sys.path:
        sys.path.insert(0, highway_dist_path)

    # Now import the training module normally
    import training
    return training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Generalist Agents")
    parser.add_argument("--agent", type=str, choices=['lidar', 'grayscale', 'all'], default='all',
                      help="Which agent to train (default: all)")
    args = parser.parse_args()
    
    print("="*70)
    print("HIGHWAY DISTILLATION - GENERALIST AGENT TRAINING")
    print("="*70)
    print("Goal: Single agent to beat Highway, Merge, and Intersection")
    print("Strategy: Train on mixed scenarios for generalization")
    print("="*70)
    
    try:
        # Run training by calling the training script directly
        import subprocess

        if args.agent in ['lidar', 'all']:
            print("\n[1/2] Training Lidar Generalist Agent...")
            cmd = [sys.executable, "highway_distillation/training.py", "--agent", "lidar"]
            result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd)

        if args.agent in ['grayscale', 'all']:
            print("\n[2/2] Training Grayscale Generalist Agent...")
            cmd = [sys.executable, "highway_distillation/training.py", "--agent", "grayscale"]
            result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd)

        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print("Models saved in: highway_distillation/outputs/models/")
        print("Next: python run_evaluation.py")

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
