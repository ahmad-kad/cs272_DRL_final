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
import time
import wandb

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
    parser.add_argument("--wandb-project", type=str, default="highway-distillation",
                      help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                      help="WandB entity (team/user)")
    args = parser.parse_args()

    # Initialize WandB for the training run
    wandb_config = {
        "training_type": "generalist_agents",
        "agents_to_train": args.agent,
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "goal": "Single agent to beat Highway, Merge, and Intersection environments",
        "strategy": "Train on mixed scenarios for generalization"
    }

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"generalist_training_{args.agent}_{int(time.time())}",
        config=wandb_config,
        notes=f"Training {args.agent} generalist agents with WandB monitoring"
    )

    print("="*70)
    print("HIGHWAY DISTILLATION - GENERALIST AGENT TRAINING")
    print("="*70)
    print("Goal: Single agent to beat Highway, Merge, and Intersection")
    print("Strategy: Train on mixed scenarios for generalization")
    print(f"WandB Project: {args.wandb_project}")
    print("="*70)

    try:
        # Run training by calling the training script directly
        import subprocess
        training_start_time = time.time()

        if args.agent in ['lidar', 'all']:
            print("\n[1/2] Training Lidar Generalist Agent...")
            lidar_start = time.time()
            cmd = [sys.executable, "highway_distillation/training.py", "--agent", "lidar"]
            result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd)
            lidar_time = time.time() - lidar_start

            # Log lidar training completion
            wandb.log({
                "lidar_training_completed": True,
                "lidar_training_time_minutes": lidar_time / 60,
                "timestamp": time.time()
            })

        if args.agent in ['grayscale', 'all']:
            print("\n[2/2] Training Grayscale Generalist Agent...")
            gray_start = time.time()
            cmd = [sys.executable, "highway_distillation/training.py", "--agent", "grayscale"]
            result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd)
            gray_time = time.time() - gray_start

            # Log grayscale training completion
            wandb.log({
                "grayscale_training_completed": True,
                "grayscale_training_time_minutes": gray_time / 60,
                "timestamp": time.time()
            })

        total_training_time = time.time() - training_start_time

        # Log final summary
        final_metrics = {
            "training_completed": True,
            "total_training_time_minutes": total_training_time / 60,
            "agents_trained": args.agent,
            "models_saved": True,
            "final_timestamp": time.time()
        }

        if args.agent in ['lidar', 'all']:
            final_metrics["lidar_training_time"] = lidar_time / 60
        if args.agent in ['grayscale', 'all']:
            final_metrics["grayscale_training_time"] = gray_time / 60

        wandb.log(final_metrics)

        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print("Models saved in: highway_distillation/outputs/models/")
        print("Next: python run_evaluation.py")
        print(f"WandB Dashboard: https://wandb.ai/{wandb.run.entity}/{args.wandb_project}")

    except Exception as e:
        # Log error to WandB
        wandb.log({
            "training_failed": True,
            "error_message": str(e),
            "error_timestamp": time.time()
        })

        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
