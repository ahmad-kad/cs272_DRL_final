#!/usr/bin/env python3
"""
Curriculum Learning for Multi-Modal Autonomous Driving

This script demonstrates curriculum learning that progressively teaches
the agent to use both lidar and grayscale observations together.

The curriculum follows this progression:
1. Single modality learning (lidar/grayscale alternation)
2. Mixed single modalities
3. Introduction of combined "both" modality
4. Full multi-modal learning

Usage:
    python train_curriculum_both.py --total-timesteps 100000
"""

import argparse
from training.adaptive_trainer import run_adaptive_curriculum

def main():
    parser = argparse.ArgumentParser(description="Train multi-modal curriculum learning")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--target-difficulty",
        type=float,
        default=1.0,
        help="Target difficulty level (0.0-1.0)"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--use-attention",
        action="store_true",
        help="Enable attention mechanisms"
    )

    args = parser.parse_args()

    print("Starting Multi-Modal Curriculum Learning")
    print("=" * 60)
    print("Curriculum Progression:")
    print("  0.0-0.2: Single modality focus (lidar/grayscale)")
    print("  0.2-0.4: Mixed single modalities")
    print("  0.4-0.6: Introduction of combined 'both' modality")
    print("  0.6-0.8: Combined modality becomes primary")
    print("  0.8-1.0: Full multi-modal mastery")
    print("=" * 60)

    # Run curriculum learning with modality progression enabled
    final_model = run_adaptive_curriculum(
        modality="both",  # Start with both as base
        total_timesteps=args.total_timesteps,
        use_attention=args.use_attention,
        use_wandb=not args.no_wandb,
        target_difficulty=args.target_difficulty,
        enable_modality_curriculum=True
    )

    print("\nMulti-Modal Curriculum Training Complete!")
    print("The agent has learned to effectively combine lidar and grayscale observations.")

if __name__ == "__main__":
    main()
