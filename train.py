import argparse
import wandb
from training.train_lidar import run_lidar_curriculum
from training.train_grayscale import run_grayscale_curriculum
from training.train_contrastive import ContrastiveTrainer
from training.adaptive_trainer import run_adaptive_curriculum

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Driving Training with Adaptive Curriculum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  curriculum    - Fixed phases (easy -> medium -> hard)
  adaptive      - Performance-driven difficulty scaling (RECOMMENDED)
  contrastive   - Fine-tuning with contrastive learning

Agent Modalities:
  lidar         - Fast, structured sensor data (RECOMMENDED)
  grayscale     - Visual camera input (experimental)

Examples:
  # Best starting point: Adaptive Lidar with attention
  python train.py --mode adaptive --agent lidar --attention --wandb

  # Fixed curriculum comparison
  python train.py --mode curriculum --agent lidar --wandb

  # Visual agent (slower, experimental)
  python train.py --mode adaptive --agent grayscale --wandb

  # Fine-tuning existing model
  python train.py --mode contrastive --agent lidar --base-model path/to/model.zip
        """
    )

    parser.add_argument("--mode", type=str,
                       choices=["curriculum", "adaptive", "contrastive"],
                       default="adaptive",
                       help="Training mode (adaptive recommended)")
    parser.add_argument("--agent", type=str,
                       choices=["lidar", "grayscale"],
                       required=True,
                       help="Agent modality (lidar recommended)")
    parser.add_argument("--attention", action="store_true",
                       help="Enable attention mechanisms (lidar only)")
    parser.add_argument("--timesteps", type=int, default=50000,
                       help="Total training timesteps")
    parser.add_argument("--base-model", type=str,
                       help="Path to base model for contrastive fine-tuning")
    parser.add_argument("--checkpoint", type=str,
                       help="Path to checkpoint model to resume training from")
    parser.add_argument("--target-difficulty", type=float, default=1.0,
                       help="Target difficulty level to reach (0.0-1.0)")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable Weights & Biases logging")

    args = parser.parse_args()

    # Setup wandb if requested
    if not args.no_wandb:
        wandb.init(
            project="autonomous-driving-adaptive",
            name=f"{args.mode}-{args.agent}{'-attention' if args.attention else ''}",
            config=vars(args)
        )

    # Dispatch to appropriate training mode
    if args.mode == "curriculum":
        if args.agent == "lidar":
            run_lidar_curriculum(timesteps_per_phase=args.timesteps)
        elif args.agent == "grayscale":
            run_grayscale_curriculum(timesteps_per_phase=args.timesteps)

    elif args.mode == "adaptive":
        run_adaptive_curriculum(
            modality=args.agent,
            total_timesteps=args.timesteps,
            use_attention=args.attention,
            use_wandb=not args.no_wandb,
            checkpoint_path=args.checkpoint,
            target_difficulty=args.target_difficulty
        )

    elif args.mode == "contrastive":
        if not args.base_model:
            parser.error("--base-model is required for contrastive mode")
            return

        trainer = ContrastiveTrainer(
            base_model_path=args.base_model,
            modality=args.agent
        )
        trainer.train(timesteps=args.timesteps, difficulty="hard")

if __name__ == "__main__":
    main()
