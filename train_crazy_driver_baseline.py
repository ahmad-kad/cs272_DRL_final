# train_crazy_driver_baseline.py

import argparse
import os
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

# Import the custom environment
from team2_env.crazy_driver_enviornment import crazy_driver_env

# Optional: Import wandb helpers if you want logging
try:
    from wandb_single_helpers import make_wandb_single_callback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def make_env(render_mode=None):
    """
    Create the crazy_driver_env for training.

    Uses kinematic observations (the default for this env).
    """
    config = crazy_driver_env.default_config()

    # Training-specific config overrides
    config.update({
        "offscreen_rendering": render_mode == "rgb_array",
        "render_agent": False,  # Don't render during training
        "show_trajectories": False,
    })

    env = gym.make(
        "CopChase-v0",
        render_mode=render_mode,
        config=config,
    )

    if render_mode is None:
        # For training: wrap in Monitor so infos contain 'episode'
        env = Monitor(env)

    return env


def make_env_thunk(render_mode=None):
    """
    Factory for SubprocVecEnv.
    Needs to be top-level so it's picklable.
    """
    def _init():
        return make_env(render_mode=render_mode)
    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline PPO on the crazy_driver_env."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500k)",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=8,
        help="Number of parallel envs for training (default: 8)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="If set, do not use Weights & Biases logging.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="crazy_driver_baseline",
        help="Name for saved model (default: crazy_driver_baseline)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    total_timesteps = args.steps

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Build SubprocVecEnv for true parallel env stepping
    env_fns = [make_env_thunk(render_mode=None) for _ in range(args.n_envs)]
    env = SubprocVecEnv(env_fns)

    # Decide device (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Model save path
    model_path = f"outputs/models/{args.model_name}_{total_timesteps//1000}k.zip"

    # Baseline PPO configuration for continuous action space
    model = PPO(
        "MlpPolicy",  # Multi-layer perceptron policy (standard baseline)
        env,
        verbose=1,
        learning_rate=3e-4,  # Standard learning rate
        ent_coef=0.05,      # Standard entropy coefficient
        n_steps=2048,       # Standard rollout length
        batch_size=64,      # Standard batch size
        gamma=0.99,         # Standard discount factor
        gae_lambda=0.95,    # Standard GAE parameter
        clip_range=0.2,     # Standard PPO clip range
        tensorboard_log="./logs/tb_crazy_driver_baseline/",
        device=device,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 64], vf=[128, 64])  # Standard architecture
        )
    )

    # W&B callback (optional)
    if args.no_wandb or not WANDB_AVAILABLE:
        callbacks = None
        if args.no_wandb:
            print("[INFO] W&B logging disabled (--no_wandb set).")
        else:
            print("[INFO] W&B not available.")
    else:
        callbacks = make_wandb_single_callback(
            total_timesteps=total_timesteps,
            env_id="CopChase-v0",
            obs_type="kinematic",
            project="crazy-driver-baseline",
            run_name=f"crazy_driver_baseline-{total_timesteps//1000}k",
            verbose=1,
        )

    print(f"[TRAIN] Baseline PPO on Crazy Driver environment")
    print(f"  Action space: Continuous (acceleration, steering)")
    print(f"  Observation space: Kinematic ({env.observation_space.shape[0]}D)")
    print(f"  Policy: MLP with architecture [128, 64] for both actor/critic")
    print(f"  Steps: {total_timesteps}, n_envs: {args.n_envs}")

    # Train the model
    if callbacks is not None:
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    else:
        model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Save the model
    model.save(model_path)
    env.close()
    print(f"[SAVE] Baseline model saved to {model_path}")
    print("[DONE] Baseline training finished.")


if __name__ == "__main__":
    main()
