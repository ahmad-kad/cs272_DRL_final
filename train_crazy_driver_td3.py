# train_crazy_driver_td3.py

import argparse
import os
import torch
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# Import the custom environment, reward wrapper, and spatial features
from team2_env.crazy_driver_enviornment import crazy_driver_env
from team2_env.reward_wrapper import CrazyDriverRewardWrapper
from custom_policies import SimpleSpatialExtractor
from training.adaptive_curriculum_trainer import create_adaptive_curriculum_callback

# Optional: Import wandb helpers if you want logging
try:
    from wandb_single_helpers import make_wandb_single_callback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def make_env(render_mode=None, duration=None):
    """
    Create the crazy_driver_env with anti-crash reward wrapper for training.

    Uses harsh penalties and survival incentives to prevent reward gaming.

    Args:
        render_mode: Rendering mode for environment
        duration: Episode duration in seconds (optional, defaults to 60)
    """
    config = crazy_driver_env.default_config()

    # Basic environment configuration (rewards handled by wrapper)
    config.update({
        "duration": duration if duration is not None else 60,  # Dynamic duration for curriculum
        "offroad_terminal": True,  # STRICT: Terminate immediately when going off-road to force on-road navigation
        "vehicles_count": 25,     # Reduced traffic density for navigable on-road survival
        "npc_spawn_min_x": 300,   # Push traffic back to give more learning space
        "offscreen_rendering": render_mode == "rgb_array",
        "render_agent": False,  # Don't render during training
        "show_trajectories": False,
    })

    env = gym.make(
        "CopChase-v0",
        render_mode=render_mode,
        config=config,
    )

    # Apply reward wrapper to prevent crash exploitation
    env = CrazyDriverRewardWrapper.create_for_sac_td3(env)

    if render_mode is None:
        # For training: wrap in Monitor so infos contain 'episode'
        env = Monitor(env)

    return env


def make_env_thunk(render_mode=None, duration=None):
    """
    Factory for SubprocVecEnv.
    Needs to be top-level so it's picklable.

    Args:
        render_mode: Rendering mode
        duration: Episode duration in seconds
    """
    def _init():
        return make_env(render_mode=render_mode, duration=duration)
    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TD3 on the crazy_driver_env."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
        help="Total training timesteps (default: 100k)",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=8,  # Increased for faster learning
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
        default="crazy_driver_td3",
        help="Name for saved model (default: crazy_driver_td3)",
    )
    parser.add_argument(
        "--adaptive_curriculum",
        action="store_true",
        help="Enable adaptive curriculum training with increasing episode lengths",
    )
    parser.add_argument(
        "--curriculum_verbose",
        type=int,
        default=1,
        help="Verbosity level for curriculum progression (default: 1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    total_timesteps = args.steps

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Build environment - use DummyVecEnv for TD3 (more stable than SubprocVecEnv)
    from stable_baselines3.common.vec_env import DummyVecEnv

    # Start with curriculum stage 1 duration (30s) if using adaptive curriculum
    initial_duration = 30 if args.adaptive_curriculum else 60

    if args.n_envs == 1:
        env = DummyVecEnv([make_env_thunk(render_mode=None, duration=initial_duration)])
    else:
        env = DummyVecEnv([make_env_thunk(render_mode=None, duration=initial_duration) for _ in range(args.n_envs)])

    # Apply reward normalization to prevent reward gaming
    # VecNormalize normalizes rewards to mean=0, std=1 to make them comparable
    env = VecNormalize(env, norm_reward=True, gamma=0.95)  # Lower gamma for TD3 stability

    # Force CPU training for stability
    device = "cpu"
    print(f"Using {device} device (forced CPU for stability)")

    # Model save path following convention: {env}_{policy}_{step}_{type}
    env_name = "copchase"
    policy_name = "td3"
    step_name = f"{total_timesteps//1000}k"
    type_name = "safe_navigation"  # Safe navigation through dense traffic
    model_path = f"outputs/models/{env_name}_{policy_name}_{step_name}_{type_name}.zip"

    # TD3 configuration optimized for traffic navigation
    model = TD3(
        "MlpPolicy",  # Multi-layer perceptron policy with spatial CNN features
        env,
        verbose=1,
        learning_rate=1e-3,  # Higher than SAC for faster convergence
        buffer_size=2_000_000,  # Large buffer for diverse traffic experiences
        learning_starts=10_000,  # Start learning later with more exploration
        batch_size=512,  # Large batches for better gradients
        tau=0.005,  # Slower target updates for stability
        gamma=0.95,  # Lower discounting for traffic (near-term focus)
        train_freq=(1, "step"),  # Train every step
        gradient_steps=1,  # Single gradient step per train call
        policy_delay=2,  # TD3: Update policy every 2 critic updates
        target_policy_noise=0.2,  # Exploration noise for deterministic policy
        target_noise_clip=0.5,  # Clip noise for stability
        tensorboard_log="./logs/tb_crazy_driver_td3_safe_navigation/",
        device=device,
        policy_kwargs=dict(
            features_extractor_class=SimpleSpatialExtractor,  # CNN-based spatial features
            features_extractor_kwargs={"features_dim": 256},  # Higher dimensional features
            net_arch=dict(pi=[256, 256], qf=[256, 256])  # Larger networks for better capacity
        )
    )

    # Checkpoint callback (saves every 10k steps for 100k training)
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,  # Save every 10k steps for more frequent checkpoints
        save_path="./outputs/models/checkpoints/",
        name_prefix="td3_safe_navigation_checkpoint",
        save_replay_buffer=True,  # Save replay buffer for TD3
        save_vecnormalize=True,   # Save VecNormalize statistics
    )

    # Adaptive curriculum callback (optional)
    curriculum_callback = None
    if args.adaptive_curriculum:
        curriculum_callback = create_adaptive_curriculum_callback(
            verbose=args.curriculum_verbose
        )
        print("[CURRICULUM] Adaptive curriculum training enabled")
        print("[CURRICULUM] Episode length will increase as agent improves")
    else:
        print("[CURRICULUM] Fixed episode length training (60s episodes)")

    # W&B callback (optional)
    if args.no_wandb or not WANDB_AVAILABLE:
        callbacks = [checkpoint_callback]
        if args.no_wandb:
            print("[INFO] W&B logging disabled (--no_wandb set).")
        else:
            print("[INFO] W&B not available.")
        print("[SAVE] Checkpoint saving enabled (every 10k steps)")
    else:
        wandb_callback = make_wandb_single_callback(
            total_timesteps=total_timesteps,
            env_id="CopChase-v0",
            obs_type="kinematic",
            project="crazy-driver-baseline",
            run_name=f"crazy_driver_td3_safe_navigation-{total_timesteps//1000}k",
            verbose=1,
        )
        callbacks = [checkpoint_callback, wandb_callback]
        print("[SAVE] Checkpoint saving enabled (every 10k steps)")
        print("[LOG] W&B logging enabled")

    # Add curriculum callback if enabled
    if curriculum_callback is not None:
        callbacks.append(curriculum_callback)

    print(f"[TRAIN] TD3 on Crazy Driver environment (Safe Navigation Focus)")
    print(f"  Action space: Continuous (acceleration, steering)")
    print(f"  Observation space: Kinematic ({env.observation_space.shape[0]}D)")
    print(f"  Policy: MLP with CNN spatial features [256, 256] networks")
    print(f"  Features: 256D spatial CNN processing (vehicles as spatial grid)")
    print(f"  Navigation focus: Safe distance, traffic awareness, smooth driving")
    print(f"  Traffic handling: Avoid 'walls of cars', maintain safe buffers")
    if args.adaptive_curriculum:
        print(f"  Curriculum: Adaptive (starts at 30s, increases to 300s)")
        print(f"  Episode duration: Dynamic (30s -> 60s -> 120s -> 180s -> 300s)")
    else:
        print(f"  Episode duration: Fixed 60s (shorter for faster iteration)")
    print(f"  Buffer: 2M experiences, Batch: 512, Learning starts: 10k steps")
    print(f"  TD3 Features: Policy delay=2, Target noise=0.2, Tau=0.005")
    print(f"  Steps: {total_timesteps}, n_envs: {args.n_envs}, device: {device}")

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

    # Save the model and VecNormalize statistics
    model.save(model_path)
    # Save VecNormalize statistics for proper reward scaling during inference
    vec_normalize_path = model_path.replace('.zip', '_vecnormalize.pkl')
    env.save(vec_normalize_path)
    env.close()
    print(f"[SAVE] TD3+Attention model saved to {model_path}")
    print(f"[SAVE] VecNormalize stats saved to {vec_normalize_path}")
    print("[DONE] TD3 training with safe navigation focus finished.")


if __name__ == "__main__":
    main()

