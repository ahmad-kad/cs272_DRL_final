# train_crazy_driver_i2a.py - Imagination-Augmented Agents Training
# The ultimate overkill solution for curriculum learning challenges

import argparse
import os
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# Import the I2A policy and other components
from i2a_policy import I2APolicy
from team2_env.crazy_driver_enviornment import crazy_driver_env
from team2_env.reward_wrapper import CrazyDriverRewardWrapper
from training.adaptive_curriculum_trainer import create_adaptive_curriculum_callback
from stable_baselines3.common.callbacks import BaseCallback

# Optional: Import wandb helpers if you want logging
try:
    from wandb_single_helpers import make_wandb_single_callback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WorldModelCacheCallback(BaseCallback):
    """Callback to automatically save world model cache during training."""

    def __init__(self, save_interval: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.save_interval = save_interval

    def _on_step(self) -> bool:
        # Auto-save world model at regular intervals
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'auto_save_world_model'):
            self.model.policy.auto_save_world_model(
                self.num_timesteps, self.save_interval
            )
        return True


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

    # Apply reward wrapper optimized for I2A
    env = CrazyDriverRewardWrapper.create_for_i2a(env)

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
        description="Train I2A (Imagination-Augmented Agents) on the crazy_driver_env."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50_000,
        help="Total training timesteps (default: 50k - I2A is sample efficient)",
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
        default="crazy_driver_i2a",
        help="Name for saved model (default: crazy_driver_i2a)",
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
    parser.add_argument(
        "--imagination_horizon",
        type=int,
        default=5,
        help="How many steps ahead I2A imagines (default: 5)",
    )
    parser.add_argument(
        "--num_imaginations",
        type=int,
        default=8,
        help="Number of imagined trajectories per decision (default: 8)",
    )
    parser.add_argument(
        "--world_model_cache",
        type=str,
        default=None,
        help="Path to save/load world model cache for faster iteration",
    )
    parser.add_argument(
        "--freeze_world_model",
        action="store_true",
        help="Freeze world model after loading cache (only train policy)",
    )
    parser.add_argument(
        "--cache_save_interval",
        type=int,
        default=10000,
        help="Steps between automatic world model cache saves (default: 10k)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    total_timesteps = args.steps

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Build environment - use DummyVecEnv for PPO (more stable than SubprocVecEnv)
    from stable_baselines3.common.vec_env import DummyVecEnv

    # Start with curriculum stage 1 duration (30s) if using adaptive curriculum
    initial_duration = 30 if args.adaptive_curriculum else 60

    if args.n_envs == 1:
        env = DummyVecEnv([make_env_thunk(render_mode=None, duration=initial_duration)])
    else:
        env = DummyVecEnv([make_env_thunk(render_mode=None, duration=initial_duration) for _ in range(args.n_envs)])

    # Apply reward normalization to prevent reward gaming
    # VecNormalize normalizes rewards to mean=0, std=1 to make them comparable
    env = VecNormalize(env, norm_reward=True, gamma=0.99)  # Lower gamma for I2A planning

    # Force CPU training for stability
    device = "cpu"
    print(f"Using {device} device (forced CPU for stability)")

    # Model save path following convention: {env}_{policy}_{step}_{type}
    env_name = "copchase"
    policy_name = "i2a"
    step_name = f"{total_timesteps//1000}k"
    type_name = "imagination_augmented"  # I2A with imagination
    model_path = f"outputs/models/{env_name}_{policy_name}_{step_name}_{type_name}.zip"

    # I2A-PPO configuration - the ultimate curriculum breaker
    model = PPO(
        I2APolicy,  # Use our custom I2A policy directly
        env,
        verbose=1,
        learning_rate=3e-4,  # Standard PPO learning rate
        n_steps=2048,        # Longer trajectories for better credit assignment
        batch_size=512,      # Large batches for stable gradients
        n_epochs=10,         # Multiple epochs per update (PPO standard)
        gamma=0.95,          # Lower discounting for traffic (near-term focus)
        gae_lambda=0.95,     # Better credit assignment for curriculum
        clip_range=0.2,      # Adaptive clipping for stability
        ent_coef=0.01,       # Low entropy bonus (I2A explores via imagination)
        vf_coef=0.5,         # Balanced value/policy loss
        max_grad_norm=0.5,   # Gradient clipping for stability
        tensorboard_log="./logs/tb_crazy_driver_i2a_imagination_augmented/",
        device=device,
        policy_kwargs=dict(
            # I2A-specific parameters
            imagination_horizon=args.imagination_horizon,
            num_imaginations=args.num_imaginations,
            world_model_update_freq=1,
        ),
    )

    # Checkpoint callback (saves every 10k steps for 50k training)
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,  # Save every 10k steps for more frequent checkpoints
        save_path="./outputs/models/checkpoints/",
        name_prefix="i2a_imagination_augmented_checkpoint",
        save_replay_buffer=True,  # Save replay buffer for PPO
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
            run_name=f"crazy_driver_i2a_imagination_augmented-{total_timesteps//1000}k",
            verbose=1,
        )
        callbacks = [checkpoint_callback, wandb_callback]
        print("[SAVE] Checkpoint saving enabled (every 10k steps)")
        print("[LOG] W&B logging enabled")

    # Add curriculum callback if enabled
    if curriculum_callback is not None:
        callbacks.append(curriculum_callback)

    # World Model Caching Setup
    if args.world_model_cache:
        model.policy.set_world_model_cache_path(args.world_model_cache)

        # Try to load existing cache
        cache_loaded = model.policy.load_world_model_cache(
            args.world_model_cache + ".pt",
            freeze_world_model=args.freeze_world_model
        )

        if cache_loaded and args.freeze_world_model:
            print("[CACHE] World model loaded and frozen - policy-only training")
        elif cache_loaded:
            print("[CACHE] World model loaded - will continue training")
        else:
            print("[CACHE] No world model cache found - training from scratch")

        # Add automatic caching callback
        cache_callback = WorldModelCacheCallback(
            save_interval=args.cache_save_interval,
            verbose=1 if args.curriculum_verbose else 0
        )
        callbacks.append(cache_callback)
        print(f"[CACHE] Automatic world model saving every {args.cache_save_interval} steps")

    print(f"[TRAIN] I2A (Imagination-Augmented Agents) on Crazy Driver environment")
    print(f"  Action space: Continuous (acceleration, steering)")
    print(f"  Observation space: Kinematic ({env.observation_space.shape[0]}D)")
    print(f"  Policy: I2A with CNN spatial features [512, 256] networks")
    print(f"  Features: 256D spatial CNN processing (vehicles as spatial grid)")
    print(f"  Imagination: {args.imagination_horizon}-step horizon, {args.num_imaginations} trajectories")
    print(f"  Navigation focus: Safe distance, traffic awareness, smooth driving")
    print(f"  Traffic handling: Avoid 'walls of cars', maintain safe buffers")
    if args.adaptive_curriculum:
        print(f"  Curriculum: Adaptive (starts at 30s, increases to 300s)")
        print(f"  Episode duration: Dynamic (30s -> 60s -> 120s -> 180s -> 300s)")
    else:
        print(f"  Episode duration: Fixed 60s (shorter for faster iteration)")
    print(f"  World Model: Learns traffic dynamics for imagination")
    print(f"  Planning: Imagines {args.num_imaginations} trajectories, {args.imagination_horizon} steps ahead")
    print(f"  Steps: {total_timesteps}, n_envs: {args.n_envs}, device: {device}")

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

    # Save the model and VecNormalize statistics
    model.save(model_path)
    # Save VecNormalize statistics for proper reward scaling during inference
    vec_normalize_path = model_path.replace('.zip', '_vecnormalize.pkl')
    env.save(vec_normalize_path)

    # Save final world model cache
    if args.world_model_cache:
        final_cache_path = args.world_model_cache + "_final.pt"
        model.policy.save_world_model_cache(final_cache_path)

    env.close()
    print(f"[SAVE] I2A+Imagination model saved to {model_path}")
    print(f"[SAVE] VecNormalize stats saved to {vec_normalize_path}")
    if args.world_model_cache:
        print(f"[SAVE] World model cache saved to {final_cache_path}")
    print("[DONE] I2A imagination-augmented training finished.")


if __name__ == "__main__":
    main()
