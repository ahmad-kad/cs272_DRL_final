# train_crazy_driver_sac.py

import argparse
import os
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# Import the custom environment, reward wrapper, and transformer features
from team2_env.crazy_driver_enviornment import crazy_driver_env
from team2_env.reward_wrapper import CrazyDriverRewardWrapper
from custom_policies import TransformerFeaturesExtractor

# Optional: Import wandb helpers if you want logging
try:
    from wandb_single_helpers import make_wandb_single_callback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def make_env(render_mode=None):
    """
    Create the crazy_driver_env with anti-crash reward wrapper for training.

    Uses harsh penalties and survival incentives to prevent reward gaming.
    """
    config = crazy_driver_env.default_config()

    # Basic environment configuration (rewards handled by wrapper)
    config.update({
        "duration": 60,  # Shorter episodes initially for faster iteration
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
        description="Train SAC on the crazy_driver_env."
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
        default=2,  # Reduced for stability with reward wrapper
        help="Number of parallel envs for training (default: 2)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="If set, do not use Weights & Biases logging.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="crazy_driver_sac",
        help="Name for saved model (default: crazy_driver_sac)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    total_timesteps = args.steps

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Build environment - use DummyVecEnv for SAC (more stable than SubprocVecEnv)
    from stable_baselines3.common.vec_env import DummyVecEnv
    if args.n_envs == 1:
        env = DummyVecEnv([make_env_thunk(render_mode=None)])
    else:
        env = DummyVecEnv([make_env_thunk(render_mode=None) for _ in range(args.n_envs)])

    # Apply reward normalization to prevent reward gaming
    # VecNormalize normalizes rewards to mean=0, std=1 to make them comparable
    env = VecNormalize(env, norm_reward=True, gamma=0.99)

    # Force CPU training for stability
    device = "cpu"
    print(f"Using {device} device (forced CPU for stability)")

    # Model save path following convention: {env}_{policy}_{step}_{type}
    env_name = "copchase"
    policy_name = "sac"
    step_name = f"{total_timesteps//1000}k"
    type_name = "safe_navigation"  # Safe navigation through dense traffic
    model_path = f"outputs/models/{env_name}_{policy_name}_{step_name}_{type_name}.zip"

    # SAC configuration with attention for better vehicle relationship modeling
    model = SAC(
        "MlpPolicy",  # Multi-layer perceptron policy with attention features
        env,
        verbose=1,
        learning_rate=3e-4,  # Slightly lower for transformer stability
        buffer_size=500000,  # Reduced buffer for CPU memory
        learning_starts=5000,  # Start learning after collecting experiences
        batch_size=128,  # Smaller batch size for CPU stability
        tau=0.01,  # Faster target updates
        gamma=0.95,  # Lower discounting to encourage longer-term planning
        train_freq=1,  # Train every step
        gradient_steps=1,  # Single gradient step per train call
        ent_coef='auto_0.15',  # More exploration for complex relationships
        target_update_interval=2,  # Update targets every 2 steps
        tensorboard_log="./logs/tb_crazy_driver_sac_safe_navigation/",
        device=device,
        policy_kwargs=dict(
            features_extractor_class=TransformerFeaturesExtractor,  # Attention-based features
            features_extractor_kwargs={"features_dim": 128},
            net_arch=dict(pi=[64, 64], qf=[64, 64])  # Smaller networks since features are richer
        )
    )

    # Checkpoint callback (saves every 10k steps for 100k training)
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,  # Save every 10k steps for more frequent checkpoints
        save_path="./outputs/models/checkpoints/",
        name_prefix="sac_safe_navigation_checkpoint",
        save_replay_buffer=True,  # Save replay buffer for SAC
        save_vecnormalize=True,   # Save VecNormalize statistics
    )

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
            run_name=f"crazy_driver_sac_safe_navigation-{total_timesteps//1000}k",
            verbose=1,
        )
        callbacks = [checkpoint_callback, wandb_callback]
        print("[SAVE] Checkpoint saving enabled (every 10k steps)")
        print("[LOG] W&B logging enabled")

    print(f"[TRAIN] SAC on Crazy Driver environment (Safe Navigation Focus)")
    print(f"  Action space: Continuous (acceleration, steering)")
    print(f"  Observation space: Kinematic ({env.observation_space.shape[0]}D)")
    print(f"  Policy: MLP with Transformer attention features [64, 64] networks")
    print(f"  Navigation focus: Safe distance, traffic awareness, smooth driving")
    print(f"  Traffic handling: Avoid 'walls of cars', maintain safe buffers")
    print(f"  Episode duration: 60s (shorter for faster iteration)")
    print(f"  Steps: {total_timesteps}, n_envs: {args.n_envs}, device: {device}")

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

    # Save the model and VecNormalize statistics
    model.save(model_path)
    # Save VecNormalize statistics for proper reward scaling during inference
    vec_normalize_path = model_path.replace('.zip', '_vecnormalize.pkl')
    env.save(vec_normalize_path)
    env.close()
    print(f"[SAVE] SAC+Attention model saved to {model_path}")
    print(f"[SAVE] VecNormalize stats saved to {vec_normalize_path}")
    print("[DONE] SAC training with safe navigation focus finished.")


if __name__ == "__main__":
    main()
