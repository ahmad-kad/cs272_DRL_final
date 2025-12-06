# train_crazy_driver_tqc.py

import argparse
import os
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# Try to import TQC from sb3-contrib, fallback to SAC
try:
    from sb3_contrib import TQC
    TQC_AVAILABLE = True
except ImportError:
    TQC = None  # Define TQC as None when not available
    TQC_AVAILABLE = False
    print("sb3-contrib not found, falling back to SAC")
    print("Install sb3-contrib for TQC: pip install sb3-contrib")

# Import the custom environment, reward wrapper, and ego-attention policy
from team2_env.crazy_driver_enviornment import crazy_driver_env
from team2_env.reward_wrapper import CrazyDriverRewardWrapper
from custom_policies import EgoAttentionExtractor

# Optional: Import wandb helpers if you want logging
try:
    from wandb_single_helpers import make_wandb_single_callback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def make_env(render_mode=None, duration=None):
    """
    Create the crazy_driver_env with reward shaping wrapper for TQC training.

    Uses survival bonuses and clamped penalties to prevent crash exploitation.

    Args:
        render_mode: Rendering mode for environment
        duration: Episode duration in seconds (optional, defaults to 60)
    """
    config = crazy_driver_env.default_config()

    # Basic environment configuration for continuous control
    config.update({
        "duration": duration if duration is not None else 60,  # Shorter episodes for faster iteration
        "offroad_terminal": True,  # STRICT: Terminate immediately when going off-road
        "vehicles_count": 30,     # Moderate traffic density
        "cop_count": 4,           # Standard cop count
        "npc_spawn_min_x": 200,   # Push traffic back to give more reaction space
        "offscreen_rendering": render_mode == "rgb_array",
        "render_agent": False,  # Don't render during training
        "show_trajectories": False,
    })

    env = gym.make(
        "CopChase-v0",
        render_mode=render_mode,
        config=config,
    )

    # Apply reward wrapper to prevent crash exploitation (suicide bug)
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
        description="Train TQC (or SAC fallback) on CopChase-v0 with Ego-Attention."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200_000,
        help="Total training timesteps (default: 200k)",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=4,  # More parallel envs for faster training
        help="Number of parallel envs for training (default: 4)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="If set, do not use Weights & Biases logging.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="crazy_driver_tqc",
        help="Name for saved model (default: crazy_driver_tqc)",
    )
    parser.add_argument(
        "--use_sac",
        action="store_true",
        help="Force use SAC instead of TQC even if sb3-contrib is available.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    total_timesteps = args.steps

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Determine algorithm: TQC if available, SAC otherwise
    if TQC_AVAILABLE and not args.use_sac:
        ALGO = TQC
        algo_name = "TQC"
        print("[ALGO] Using TQC (Truncated Quantile Critics) - SOTA for highway-env")
    else:
        ALGO = SAC
        algo_name = "SAC"
        if args.use_sac:
            print("[ALGO] Using SAC (forced via --use_sac)")
        else:
            print("[ALGO] Using SAC (fallback, install sb3-contrib for TQC)")

    # Build vectorized environment with normalization
    env = SubprocVecEnv([make_env_thunk(render_mode=None, duration=60) for _ in range(args.n_envs)])

    # Critical: Normalize observations and rewards for stable training
    env = VecNormalize(
        env,
        norm_obs=True,      # Normalize observations to mean=0, std=1
        norm_reward=True,   # Normalize rewards to mean=0, std=1
        clip_obs=10.0,      # Clip extreme observation values
        gamma=0.98          # Discount factor for reward normalization
    )

    # Force GPU training if available, CPU otherwise
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Model save path following convention: {env}_{policy}_{step}_{type}
    env_name = "copchase"
    policy_name = f"{algo_name.lower()}_ego_attention"
    step_name = f"{total_timesteps//1000}k"
    type_name = "sota_highway"  # State-of-the-art highway driving
    model_path = f"outputs/models/{env_name}_{policy_name}_{step_name}_{type_name}.zip"

    # TQC/SAC hyperparameters tuned for highway-env continuous control
    model_params = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": 5e-4,      # Standard learning rate for continuous control
        "buffer_size": 500_000,     # Large replay buffer for experience diversity
        "learning_starts": 1_000,   # Start learning early with diverse experiences
        "batch_size": 256,          # Standard batch size
        "tau": 0.02,                # Soft update coefficient
        "gamma": 0.98,              # Discount factor (look ahead but not too far)
        "train_freq": 1,            # Train every step
        "gradient_steps": 1,        # Single gradient step per train call
        "ent_coef": "auto_0.1",     # Automatic entropy tuning for exploration
        "target_update_interval": 1, # Update targets every step
        "tensorboard_log": f"./logs/tb_crazy_driver_{algo_name.lower()}_ego_attention/",
        "device": device,
        "verbose": 1,
        # Ego-Attention policy architecture
        "policy_kwargs": {
            "features_extractor_class": EgoAttentionExtractor,
            "features_extractor_kwargs": {
                "features_dim": 256,  # Rich feature representation
                "n_heads": 4,         # Multi-head attention
                "n_layers": 2         # Attention layers
            },
            # Larger networks for complex ego-attention features
            "net_arch": dict(pi=[256, 256], qf=[256, 256])
        }
    }

    # TQC-specific parameters for risk-aware training
    if TQC_AVAILABLE and not args.use_sac:
        model_params.update({
            "top_quantiles_to_drop_per_net": 2,  # Drop top 2 quantiles for conservatism
            "policy_kwargs": {
                **model_params["policy_kwargs"],
                "n_critics": 2,      # Number of critic networks
                "n_quantiles": 25    # Number of quantiles per critic
            }
        })

    print(f"[TRAIN] {algo_name} on CopChase-v0 with Ego-Attention")
    print(f"  Action space: Continuous (acceleration, steering)")
    print(f"  Observation space: Kinematic ({env.observation_space.shape[0]}D)")
    print(f"  Policy: MLP with Ego-Attention features [256, 256] networks")
    print(f"  Features: 256D ego-attention processing (permutation-invariant vehicles)")
    print(f"  Attention: {model_params['policy_kwargs']['features_extractor_kwargs']['n_heads']} heads, {model_params['policy_kwargs']['features_extractor_kwargs']['n_layers']} layers")
    print(f"  Algorithm: {algo_name} {'(distributional, pessimistic)' if ALGO == TQC else '(standard SAC)'}")
    print(f"  Reward shaping: Survival bonus + clamped penalties (anti-crash exploitation)")
    print(f"  Navigation focus: Ego-centric threat assessment, smooth continuous control")
    print(f"  Traffic handling: Attention focuses on immediate threats, safe distances")
    print(f"  Episode duration: 60s (shorter for faster iteration)")
    print(f"  Buffer: {model_params['buffer_size']:,} experiences, Batch: {model_params['batch_size']}")
    print(f"  Steps: {total_timesteps:,}, n_envs: {args.n_envs}, device: {device}")

    model = ALGO(**model_params)

    # Checkpoint callback (saves every 20k steps for 200k training)
    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,  # Save every 20k steps
        save_path="./outputs/models/checkpoints/",
        name_prefix=f"{algo_name.lower()}_ego_attention_checkpoint",
        save_replay_buffer=True,  # Save replay buffer for SAC/TQC
        save_vecnormalize=True,   # Save VecNormalize statistics
    )

    # W&B callback (optional)
    if args.no_wandb or not WANDB_AVAILABLE:
        callbacks = [checkpoint_callback]
        if args.no_wandb:
            print("[INFO] W&B logging disabled (--no_wandb set).")
        else:
            print("[INFO] W&B not available.")
        print("[SAVE] Checkpoint saving enabled (every 20k steps)")
    else:
        wandb_callback = make_wandb_single_callback(
            total_timesteps=total_timesteps,
            env_id="CopChase-v0",
            obs_type="kinematic_attention",
            project="crazy-driver-sota",
            run_name=f"crazy_driver_{algo_name.lower()}_ego_attention-{total_timesteps//1000}k",
            verbose=1,
        )
        callbacks = [checkpoint_callback, wandb_callback]
        print("[SAVE] Checkpoint saving enabled (every 20k steps)")
        print("[LOG] W&B logging enabled")

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

    # Save the model and VecNormalize statistics
    model.save(model_path)
    # Save VecNormalize statistics for proper reward scaling during inference
    vec_normalize_path = model_path.replace('.zip', '_vecnormalize.pkl')
    env.save(vec_normalize_path)
    env.close()
    print(f"[SAVE] {algo_name}+Ego-Attention model saved to {model_path}")
    print(f"[SAVE] VecNormalize stats saved to {vec_normalize_path}")
    print(f"[DONE] {algo_name} training with ego-attention focus finished.")


if __name__ == "__main__":
    main()
