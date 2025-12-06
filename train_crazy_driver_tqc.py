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
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from (e.g., outputs/models/checkpoints/tqc_ego_attention_checkpoint_80000_steps.zip)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force training on CPU instead of GPU (useful for systems without CUDA)",
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

    # Build vectorized environment with normalization (REDUCED ENVIRONMENT COUNT)
    # Memory optimization: reduce parallel environments to prevent memory pressure
    n_envs_optimized = min(args.n_envs, 2)  # Cap at 2 for memory stability
    if args.n_envs > 2:
        print(f"[MEMORY] Reducing environments from {args.n_envs} to {n_envs_optimized} for stability")

    env = SubprocVecEnv([make_env_thunk(render_mode=None, duration=60) for _ in range(n_envs_optimized)])

    # Critical: Normalize observations and rewards for stable training
    env = VecNormalize(
        env,
        norm_obs=True,      # Normalize observations to mean=0, std=1
        norm_reward=True,   # Normalize rewards to mean=0, std=1
        clip_obs=5.0,       # REDUCED: Tighter clipping for numerical stability (was 10.0)
        gamma=0.98          # Discount factor for reward normalization
    )

    # Device selection: respect --cpu flag, otherwise use GPU if available
    if args.cpu:
        device = "cpu"
        print("Using CPU device (--cpu flag set)")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")

    # Model save path following convention: {env}_{policy}_{step}_{type}
    env_name = "copchase"
    policy_name = f"{algo_name.lower()}_ego_attention"
    step_name = f"{total_timesteps//1000}k"
    type_name = "sota_highway"  # State-of-the-art highway driving
    model_path = f"outputs/models/{env_name}_{policy_name}_{step_name}_{type_name}.zip"

    # TQC/SAC hyperparameters tuned for highway-env continuous control (MEMORY OPTIMIZED)
    model_params = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": 5e-4,      # Standard learning rate for continuous control
        "buffer_size": 250_000,     # REDUCED: Half buffer size for memory stability (was 500k)
        "learning_starts": 1_000,   # Start learning early with diverse experiences
        "batch_size": 128,          # REDUCED: Half batch size for memory stability (was 256)
        "tau": 0.02,                # Soft update coefficient
        "gamma": 0.98,              # Discount factor (look ahead but not too far)
        "train_freq": 1,            # Train every step
        "gradient_steps": 1,        # Single gradient step per train call
        "ent_coef": "auto_0.1",     # Automatic entropy tuning for exploration
        "target_update_interval": 2, # REDUCED: Less frequent target updates for stability (was 1)
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
    print(f"  Steps: {total_timesteps:,}, n_envs: {n_envs_optimized}, device: {device}")
    print(f"  Memory optimizations: 50% buffer, 50% batch, 50% envs, tighter obs clipping")
    if device == "cuda":
        print("  Stability: CUDA cache clearing enabled, reduced target updates")
    else:
        print("  Device: CPU mode - slower but more stable for long training runs")

    # Handle resume from checkpoint
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_from}")

        print(f"[RESUME] Loading checkpoint: {args.resume_from}")

        # Load the model with the environment to handle env count mismatch
        model = ALGO.load(args.resume_from, env=env)

        # Load VecNormalize stats
        vec_normalize_path = args.resume_from.replace('.zip', '_vecnormalize.pkl')
        if os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, env)
            print(f"[RESUME] Loaded VecNormalize stats from {vec_normalize_path}")
        else:
            print(f"[WARNING] VecNormalize stats not found at {vec_normalize_path}")

        # Load replay buffer if available
        replay_buffer_path = args.resume_from.replace('.zip', '_replay_buffer.pkl')
        if os.path.exists(replay_buffer_path):
            model.load_replay_buffer(replay_buffer_path)
            print(f"[RESUME] Loaded replay buffer from {replay_buffer_path}")
        else:
            print(f"[WARNING] Replay buffer not found at {replay_buffer_path}")

        # Try to extract already trained steps from checkpoint filename
        # Expected format: *_checkpoint_{steps}_steps.zip
        import re
        steps_match = re.search(r'_checkpoint_(\d+)_steps\.zip$', args.resume_from)
        if steps_match:
            trained_steps = int(steps_match.group(1))
            print(f"[RESUME] Detected {trained_steps:,} already trained steps")
            # If --steps was specified as additional steps, use it as-is
            # Otherwise, interpret --steps as total desired steps
            if args.steps == 200_000:  # Default value, probably means total desired
                additional_steps = max(0, args.steps - trained_steps)
                if additional_steps == 0:
                    print("[WARNING] Checkpoint already has enough steps. Specify --steps for additional training.")
                    additional_steps = 1000  # Default additional steps
                total_timesteps = additional_steps
                print(f"[RESUME] Will train additional {total_timesteps:,} steps (total will be {trained_steps + total_timesteps:,})")
            else:
                print(f"[RESUME] --steps interpreted as additional steps: {total_timesteps:,}")
        else:
            print("[WARNING] Could not extract trained steps from checkpoint filename")
            print("         Training will proceed with full specified timesteps")

    else:
        # Normal training initialization
        model = ALGO(**model_params)

    # Checkpoint callback (saves every 20k steps for 200k training)
    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,  # Save every 20k steps
        save_path="./outputs/models/checkpoints/",
        name_prefix=f"{algo_name.lower()}_ego_attention_checkpoint",
        save_replay_buffer=True,  # Save replay buffer for SAC/TQC
        save_vecnormalize=True,   # Save VecNormalize statistics
    )

    # Create memory management callback
    from stable_baselines3.common.callbacks import BaseCallback

    class MemoryManagementCallback(BaseCallback):
        def __init__(self, clear_interval=5000, verbose=0):
            super().__init__(verbose)
            self.clear_interval = clear_interval
            self.steps_since_clear = 0

        def _on_step(self) -> bool:
            self.steps_since_clear += 1
            if self.steps_since_clear >= self.clear_interval:
                if torch.cuda.is_available() and torch.cuda.current_device() >= 0:
                    torch.cuda.empty_cache()
                    if self.verbose > 0:
                        print(f"[MEMORY] CUDA cache cleared at step {self.num_timesteps}")
                self.steps_since_clear = 0
            return True

    # Add memory management callback
    memory_callback = MemoryManagementCallback(clear_interval=5000, verbose=1)

    # W&B callback (optional)
    if args.no_wandb or not WANDB_AVAILABLE:
        callbacks = [checkpoint_callback, memory_callback]
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
        callbacks = [checkpoint_callback, wandb_callback, memory_callback]
        print("[SAVE] Checkpoint saving enabled (every 20k steps)")
        print("[LOG] W&B logging enabled")

    # Train the model with memory management and stability improvements
    print(f"\n[TRAIN] Starting {algo_name} training with memory optimizations...")
    print(f"[MEMORY] Buffer: {model_params['buffer_size']:,} experiences")
    print(f"[MEMORY] Batch: {model_params['batch_size']} samples")
    print(f"[MEMORY] Environments: {n_envs_optimized}")
    print(f"[MEMORY] CUDA cache clearing: Every 5000 steps")
    print(f"[STABILITY] Target updates: Every {model_params['target_update_interval']} steps")
    print(f"[STABILITY] Observation clipping: 5.0 (tighter for numerical stability)")

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
