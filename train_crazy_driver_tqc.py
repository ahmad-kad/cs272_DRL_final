# train_crazy_driver_tqc.py

import argparse
import os
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

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


class MemoryManagementCallback(BaseCallback):
    """Callback for periodic CUDA memory clearing and stability monitoring."""
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
        description="Train TQC on CopChase-v0 with Ego-Attention."
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Split training into N epochs (e.g., --epochs 20 --steps 100000 trains 20 epochs of 5k steps each)",
    )
    parser.add_argument(
        "--epoch_steps",
        type=int,
        default=None,
        help="Steps per epoch when using --epochs (overrides --steps if both specified)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Handle epoch-based training
    if args.epochs is not None:
        if args.epoch_steps is not None:
            epoch_steps = args.epoch_steps
            total_epochs = args.epochs
            print(f"[EPOCHS] Training in {total_epochs} epochs of {epoch_steps:,} steps each")
        else:
            # If --epochs specified but --epoch_steps not, divide total steps by epochs
            epoch_steps = args.steps // args.epochs
            total_epochs = args.epochs
            print(f"[EPOCHS] Training in {total_epochs} epochs of {epoch_steps:,} steps each (from {args.steps:,} total)")

        # Run epoch training
        run_epoch_training(args, total_epochs, epoch_steps)
        return

    # Normal single-run training
    total_timesteps = args.steps

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Determine algorithm: TQC only
    if TQC_AVAILABLE:
        ALGO = TQC
        algo_name = "TQC"
        print("[ALGO] Using TQC (Truncated Quantile Critics) - SOTA for highway-env")
    else:
        raise ImportError("TQC not available. Install sb3-contrib: pip install sb3-contrib")

    # Build vectorized environment with normalization
    # When resuming from checkpoint, use original n_envs to avoid buffer mismatch
    if args.resume_from:
        # Don't optimize n_envs when resuming - use the original count
        n_envs_to_use = args.n_envs
        print(f"[RESUME] Using original n_envs={n_envs_to_use} to match checkpoint buffer")
    else:
        # Memory optimization: reduce parallel environments to prevent memory pressure
        n_envs_to_use = min(args.n_envs, 2)  # Cap at 2 for memory stability
        if args.n_envs > 2:
            print(f"[MEMORY] Reducing environments from {args.n_envs} to {n_envs_to_use} for stability")

    env = SubprocVecEnv([make_env_thunk(render_mode=None, duration=60) for _ in range(n_envs_to_use)])

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
    if TQC_AVAILABLE:
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
    print(f"  Steps: {total_timesteps:,}, n_envs: {n_envs_to_use}, device: {device}")
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
        vec_normalize_path = args.resume_from.replace('_checkpoint_', '_checkpoint_vecnormalize_').replace('.zip', '.pkl')
        if os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, env)
            print(f"[RESUME] Loaded VecNormalize stats from {vec_normalize_path}")
        else:
            print(f"[WARNING] VecNormalize stats not found at {vec_normalize_path}")

        # Load replay buffer if available
        replay_buffer_path = args.resume_from.replace('_checkpoint_', '_checkpoint_replay_buffer_').replace('.zip', '.pkl')
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
            if args.steps == 100_000:  # Default value, probably means total desired
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

    # Checkpoint callback (saves every 10k steps for 100k training)
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,  # Save every 10k steps
        save_path="./outputs/models/checkpoints/",
        name_prefix=f"{algo_name.lower()}_ego_attention_checkpoint",
        save_replay_buffer=True,  # Save replay buffer for SAC/TQC
        save_vecnormalize=True,   # Save VecNormalize statistics
    )

    # MemoryManagementCallback is now defined at module level

    # Add memory management callback
    memory_callback = MemoryManagementCallback(clear_interval=5000, verbose=1)

    # W&B callback (optional)
    if args.no_wandb or not WANDB_AVAILABLE:
        callbacks = [checkpoint_callback, memory_callback]
        if args.no_wandb:
            print("[INFO] W&B logging disabled (--no_wandb set).")
        else:
            print("[INFO] W&B not available.")
        print("[SAVE] Checkpoint saving enabled (every 10k steps)")
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
        print("[SAVE] Checkpoint saving enabled (every 10k steps)")
        print("[LOG] W&B logging enabled")

    # Train the model with memory management and stability improvements
    print(f"\n[TRAIN] Starting {algo_name} training with memory optimizations...")
    print(f"[MEMORY] Buffer: {model_params['buffer_size']:,} experiences")
    print(f"[MEMORY] Batch: {model_params['batch_size']} samples")
    print(f"[MEMORY] Environments: {n_envs_to_use}")
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


def run_epoch_training(args, total_epochs, epoch_steps):
    """Run training in epochs for maximum stability with unified wandb logging."""
    print(f"\n[EPOCHS] Starting epoch-based training: {total_epochs} epochs × {epoch_steps:,} steps")
    print(f"[EPOCHS] Total training: {total_epochs * epoch_steps:,} steps")
    print(f"[EPOCHS] Memory will be cleared between epochs")
    print(f"[EPOCHS] STABILITY: Each epoch uses single environment for crash prevention")
    print(f"[EPOCHS] WARNING: Highway environment + ego-attention is memory intensive\n")

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        print("[INFO] tqdm not available, using basic progress display")
        use_tqdm = False

    current_checkpoint = args.resume_from
    completed_epochs = 0

    # Create single wandb run for all epochs (if wandb is enabled)
    global_wandb_callback = None
    if not args.no_wandb and WANDB_AVAILABLE:
        global_wandb_callback = make_wandb_single_callback(
            total_timesteps=total_epochs * epoch_steps,  # Total across all epochs
            env_id="CopChase-v0",
            obs_type="kinematic_attention",
            project="crazy-driver-sota",
            run_name=f"crazy_driver_tqc_ego_attention_{total_epochs}_epochs_{total_epochs * epoch_steps // 1000}k_steps",
            verbose=1,
        )
        print("[WANDB] Created unified run for all epochs")

    # Main epoch progress bar
    epoch_pbar = tqdm(total=total_epochs, desc=f"Epochs (0/{total_epochs})", unit="epoch", position=0, leave=True) if use_tqdm else None

    for epoch in range(1, total_epochs + 1):
        if not use_tqdm:
            print(f"{'='*60}")
            print(f"[EPOCH {epoch}/{total_epochs}] Starting epoch {epoch}")
            print(f"{'='*60}")

        # Create epoch-specific arguments
        epoch_args = args
        epoch_args.steps = epoch_steps

        if current_checkpoint:
            epoch_args.resume_from = current_checkpoint
            if not use_tqdm:
                print(f"[EPOCH] Resuming from: {current_checkpoint}")
        else:
            epoch_args.resume_from = None

        try:
            # Run one epoch of training with shared wandb callback
            run_single_epoch(epoch_args, epoch, global_wandb_callback)

            # Find the latest checkpoint from this epoch
            checkpoint_dir = "./outputs/models/checkpoints/"
            if os.path.exists(checkpoint_dir):
                checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
                if checkpoint_files:
                    # Sort by modification time to get the latest
                    checkpoint_files_with_time = [(f, os.path.getmtime(os.path.join(checkpoint_dir, f)))
                                                 for f in checkpoint_files]
                    checkpoint_files_with_time.sort(key=lambda x: x[1], reverse=True)
                    current_checkpoint = os.path.join(checkpoint_dir, checkpoint_files_with_time[0][0])
                    if not use_tqdm:
                        print(f"[EPOCH] Checkpoint saved: {current_checkpoint}")

            completed_epochs += 1

            if use_tqdm:
                epoch_pbar.set_description(f"Epochs ({completed_epochs}/{total_epochs})")
                epoch_pbar.update(1)
            else:
                print(f"[EPOCH {epoch}] Completed successfully! ({completed_epochs}/{total_epochs} done)")

            # Force garbage collection between epochs
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if not use_tqdm:
                    print("[EPOCH] Memory cleared between epochs")

            if not use_tqdm:
                print()

        except Exception as e:
            print(f"[EPOCH {epoch}] Failed with error: {e}")
            import traceback
            traceback.print_exc()
            if use_tqdm:
                epoch_pbar.set_description(f"Epochs (ERROR in {epoch})")
            else:
                print("[EPOCH] Attempting to continue with next epoch...")
            continue

    if epoch_pbar:
        epoch_pbar.close()

    print(f"{'='*60}")
    print(f"[EPOCHS] Training completed!")
    print(f"[EPOCHS] {completed_epochs}/{total_epochs} epochs completed successfully")
    print(f"[EPOCHS] Total steps trained: {completed_epochs * epoch_steps:,}")
    print(f"[EPOCHS] Final checkpoint: {current_checkpoint}")
    print(f"{'='*60}")


def run_single_epoch(args, epoch_num=None, shared_wandb_callback=None):
    """Run a single epoch of training."""
    # Determine algorithm
    if TQC_AVAILABLE:
        ALGO = TQC
        algo_name = "TQC"
        print("[ALGO] Using TQC (Truncated Quantile Critics) - SOTA for highway-env")
    else:
        raise ImportError("TQC not available. Install sb3-contrib: pip install sb3-contrib")

    # Build environment with AGGRESSIVE memory optimization for epoch training
    # When resuming from checkpoint, use original n_envs to avoid buffer mismatch
    if args.resume_from:
        # Don't optimize n_envs when resuming - use the original count
        n_envs_to_use = args.n_envs
        print(f"[RESUME] Using original n_envs={n_envs_to_use} to match checkpoint buffer")
    else:
        # AGGRESSIVE memory optimization for epoch training: single environment for stability
        n_envs_to_use = 1  # Use only 1 environment for maximum stability in epochs
        if args.n_envs > 1:
            print(f"[MEMORY] EPOCH TRAINING: Using single environment (n_envs=1) for maximum stability")
            print(f"[MEMORY] Original n_envs={args.n_envs} reduced to prevent multiprocessing crashes")

    try:
        env = SubprocVecEnv([make_env_thunk(render_mode=None, duration=60) for _ in range(n_envs_to_use)])
    except Exception as e:
        print(f"[ENV] SubprocVecEnv failed ({e}), using DummyVecEnv...")
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([make_env_thunk(render_mode=None, duration=60) for _ in range(n_envs_to_use)])

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=5.0,
        gamma=0.98
    )

    # Device selection
    if args.cpu:
        device = "cpu"
        print("Using CPU device (--cpu flag set)")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")

    # Model hyperparameters (optimized)
    model_params = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": 5e-4,
        "buffer_size": 250_000,     # Memory optimized
        "learning_starts": 1_000,
        "batch_size": 128,          # Memory optimized
        "tau": 0.02,
        "gamma": 0.98,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto_0.1",
        "target_update_interval": 2, # Memory optimized
        "tensorboard_log": f"./logs/tb_crazy_driver_{algo_name.lower()}_ego_attention/",
        "device": device,
        "verbose": 1,
        "policy_kwargs": {
            "features_extractor_class": EgoAttentionExtractor,
            "features_extractor_kwargs": {
                "features_dim": 256,
                "n_heads": 4,
                "n_layers": 2
            },
            "net_arch": dict(pi=[256, 256], qf=[256, 256])
        }
    }

    # TQC-specific parameters
    if TQC_AVAILABLE:
        model_params.update({
            "top_quantiles_to_drop_per_net": 2,
            "policy_kwargs": {
                **model_params["policy_kwargs"],
                "n_critics": 2,
                "n_quantiles": 25
            }
        })

    # Handle resume from checkpoint
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_from}")

        print(f"[RESUME] Loading checkpoint: {args.resume_from}")
        model = ALGO.load(args.resume_from, env=env)

        # Load VecNormalize stats
        vec_normalize_path = args.resume_from.replace('_checkpoint_', '_checkpoint_vecnormalize_').replace('.zip', '.pkl')
        if os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, env)
            print(f"[RESUME] Loaded VecNormalize stats")

        # Load replay buffer if available
        replay_buffer_path = args.resume_from.replace('_checkpoint_', '_checkpoint_replay_buffer_').replace('.zip', '.pkl')
        if os.path.exists(replay_buffer_path):
            model.load_replay_buffer(replay_buffer_path)
            print(f"[RESUME] Loaded replay buffer")

    else:
        model = ALGO(**model_params)

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./outputs/models/checkpoints/",
        name_prefix=f"{algo_name.lower()}_ego_attention_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    memory_callback = MemoryManagementCallback(clear_interval=5000, verbose=1)

    if args.no_wandb or not WANDB_AVAILABLE:
        callbacks = [checkpoint_callback, memory_callback]
    elif shared_wandb_callback is not None:
        # Use the shared wandb callback from epoch training
        callbacks = [checkpoint_callback, shared_wandb_callback, memory_callback]
        print(f"[WANDB] Using shared wandb run for epoch {epoch_num}")
    else:
        # Create individual wandb run (fallback for non-epoch training)
        wandb_callback = make_wandb_single_callback(
            total_timesteps=args.steps,
            env_id="CopChase-v0",
            obs_type="kinematic_attention",
            project="crazy-driver-sota",
            run_name=f"crazy_driver_{algo_name.lower()}_ego_attention_epoch_{epoch_num if epoch_num else 'single'}",
            verbose=1,
        )
        callbacks = [checkpoint_callback, wandb_callback, memory_callback]

    # Run training with custom progress reporting and stability measures
    print(f"[TRAIN] Starting {algo_name} training for {args.steps:,} steps")
    print(f"[STABILITY] Single environment mode for maximum crash resistance")

    # Force garbage collection before training
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[MEMORY] CUDA cache cleared before epoch training")

    # Run training with error resilience
    try:
        model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=True)
        print(f"[EPOCH] Training completed successfully")
    except Exception as e:
        print(f"[EPOCH] Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise to let epoch training handle it


if __name__ == "__main__":
    main()
