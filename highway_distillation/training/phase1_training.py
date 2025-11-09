#!/usr/bin/env python3
"""Phase 1: Multi-Modal Foundation Training

Learn basic driving with kinematics + lidar + visual fusion.
Optimized for speed: 1M timesteps, 128-unit networks, 8 vehicles.
"""

import os
import logging
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from environments.urban_junction_env import UrbanJunctionEnv
from custom_policies import MultiModalActorCriticPolicy
from training_logger import create_phase1_logger, create_episode_callback

PHASE1_MODEL_DIR = "models/phase1"
LOGS_DIR = "logs/phase1"

def create_stage_a_env():
    """Create Phase 1 environment with multi-modal observations."""
    config = UrbanJunctionEnv.default_config()

    # Multi-modal observations: kinematics + lidar + visual
    config["observation"].update({
        "multi_modal": True,
        "lidar_rays": 64,
        "lidar_range": 50.0,
        "visual_width": 84,
        "visual_height": 84,
        "vehicles_count": 8,  # Light traffic for learning
    })

    # Deterministic stages, no antagonistic vehicles
    config.update({
        "stage_mode": "deterministic",
        "antagonistic_vehicles": False,
        "vehicles_count": 8,
        "duration": 200,
    })

    env = UrbanJunctionEnv(config)

    # Standard RL wrappers
    env = Monitor(env, filename=os.path.join(LOGS_DIR, "monitor.csv"))
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    env = VecFrameStack(env, n_stack=2)

    return env

def create_ppo_model(env):
    """Create PPO model with multi-modal policy for sensor fusion."""
    model = PPO(
        policy=MultiModalActorCriticPolicy,
        env=env,
        learning_rate=1e-3,  # Fast learning
        n_steps=1024,
        batch_size=32,
        n_epochs=5,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.0,  # Deterministic
        policy_kwargs={"net_arch": [dict(pi=[128, 128], vf=[128, 128])]},
        verbose=1,
        seed=42,
        device="auto"
    )
    return model

def setup_directories():
    """Create model and log directories."""
    for directory in [PHASE1_MODEL_DIR, LOGS_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)

def train_phase1():
    """Train Phase 1: Multi-modal sensor fusion foundation."""
    # Initialize enhanced logger
    # Use compressed logging for cleaner output (set verbose=True for debugging)
    train_logger = create_phase1_logger(verbose=False)

    train_logger.logger.info("Phase 1: Multi-Modal Foundation Training")

    setup_directories()

    # Create environment and model
    env = create_stage_a_env()
    model = create_ppo_model(env)

    # Log configuration details
    config = {
        "phase": "Phase 1: Multi-Modal Foundation",
        "total_timesteps": "1,000,000 (100,000 test)",
        "environment": "Deterministic highway stages",
        "antagonistic_vehicles": "Disabled",
        "multi_modal": "Enabled (kinematics + lidar + visual)",
        "vehicles_count": 8,
        "network_architecture": "MultiModalActorCriticPolicy",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 5
    }
    train_logger.log_training_start(config)

    # Log environment setup
    env_config = {
        "observation_type": "Multi-modal (kinematics + lidar + visual)",
        "vehicles_count": 8,
        "stage_mode": "deterministic",
        "antagonistic_vehicles": False,
        "duration": 200
    }
    train_logger.log_environment_setup(env_config)

    # Log model architecture
    model_info = {
        "policy_type": "MultiModalActorCriticPolicy",
        "network_arch": "128x128 MLP + Conv branches",
        "parameters": "~500K estimated"
    }
    train_logger.log_model_architecture(model_info)

    # Create episode progress callback
    episode_callback = create_episode_callback("phase1")

    # Training setup
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=PHASE1_MODEL_DIR,
        name_prefix="ppo_stage_a",
        save_vecnormalize=True
    )

    sb3_logger = configure(LOGS_DIR, ["stdout", "csv"])

    # Train for short test (10k timesteps) or full training (1M)
    test_mode = os.environ.get("TEST_MODE", "false").lower() == "true"
    total_timesteps = 10_000 if test_mode else 1_000_000

    train_logger.logger.info(f"Starting training: {total_timesteps:,} timesteps")

    model.set_logger(sb3_logger)

    try:
        from stable_baselines3.common.callbacks import CallbackList
        callbacks = CallbackList([checkpoint_callback, episode_callback])

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=1 if test_mode else 10,
            progress_bar=True
        )

        # Save model and normalization stats
        final_model_path = os.path.join(PHASE1_MODEL_DIR, "ppo_stage_a_final")
        model.save(final_model_path)
        env.save(os.path.join(PHASE1_MODEL_DIR, "vec_normalize_stage_a.pkl"))

        train_logger.logger.info("Phase 1 training completed successfully")
        train_logger.logger.info(f"Model saved to: {final_model_path}")

        # Export convergence data for plotting
        episode_callback.training_logger.export_convergence_data()

        # Create training plots and final statistics
        episode_callback.training_logger.create_training_plots()
        stats = episode_callback.training_logger.get_summary_stats()
        if stats:
            episode_callback.training_logger.log_training_complete(total_timesteps, stats.get('total_episodes', 0))

    except KeyboardInterrupt:
        train_logger.logger.info("Training interrupted by user")
        interrupt_path = os.path.join(PHASE1_MODEL_DIR, "ppo_stage_a_interrupted")
        model.save(interrupt_path)
        env.save(os.path.join(PHASE1_MODEL_DIR, "vec_normalize_stage_a_interrupted.pkl"))
        train_logger.logger.info(f"Partial model saved to: {interrupt_path}")

    except Exception as e:
        episode_callback.training_logger.log_error(e, "Phase 1 training")
        raise

    finally:
        env.close()
        train_logger.logger.info("Environment closed")

if __name__ == "__main__":
    train_phase1()
