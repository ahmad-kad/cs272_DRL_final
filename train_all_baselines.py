#!/usr/bin/env python3
"""
Multi-Environment Baseline Training System

Trains models on all combinations of environments and observation types:
- Environments: highway-v0, merge-v0, intersection-v0
- Observations: Kinematics, Lidar, GrayscaleObservation

Supports both single-environment and multi-environment training modes.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

import wandb

# Training configurations for different modes
QUICK_CONFIG = {
    'timesteps': 50000,
    'num_envs': 8,
    'n_steps': 1024,
    'batch_size': 128,
    'learning_rate': 5e-4,
    'checkpoint_freq': 10000,
}

STANDARD_CONFIG = {
    'timesteps': 150000,
    'num_envs': 4,
    'n_steps': 2048,
    'batch_size': 64,
    'learning_rate': 3e-4,
    'checkpoint_freq': 25000,
}

FULL_CONFIG = {
    'timesteps': 300000,
    'num_envs': 4,
    'n_steps': 2048,
    'batch_size': 64,
    'learning_rate': 3e-4,
    'checkpoint_freq': 50000,
}


class WandBCallback(BaseCallback):
    """WandB callback for baseline training with vectorized environment support."""

    def __init__(self, env_name: str, obs_type: str, log_freq: int = 250, verbose: int = 0):
        super().__init__(verbose)
        self.env_name = env_name
        self.obs_type = obs_type
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.num_envs = None  # Will be set on first step
        self.episodes_completed_this_interval = 0

    def _on_step(self) -> bool:
        # Get environment data
        dones = self.locals.get('dones', np.array([]))
        rewards = self.locals.get('rewards', np.array([]))

        # Initialize per-environment tracking on first step
        if self.num_envs is None:
            self.num_envs = len(dones) if len(dones) > 0 else 1
            # Initialize tracking for each environment
            for env_idx in range(self.num_envs):
                setattr(self, f'current_reward_{env_idx}', 0.0)
                setattr(self, f'current_length_{env_idx}', 0)
                setattr(self, f'env_episode_rewards_{env_idx}', [])

        # Track episodes for each environment
        for env_idx in range(self.num_envs):
            if env_idx >= len(rewards):
                continue

            # Accumulate reward and length for this environment
            current_reward = getattr(self, f'current_reward_{env_idx}')
            current_length = getattr(self, f'current_length_{env_idx}')

            current_reward += rewards[env_idx]
            current_length += 1

            setattr(self, f'current_reward_{env_idx}', current_reward)
            setattr(self, f'current_length_{env_idx}', current_length)

            # Check if episode ended for this environment
            if env_idx < len(dones) and dones[env_idx]:
                # Episode completed
                env_episode_rewards = getattr(self, f'env_episode_rewards_{env_idx}')
                env_episode_rewards.append(current_reward)
                self.episode_rewards.append(current_reward)
                self.episode_lengths.append(current_length)
                self.episodes_completed_this_interval += 1

                # Log episode immediately when it completes
                episode_metrics = {
                    "episode_reward": current_reward,
                    "episode_length": current_length,
                    "episode_env_id": env_idx,
                    "episode_total_count": len(self.episode_rewards),
                }

                # Add running statistics if we have enough episodes
                if len(self.episode_rewards) >= 5:
                    recent_rewards = self.episode_rewards[-10:]
                    episode_metrics.update({
                        "episode_recent_mean": np.mean(recent_rewards),
                        "episode_recent_std": np.std(recent_rewards) if len(recent_rewards) > 1 else 0,
                    })
                
                wandb.log(episode_metrics, step=self.num_timesteps)

                # Keep only recent episodes (global and per-environment)
                if len(self.episode_rewards) > 100:
                    self.episode_rewards = self.episode_rewards[-100:]
                    self.episode_lengths = self.episode_lengths[-100:]

                # Keep per-environment episodes for rolling stats
                if len(env_episode_rewards) > 50:
                    env_episode_rewards[:] = env_episode_rewards[-50:]

                # Reset for next episode in this environment
                setattr(self, f'current_reward_{env_idx}', 0.0)
                setattr(self, f'current_length_{env_idx}', 0)

        # Log at regular intervals (more frequent for multi-env)
        # Use adaptive frequency: more frequent early in training
        adaptive_freq = 50 if self.num_timesteps < 1000 else self.log_freq

        if self.n_calls % adaptive_freq == 0:
            # === TRAINING METRICS (should show as line plots over time) ===
            metrics = {
                # Basic training progress
                "timesteps": self.num_timesteps,
                "updates": self.n_calls,
                "step_count": self.n_calls,
                "env_steps": self.num_timesteps // self.num_envs if self.num_envs > 0 else 0,
                "parallel_envs": self.num_envs,
                "episodes_this_interval": self.episodes_completed_this_interval,
                "has_episodes": 1 if len(self.episode_rewards) > 0 else 0,  # Numeric flag
            }

            # === PPO TRAINING METRICS (should show as line plots) ===
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                logs = self.model.logger.name_to_value
                ppo_metrics = {
                    'ppo_value_loss': 'train/value_loss',
                    'ppo_policy_loss': 'train/policy_gradient_loss',
                    'ppo_entropy_loss': 'train/entropy_loss',
                    'ppo_approx_kl': 'train/approx_kl',
                    'ppo_clip_fraction': 'train/clip_fraction',
                    'ppo_learning_rate': 'train/learning_rate',
                    'ppo_n_updates': 'train/n_updates'
                }
                for metric_name, log_key in ppo_metrics.items():
                    if log_key in logs:
                        metrics[metric_name] = logs[log_key]

            # === EPISODE PERFORMANCE METRICS (should show as line plots) ===
            if self.episode_rewards:
                recent_rewards = self.episode_rewards[-20:]  # Last 20 episodes across all envs
                recent_lengths = self.episode_lengths[-20:]
                all_rewards = self.episode_rewards

                # Core episode statistics
                metrics.update({
                    "episode_mean_reward": np.mean(recent_rewards),
                    "episode_best_reward": np.max(recent_rewards),
                    "episode_worst_reward": np.min(recent_rewards),
                    "episode_reward_std": np.std(recent_rewards) if len(recent_rewards) > 1 else 0,
                    "all_time_best_reward": np.max(all_rewards),
                    "episode_mean_length": np.mean(recent_lengths),
                    "total_episode_count": len(self.episode_rewards),
                })

                # Learning progress indicators
                if len(self.episode_rewards) > 10:
                    early_avg = np.mean(self.episode_rewards[:10])
                    recent_avg = np.mean(recent_rewards)
                    metrics["learning_improvement"] = recent_avg - early_avg

                    # Calculate success rate (episodes with positive reward)
                    positive_episodes = sum(1 for r in recent_rewards if r > 0)
                    metrics["episode_success_rate"] = positive_episodes / len(recent_rewards)

            # Reset interval counter
            self.episodes_completed_this_interval = 0

            # Log to wandb
            wandb.log(metrics, step=self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        """Log final training summary and populate WandB summary."""
        if self.episode_rewards:
            final_avg = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards)
            final_std = np.std(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.std(self.episode_rewards) if len(self.episode_rewards) > 1 else 0
            
            final_metrics = {
                "final_avg_reward": final_avg,
                "final_best_reward": np.max(self.episode_rewards),
                "final_worst_reward": np.min(self.episode_rewards),
                "final_total_episodes": len(self.episode_rewards),
                "final_reward_std": final_std,
                "training_completed": 1,  # Numeric for plotting
            }
            wandb.log(final_metrics)
            
            # Populate WandB summary for easy dashboard viewing
            wandb.summary.update({
                "best_episode_reward": float(np.max(self.episode_rewards)),
                "final_avg_reward": float(final_avg),
                "total_episodes": len(self.episode_rewards),
                "env_name": self.env_name,
                "obs_type": self.obs_type,
                "training_completed": True
            })
        else:
            # Even if no episodes, log that training completed
            wandb.summary.update({
                "training_completed": True,
                "env_name": self.env_name,
                "obs_type": self.obs_type,
                "total_episodes": 0
            })


def _read_monitor_stats(monitor_dir: Path) -> Optional[Dict]:
    """
    Read Monitor CSV files to verify episode data.
    
    Args:
        monitor_dir: Directory containing monitor CSV files
        
    Returns:
        Dictionary with aggregated statistics or None if no data
    """
    if not monitor_dir.exists():
        return None
    
    all_rewards = []
    all_lengths = []
    
    # Find all monitor CSV files
    monitor_files = list(monitor_dir.glob("*.monitor.csv"))
    
    if not monitor_files:
        return None
    
    for csv_file in monitor_files:
        try:
            with open(csv_file, 'r') as f:
                lines = f.readlines()
                # Skip header lines (start with #)
                for line in lines:
                    if line.startswith('#') or line.strip() == '' or line.startswith('r,l,t'):
                        continue
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        try:
                            reward = float(parts[0])
                            length = int(float(parts[1]))
                            all_rewards.append(reward)
                            all_lengths.append(length)
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            continue
    
    if not all_rewards:
        return None
    
    return {
        "monitor_total_episodes": len(all_rewards),
        "monitor_mean_reward": np.mean(all_rewards),
        "monitor_reward_std": np.std(all_rewards) if len(all_rewards) > 1 else 0,
        "monitor_best_reward": np.max(all_rewards),
        "monitor_worst_reward": np.min(all_rewards),
        "monitor_mean_length": np.mean(all_lengths),
        "monitor_length_std": np.std(all_lengths) if len(all_lengths) > 1 else 0,
    }


def create_highway_env(
    env_name: str,
    obs_type: str = "Kinematics",
    config_overrides: Optional[Dict] = None
) -> gym.Env:
    """
    Create a configured highway-env environment.

    Args:
        env_name: Environment name (highway-v0, merge-v0, intersection-v0)
        obs_type: Observation type (Kinematics, Lidar, GrayscaleObservation)
        config_overrides: Optional config overrides

    Returns:
        Configured environment
    """
    env = gym.make(env_name, render_mode=None)
    unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env

    # Base configuration
    base_config = {
        "observation": {"type": obs_type}
    }

    # Observation-specific settings
    if obs_type == "Kinematics":
        base_config["observation"].update({
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
            "vehicles_count": 15,
            "absolute": False,
            "normalize": True
        })
    elif obs_type == "Lidar":
        # Use LidarObservation (correct class name in highway-env)
        base_config["observation"]["type"] = "LidarObservation"
        base_config["observation"].update({
            "cells": 32,
            "maximum_range": 50.0
        })
    elif obs_type == "GrayscaleObservation":
        base_config["observation"].update({
            "observation_shape": (64, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140]
        })

    # Apply overrides
    if config_overrides:
        base_config.update(config_overrides)

    # Configure the unwrapped environment
    unwrapped_env.configure(base_config)
    env.reset()

    return env


def create_multi_env(
    env_names: List[str],
    obs_type: str,
    num_envs_per_type: int = 2,
    use_subprocess: bool = False
) -> Tuple[gym.Env, List[str]]:
    """
    Create a mixed multi-environment setup.

    Args:
        env_names: List of environment names
        obs_type: Observation type
        num_envs_per_type: Number of parallel envs per environment type
        use_subprocess: Use SubprocVecEnv for better performance

    Returns:
        Tuple of (vectorized environment, list of env names in order)
    """
    env_list = []

    for env_name in env_names:
        for _ in range(num_envs_per_type):
            def make_env(name=env_name):
                def _init():
                    env = create_highway_env(name, obs_type)
                    log_dir = Path("outputs/logs/baseline/monitor")
                    log_dir.mkdir(parents=True, exist_ok=True)
                    return Monitor(env, filename=str(log_dir / f"monitor_{name}_{os.getpid()}"))
                return _init
            env_list.append(make_env())

    # Create vectorized environment
    if use_subprocess and len(env_list) > 1:
        vec_env = SubprocVecEnv(env_list)
    else:
        vec_env = DummyVecEnv(env_list)

    # Apply normalization
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )

    return vec_env, env_names * num_envs_per_type


def train_single_baseline(
    env_name: str,
    obs_type: str,
    config: Dict,
    device: str = "cuda"
) -> Dict:
    """
    Train a baseline model on a single environment.

    Args:
        env_name: Environment name
        obs_type: Observation type
        config: Training configuration
        device: Device for training

    Returns:
        Dictionary with training results
    """
    print(f"Training {env_name} with {obs_type} observations")

    # Initialize WandB for this baseline
    wandb_config = {
        "baseline_training": True,
        "env_name": env_name,
        "obs_type": obs_type,
        "training_mode": "single_env_baseline",
        "total_timesteps": config['timesteps'],
        "num_parallel_envs": config['num_envs'],
        "device": device,
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S")
    }

    wandb.init(
        project="highway-distillation-baselines",
        name=f"baseline_{env_name}_{obs_type.lower()}_{int(time.time())}",
        config=wandb_config,
        notes=f"Training baseline agent on {env_name} with {obs_type} observations"
    )
    
    # Log initial metrics to ensure dashboard shows data immediately
    wandb.log({
        "timesteps": 0,
        "updates": 0,
        "parallel_envs": config['num_envs'],
        "has_episodes": 0,
        "training_started": 1  # Numeric flag
    }, step=0)

    # Create save directory
    env_clean = env_name.replace('-', '_')
    obs_clean = obs_type.lower()
    save_dir = Path(f"outputs/models/baseline/{env_clean}_{obs_clean}")
    save_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(f"outputs/logs/baseline/{env_clean}_{obs_clean}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    def make_env():
        def _init():
            env = create_highway_env(env_name, obs_type)
            return Monitor(env, filename=str(log_dir / "monitor" / f"env_{os.getpid()}"))
        return _init

    (log_dir / "monitor").mkdir(parents=True, exist_ok=True)

    num_envs = config['num_envs']
    train_env = DummyVecEnv([make_env() for _ in range(num_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Determine policy type
    if obs_type == "GrayscaleObservation":
        policy_type = "CnnPolicy"
        policy_kwargs = {}
    else:
        policy_type = "MlpPolicy"
        policy_kwargs = {"net_arch": [256, 256]}

    # Create model
    model = PPO(
        policy_type,
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device=device,
        tensorboard_log=str(log_dir)
    )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config['checkpoint_freq'] // num_envs,
        save_path=str(save_dir),
        name_prefix=f"baseline_{env_clean}_{obs_clean}"
    )

    # Enhanced WandB logging
    wandb_callback = WandBCallback(env_name, obs_type, log_freq=100, verbose=1)  # More frequent logging

    # Track training start time
    training_start_time = time.time()

    print(f"Training for {config['timesteps']} steps on {device}...")
    print(f"WandB logging enabled - check dashboard for metrics")

    model.learn(
        total_timesteps=config['timesteps'],
        callback=[checkpoint_callback, wandb_callback]
    )

    training_duration = time.time() - training_start_time

    # Verify Monitor data as fallback
    monitor_dir = log_dir / "monitor"
    monitor_stats = _read_monitor_stats(monitor_dir)
    if monitor_stats and wandb.run is not None:
        wandb.log(monitor_stats, step=config['timesteps'])
        print(f"Monitor verification: {monitor_stats['monitor/total_episodes']} episodes, mean reward: {monitor_stats['monitor/mean_reward']:.2f}")

    # Save final model
    model.save(str(save_dir / "final_model"))
    train_env.save(str(save_dir / "vec_normalize.pkl"))

    print(f"Saved baseline agent to: {save_dir}/final_model")

    # Evaluate final model
    print(f"Evaluating final model...")
    mean_reward, std_reward = evaluate_policy(
        model, train_env, n_eval_episodes=20, deterministic=True
    )

    print(f"Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
    # Save metadata
    metadata = {
        "env_name": env_name,
        "obs_type": obs_type,
        "total_timesteps": config['timesteps'],
        "num_envs": num_envs,
        "training_time_seconds": training_duration,
        "final_mean_reward": float(mean_reward),
        "final_std_reward": float(std_reward),
        "policy_type": policy_type,
        "training_completed": True,
        "wandb_project": "highway-distillation-baselines"
    }

    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Final wandb logging
    if wandb.run is not None:
        final_summary = {
            "final_evaluation_mean_reward": float(mean_reward),
            "final_evaluation_std_reward": float(std_reward),
            "total_training_time_minutes": training_duration / 60,
            "training_completed": 1,  # Numeric flag
        }
        wandb.log(final_summary)
        
        # Populate WandB summary for dashboard visibility
        wandb.summary.update({
            "final_eval_mean_reward": float(mean_reward),
            "final_eval_std_reward": float(std_reward),
            "total_training_time_minutes": training_duration / 60,
            "total_timesteps": config['timesteps'],
            "num_parallel_envs": num_envs,
            "env_name": env_name,
            "obs_type": obs_type,
            "policy_type": policy_type,
            "training_completed": True
        })
        
        # Ensure all data is synced
        wandb.finish()
        print(f"WandB: Final metrics logged to highway-distillation-baselines")

    train_env.close()
    return metadata


def train_multi_env_baseline(
    env_names: List[str],
    obs_type: str,
    config: Dict,
    device: str = "cuda"
) -> Dict:
    """
    Train a single model on multiple environments (generalist approach).

    Args:
        env_names: List of environment names
        obs_type: Observation type
        config: Training configuration
        device: Device for training

    Returns:
        Dictionary with training results
    """
    print(f"Training multi-env generalist with {obs_type} on: {env_names}")

    # Initialize WandB for multi-env training
    wandb_config = {
        "multi_env_training": True,
        "env_names": env_names,
        "obs_type": obs_type,
        "training_mode": "multi_env_generalist",
        "total_timesteps": config['timesteps'],
        "num_parallel_envs": config['num_envs'],
        "device": device,
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S")
    }

    wandb.init(
        project="highway-distillation-generalists",
        name=f"multi_generalist_{obs_type.lower()}_{int(time.time())}",
        config=wandb_config,
        notes=f"Training multi-environment generalist on {env_names} with {obs_type} observations"
    )

    # Create save directory
    obs_clean = obs_type.lower()
    save_dir = Path(f"outputs/models/multi_baseline/generalist_{obs_clean}")
    save_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(f"outputs/logs/multi_baseline/generalist_{obs_clean}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create multi-environment
    num_envs_per_type = max(1, config['num_envs'] // len(env_names))
    train_env, env_order = create_multi_env(
        env_names,
        obs_type,
        num_envs_per_type=num_envs_per_type,
        use_subprocess=True
    )

    # Determine policy type
    if obs_type == "GrayscaleObservation":
        policy_type = "CnnPolicy"
        policy_kwargs = {}
    else:
        policy_type = "MlpPolicy"
        policy_kwargs = {"net_arch": [256, 256]}

    # Create model
    model = PPO(
        policy_type,
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device=device,
        tensorboard_log=str(log_dir)
    )

    # Setup callbacks
    total_envs = num_envs_per_type * len(env_names)
    checkpoint_callback = CheckpointCallback(
        save_freq=config['checkpoint_freq'] // total_envs,
        save_path=str(save_dir),
        name_prefix=f"multi_generalist_{obs_clean}"
    )

    multi_env_callback = WandBCallback("multi_env", obs_type, verbose=1)

    # Train
    training_start_time = time.time()
    print(f"Training for {config['timesteps']} steps across {len(env_names)} environments...")
    print(f"WandB logging enabled - check highway-distillation-generalists project")

    model.learn(
        total_timesteps=config['timesteps'],
        callback=[checkpoint_callback, multi_env_callback]
    )

    training_duration = time.time() - training_start_time

    # Save final model
    model.save(str(save_dir / "final_model"))
    train_env.save(str(save_dir / "vec_normalize.pkl"))

    print(f"Saved multi-env generalist to: {save_dir}/final_model")

    # Evaluate on each environment
    eval_results = {}
    for env_name in env_names:
        eval_env = DummyVecEnv([lambda: create_highway_env(env_name, obs_type)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=20, deterministic=True
        )
        eval_results[env_name] = {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward)
        }
        eval_env.close()

        print(f"Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
    # Save metadata
    metadata = {
        "env_names": env_names,
        "obs_type": obs_type,
        "total_timesteps": config['timesteps'],
        "num_envs": total_envs,
        "training_time_seconds": training_duration,
        "eval_results": eval_results,
        "policy_type": policy_type,
        "training_completed": True,
        "wandb_project": "highway-distillation-generalists"
    }

    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Final wandb logging
    if wandb.run is not None:
        final_summary = {
            "multi_env_training_completed": True,
            "environments_trained": env_names,
            "per_env_performance": eval_results,
            "total_training_time_minutes": training_duration / 60,
            "model_saved_path": str(save_dir / "final_model.zip"),
            "final_timestamp": time.time()
        }
        wandb.log(final_summary)
        print(f"WandB: Final metrics logged to highway-distillation-generalists")

    train_env.close()
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline models on vanilla highway-env",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test all combinations
  python train_all_baselines.py --mode quick --all

  # Train specific environment
  python train_all_baselines.py --env highway-v0 --obs Lidar

  # Train multi-environment generalist
  python train_all_baselines.py --mode multi --obs Lidar

  # Full training all combinations
  python train_all_baselines.py --mode full --all
"""
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["quick", "standard", "full", "multi"],
        default="standard",
        help="Training mode: quick (50k), standard (150k), full (300k), multi (generalist)"
    )

    parser.add_argument(
        "--env",
        type=str,
        choices=["highway-v0", "merge-v0", "intersection-v0"],
        help="Single environment to train on"
    )

    parser.add_argument(
        "--obs",
        type=str,
        choices=["Kinematics", "Lidar", "GrayscaleObservation"],
        help="Single observation type to use"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all environment and observation combinations"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for training"
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        help="Override WandB project name"
    )

    args = parser.parse_args()

    # Select configuration
    if args.mode == "quick":
        config = QUICK_CONFIG
    elif args.mode == "standard":
        config = STANDARD_CONFIG
    elif args.mode == "full":
        config = FULL_CONFIG
    elif args.mode == "multi":
        config = STANDARD_CONFIG  # Use standard for multi-env

    print("="*70)
    print("BASELINE TRAINING SYSTEM")
    print("="*70)
    print(f"Mode: {args.mode.upper()}")
    print(f"Timesteps per model: {config['timesteps']:,}")
    print(f"Device: {args.device}")
    print("="*70)

    results = []

    # Multi-environment training mode
    if args.mode == "multi":
        env_names = ["highway-v0", "merge-v0", "intersection-v0"]
        obs_types = ["Lidar", "GrayscaleObservation"] if not args.obs else [args.obs]

        for obs_type in obs_types:
            try:
                result = train_multi_env_baseline(env_names, obs_type, config, args.device)
                results.append(result)
            except Exception as e:
                print(f"Failed to train multi-env {obs_type}: {e}")
                import traceback
                traceback.print_exc()

    # Single environment training mode
    else:
        # Determine environments to train
        if args.all:
            environments = ["highway-v0", "merge-v0", "intersection-v0"]
            observations = ["Lidar", "GrayscaleObservation"]
        elif args.env and args.obs:
            environments = [args.env]
            observations = [args.obs]
        elif args.env:
            environments = [args.env]
            observations = ["Lidar", "GrayscaleObservation"]
        elif args.obs:
            environments = ["highway-v0", "merge-v0", "intersection-v0"]
            observations = [args.obs]
        else:
            print("ERROR: Must specify --env, --obs, --all, or --mode multi")
            sys.exit(1)

        # Train all combinations
        total_combinations = len(environments) * len(observations)
        current = 0

        for env_name in environments:
            for obs_type in observations:
                current += 1
                print(f"\n[{current}/{total_combinations}] Training {env_name} + {obs_type}")

                try:
                    result = train_single_baseline(env_name, obs_type, config, args.device)
                    results.append(result)
                except Exception as e:
                    print(f"Failed to train {env_name} + {obs_type}: {e}")
                    import traceback
                    traceback.print_exc()

    # Print summary
    print("\n" + "="*70)
    print("BASELINE TRAINING COMPLETE")
    print("="*70)
    print(f"Successfully trained: {len(results)} models")
    print("\nResults:")
    for result in results:
        if "env_name" in result:
            print(f"  {result['env_name']} + {result['obs_type']}: "
                  f"{result['final_mean_reward']:.2f} ± {result['final_std_reward']:.2f}")        
        else:
            print(f"  Multi-env {result['obs_type']}: "
                  f"{len(result['eval_results'])} environments trained")
    print("="*70)


if __name__ == "__main__":
    main()
