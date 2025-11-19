#!/usr/bin/env python3
"""
Focused Baseline Training for WandB Charts

Trains a single baseline with optimized WandB logging for performance assessment.
Generates comprehensive charts showing convergence, rewards, losses, etc.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np

import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

import wandb
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style for WandB
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


class FocusedWandBCallback(BaseCallback):
    """
    Comprehensive WandB callback focused on performance assessment charts.
    """

    def __init__(self, env_name: str, obs_type: str, log_freq: int = 500, verbose: int = 1):
        super().__init__(verbose)
        self.env_name = env_name
        self.obs_type = obs_type
        self.log_freq = log_freq

        # Data storage for comprehensive logging
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0

        # PPO metrics storage
        self.value_losses = []
        self.policy_losses = []
        self.entropy_losses = []
        self.approx_kls = []
        self.clip_fractions = []
        self.learning_rates = []

        # Performance tracking
        self.best_reward = float('-inf')
        self.reward_history = []
        self.convergence_threshold = 0.95  # 95% of best reward

    def _on_step(self) -> bool:
        # Get environment data
        dones = self.locals.get('dones', [])
        rewards = self.locals.get('rewards', [])

        # Initialize per-environment tracking on first step
        if not hasattr(self, 'num_envs') or self.num_envs is None:
            self.num_envs = len(dones) if len(dones) > 0 else 1
            # Initialize tracking for each environment
            for env_idx in range(self.num_envs):
                setattr(self, f'current_reward_{env_idx}', 0.0)
                setattr(self, f'current_length_{env_idx}', 0)

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
                self.episode_rewards.append(current_reward)
                self.episode_lengths.append(current_length)
                self.episode_count += 1

                # Track best reward
                if current_reward > self.best_reward:
                    self.best_reward = current_reward

                # Reset for next episode in this environment
                setattr(self, f'current_reward_{env_idx}', 0.0)
                setattr(self, f'current_length_{env_idx}', 0)

        # Log comprehensive metrics at regular intervals
        if self.n_calls % self.log_freq == 0:
            metrics = {
                "timesteps": self.num_timesteps,
                "updates": self.n_calls,
                "env_name": self.env_name,
                "obs_type": self.obs_type,
                "episodes_completed": self.episode_count
            }

            # PPO training metrics
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                logs = self.model.logger.name_to_value
                for key in ['train/value_loss', 'train/policy_gradient_loss',
                           'train/entropy_loss', 'train/approx_kl', 'train/clip_fraction',
                           'train/learning_rate', 'train/n_updates']:
                    if key in logs:
                        clean_key = key.replace('train/', '')
                        metrics[clean_key] = logs[key]

                        # Store for trend analysis
                        if 'value_loss' in key:
                            self.value_losses.append(logs[key])
                        elif 'policy_gradient_loss' in key:
                            self.policy_losses.append(logs[key])
                        elif 'entropy_loss' in key:
                            self.entropy_losses.append(logs[key])
                        elif 'approx_kl' in key:
                            self.approx_kls.append(logs[key])
                        elif 'clip_fraction' in key:
                            self.clip_fractions.append(logs[key])
                        elif 'learning_rate' in key:
                            self.learning_rates.append(logs[key])

            # Episode statistics (rolling averages)
            if self.episode_rewards:
                recent_rewards = self.episode_rewards[-20:]  # Last 20 episodes
                recent_lengths = self.episode_lengths[-20:]

                metrics.update({
                    "episode_avg_reward": np.mean(recent_rewards),
                    "episode_std_reward": np.std(recent_rewards),
                    "episode_best_reward": np.max(recent_rewards) if recent_rewards else 0,
                    "episode_min_reward": np.min(recent_rewards) if recent_rewards else 0,
                    "episode_avg_length": np.mean(recent_lengths),
                    "episode_count": len(self.episode_rewards),
                })

                # Store reward history for convergence analysis
                self.reward_history.append(np.mean(recent_rewards))

            # Learning progress indicators
            if len(self.episode_rewards) > 10:
                early_avg = np.mean(self.episode_rewards[:10])
                recent_avg = np.mean(self.episode_rewards[-10:])
                improvement = recent_avg - early_avg

                metrics.update({
                    "learning_improvement": improvement,
                    "convergence_ratio": recent_avg / max(abs(early_avg), 1e-6) if early_avg != 0 else 0,
                    "reward_stability": 1.0 / (1.0 + np.std(self.episode_rewards[-20:])) if len(self.episode_rewards) >= 20 else 0
                })

            # Generate performance charts every 10k steps
            if self.n_calls % 10000 == 0 and self.episode_rewards:
                self._generate_performance_charts()

            # Log to wandb
            wandb.log(metrics, step=self.num_timesteps)

        return True

    def _generate_performance_charts(self):
        """Generate comprehensive performance charts for WandB."""
        if len(self.episode_rewards) < 10:
            return

        # 1. Learning Curve Chart
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Performance Analysis - {self.env_name} + {self.obs_type}', fontsize=14)

        # Episode rewards over time
        episodes = range(1, len(self.episode_rewards) + 1)
        rewards_smooth = self._smooth_data(self.episode_rewards, window=10)

        axes[0, 0].plot(episodes, self.episode_rewards, alpha=0.3, color='blue', label='Raw')
        axes[0, 0].plot(episodes, rewards_smooth, color='red', linewidth=2, label='Smoothed')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Reward distribution
        axes[0, 1].hist(self.episode_rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(np.mean(self.episode_rewards), color='red', linestyle='--',
                          label=f'Mean: {np.mean(self.episode_rewards):.2f}')
        axes[0, 1].axvline(np.median(self.episode_rewards), color='green', linestyle='--',
                          label=f'Median: {np.median(self.episode_rewards):.2f}')
        axes[0, 1].set_title('Reward Distribution')
        axes[0, 1].set_xlabel('Reward')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # PPO Loss Curves
        if self.value_losses and self.policy_losses:
            steps = range(len(self.value_losses))
            axes[1, 0].plot(steps, self.value_losses, label='Value Loss', color='orange')
            axes[1, 0].plot(steps, self.policy_losses, label='Policy Loss', color='purple')
            if self.entropy_losses:
                axes[1, 0].plot(steps, self.entropy_losses, label='Entropy Loss', color='green')
            axes[1, 0].set_title('PPO Training Losses')
            axes[1, 0].set_xlabel('Update Steps')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')

        # Convergence Analysis
        if len(self.reward_history) > 5:
            conv_steps = range(len(self.reward_history))
            axes[1, 1].plot(conv_steps, self.reward_history, marker='o', color='darkblue')
            axes[1, 1].set_title('Learning Convergence')
            axes[1, 1].set_xlabel('Measurement Point')
            axes[1, 1].set_ylabel('Avg Reward (20 episodes)')
            axes[1, 1].grid(True, alpha=0.3)

            # Add convergence line
            if len(self.reward_history) >= 3:
                # Simple linear trend
                x = np.arange(len(self.reward_history))
                slope = np.polyfit(x, self.reward_history, 1)[0]
                trend_color = 'green' if slope > 0 else 'red'
                axes[1, 1].text(0.05, 0.95, f'Trend: {"↗️" if slope > 0 else "↘️"}',
                               transform=axes[1, 1].transAxes, fontsize=12,
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor=trend_color, alpha=0.1))

        plt.tight_layout()

        # Log the comprehensive chart to WandB
        wandb.log({"performance_analysis_chart": wandb.Image(fig)}, step=self.num_timesteps)
        plt.close(fig)

        # 2. Additional detailed charts
        self._generate_detailed_charts()

    def _generate_detailed_charts(self):
        """Generate additional detailed performance charts."""

        # PPO Training Metrics Chart
        if self.approx_kls and self.clip_fractions:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # KL divergence and clip fraction
            steps = range(len(self.approx_kls))
            axes[0].plot(steps, self.approx_kls, label='Approx KL', color='blue')
            axes[0].axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='Target KL')
            axes[0].set_title('KL Divergence')
            axes[0].set_xlabel('Update Steps')
            axes[0].set_ylabel('KL')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Clip fraction
            axes[1].plot(steps, self.clip_fractions, color='orange')
            axes[1].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Target Clip')
            axes[1].set_title('Clip Fraction')
            axes[1].set_xlabel('Update Steps')
            axes[1].set_ylabel('Clip Fraction')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            wandb.log({"ppo_training_metrics": wandb.Image(fig)}, step=self.num_timesteps)
            plt.close(fig)

    def _smooth_data(self, data, window=10):
        """Simple moving average smoothing."""
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    def _on_training_end(self) -> None:
        """Log final training summary with comprehensive analysis."""
        if not self.episode_rewards:
            return

        # Calculate final statistics
        final_metrics = {
            "training_completed": True,
            "total_episodes": len(self.episode_rewards),
            "total_timesteps": self.num_timesteps,
            "final_avg_reward": np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards),
            "final_best_reward": np.max(self.episode_rewards),
            "final_worst_reward": np.min(self.episode_rewards),
            "reward_std_final": np.std(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.std(self.episode_rewards),
            "learning_stability": 1.0 / (1.0 + np.std(self.episode_rewards[-20:])) if len(self.episode_rewards) >= 20 else 0,

            # Convergence metrics
            "early_performance": np.mean(self.episode_rewards[:10]) if len(self.episode_rewards) >= 10 else 0,
            "mid_performance": np.mean(self.episode_rewards[len(self.episode_rewards)//2-5:len(self.episode_rewards)//2+5]) if len(self.episode_rewards) >= 10 else 0,
            "final_performance": np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else 0,

            # PPO final metrics
            "final_value_loss": self.value_losses[-1] if self.value_losses else 0,
            "final_policy_loss": self.policy_losses[-1] if self.policy_losses else 0,
            "final_entropy_loss": self.entropy_losses[-1] if self.entropy_losses else 0,

            "env_name": self.env_name,
            "obs_type": self.obs_type,
            "final_timestamp": time.time()
        }

        # Calculate improvement
        if len(self.episode_rewards) >= 20:
            early_avg = np.mean(self.episode_rewards[:10])
            late_avg = np.mean(self.episode_rewards[-10:])
            final_metrics["total_improvement"] = late_avg - early_avg
            final_metrics["improvement_percentage"] = ((late_avg - early_avg) / abs(early_avg)) * 100 if early_avg != 0 else 0

        wandb.log(final_metrics)

        # Generate final comprehensive report
        self._generate_final_report()

    def _generate_final_report(self):
        """Generate final comprehensive performance report."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Final Performance Report - {self.env_name} + {self.obs_type}', fontsize=16, fontweight='bold')

        # 1. Complete learning curve
        episodes = range(1, len(self.episode_rewards) + 1)
        axes[0, 0].plot(episodes, self.episode_rewards, alpha=0.6, color='lightblue', linewidth=1)
        axes[0, 0].plot(episodes, self._smooth_data(self.episode_rewards, 20), color='darkblue', linewidth=2)
        axes[0, 0].set_title('Complete Learning Curve')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Reward distribution with statistics
        axes[0, 1].hist(self.episode_rewards, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        mean_reward = np.mean(self.episode_rewards)
        median_reward = np.median(self.episode_rewards)
        axes[0, 1].axvline(mean_reward, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.2f}')
        axes[0, 1].axvline(median_reward, color='green', linestyle='--', linewidth=2, label=f'Median: {median_reward:.2f}')
        axes[0, 1].set_title('Final Reward Distribution')
        axes[0, 1].set_xlabel('Reward')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. PPO loss convergence
        if self.value_losses and self.policy_losses:
            steps = range(len(self.value_losses))
            axes[1, 0].plot(steps, self.value_losses, label='Value Loss', color='orange', alpha=0.7)
            axes[1, 0].plot(steps, self._smooth_data(self.value_losses, 5), color='orange', linewidth=2)
            axes[1, 0].plot(steps, self.policy_losses, label='Policy Loss', color='purple', alpha=0.7)
            axes[1, 0].plot(steps, self._smooth_data(self.policy_losses, 5), color='purple', linewidth=2)
            axes[1, 0].set_title('PPO Loss Convergence')
            axes[1, 0].set_xlabel('Update Steps')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')

        # 4. Performance improvement over time
        if len(self.episode_rewards) >= 30:
            window_size = 10
            rolling_means = [np.mean(self.episode_rewards[i:i+window_size])
                           for i in range(0, len(self.episode_rewards)-window_size+1, window_size)]
            windows = range(len(rolling_means))
            axes[1, 1].plot(windows, rolling_means, marker='o', color='darkgreen', linewidth=2)
            axes[1, 1].set_title('Performance Improvement Over Time')
            axes[1, 1].set_xlabel('Time Window')
            axes[1, 1].set_ylabel(f'Avg Reward ({window_size} episodes)')
            axes[1, 1].grid(True, alpha=0.3)

        # 5. Episode length analysis
        if self.episode_lengths:
            axes[2, 0].plot(range(len(self.episode_lengths)), self.episode_lengths,
                           alpha=0.6, color='lightcoral', linewidth=1)
            axes[2, 0].plot(range(len(self.episode_lengths)),
                           self._smooth_data(self.episode_lengths, 20),
                           color='darkred', linewidth=2)
            axes[2, 0].set_title('Episode Lengths')
            axes[2, 0].set_xlabel('Episode')
            axes[2, 0].set_ylabel('Length (steps)')
            axes[2, 0].grid(True, alpha=0.3)

        # 6. Summary statistics
        axes[2, 1].axis('off')
        summary_text = ".2f"".2f"".1f"".2f"".2f"f"""
PERFORMANCE SUMMARY

Total Episodes: {len(self.episode_rewards)}
Total Timesteps: {self.num_timesteps:,}

Rewards:
  Mean: {np.mean(self.episode_rewards):.2f}
  Std: {np.std(self.episode_rewards):.2f}
  Best: {np.max(self.episode_rewards):.2f}
  Worst: {np.min(self.episode_rewards):.2f}
  Median: {np.median(self.episode_rewards):.2f}

Convergence:
  Early (first 10): {np.mean(self.episode_rewards[:10]):.2f}
  Final (last 10): {np.mean(self.episode_rewards[-10:]):.2f}
  Improvement: {np.mean(self.episode_rewards[-10:]) - np.mean(self.episode_rewards[:10]):.2f}

Stability:
  Recent Std: {np.std(self.episode_rewards[-20:]):.2f}
  Consistency Score: {1.0 / (1.0 + np.std(self.episode_rewards[-20:])):.3f}
"""
        axes[2, 1].text(0.05, 0.95, summary_text, transform=axes[2, 1].transAxes,
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()

        # Log the final comprehensive report
        wandb.log({"final_performance_report": wandb.Image(fig)}, step=self.num_timesteps)
        plt.close(fig)


def create_highway_env(env_name: str, obs_type: str = "Lidar"):
    """Create a highway environment with proper configuration."""
    env = gym.make(env_name, render_mode=None)
    unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env

    # Configure based on observation type
    if obs_type == "Lidar":
        unwrapped_env.configure({
            "observation": {
                "type": "LidarObservation",
                "cells": 32,
                "maximum_range": 50.0
            }
        })
    elif obs_type == "GrayscaleObservation":
        unwrapped_env.configure({
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (64, 64),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140]
            }
        })
    elif obs_type == "Kinematics":
        unwrapped_env.configure({
            "observation": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                "vehicles_count": 15,
                "normalize": True
            }
        })

    env.reset()
    return env


def train_focused_baseline(env_name: str = "highway-v0", obs_type: str = "Lidar",
                          timesteps: int = 100000, device: str = "cuda"):
    """Train a single baseline with comprehensive WandB logging."""

    print("="*70)
    print(f"FOCUSED BASELINE TRAINING: {env_name.upper()} + {obs_type.upper()}")
    print("="*70)
    print(f"Timesteps: {timesteps:,}")
    print(f"Device: {device}")
    print("WandB Project: highway-distillation-baselines")
    print("="*70)

    # Initialize WandB with comprehensive config
    wandb_config = {
        "training_type": "focused_baseline",
        "env_name": env_name,
        "obs_type": obs_type,
        "total_timesteps": timesteps,
        "device": device,
        "ppo_config": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5
        },
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S")
    }

    wandb.init(
        project="highway-distillation-baselines",
        name=f"focused_{env_name}_{obs_type.lower()}_{int(time.time())}",
        config=wandb_config,
        notes=f"Focused baseline training on {env_name} with {obs_type} for comprehensive performance analysis"
    )

    # Create directories
    env_clean = env_name.replace('-', '_')
    obs_clean = obs_type.lower()
    save_dir = Path(f"outputs/models/baseline/{env_clean}_{obs_clean}")
    save_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(f"outputs/logs/baseline/{env_clean}_{obs_clean}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    (log_dir / "monitor").mkdir(parents=True, exist_ok=True)

    def make_env():
        def _init():
            env = create_highway_env(env_name, obs_type)
            return Monitor(env, filename=str(log_dir / "monitor" / f"env_{os.getpid()}"))
        return _init

    # Create vectorized environment with 4 parallel envs
    train_env = DummyVecEnv([make_env() for _ in range(4)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Determine policy type
    if obs_type == "GrayscaleObservation":
        policy_type = "CnnPolicy"
        policy_kwargs = {}
    else:
        policy_type = "MlpPolicy"
        policy_kwargs = {"net_arch": [256, 256]}

    # Create PPO model
    model = PPO(
        policy_type,
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
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
        save_freq=25000,  # Save every 25k steps
        save_path=str(save_dir),
        name_prefix=f"focused_{env_clean}_{obs_clean}"
    )

    # Comprehensive WandB callback
    wandb_callback = FocusedWandBCallback(env_name, obs_type, log_freq=500)

    # Start training
    training_start_time = time.time()

    print(f"Starting training for {timesteps:,} timesteps...")
    print("Comprehensive WandB logging enabled - check dashboard for live charts!")

    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback, wandb_callback]
    )

    training_duration = time.time() - training_start_time

    # Save final model
    model.save(str(save_dir / "final_model"))
    train_env.save(str(save_dir / "vec_normalize.pkl"))

    print(f"Training completed in {training_duration:.2f} seconds")
    # Evaluate final model
    print("Running final evaluation...")
    mean_reward, std_reward = evaluate_policy(
        model, train_env, n_eval_episodes=50, deterministic=True
    )

    print(f"Training completed in {training_duration:.2f} seconds")
    # Save metadata
    metadata = {
        "env_name": env_name,
        "obs_type": obs_type,
        "total_timesteps": timesteps,
        "training_time_seconds": training_duration,
        "final_mean_reward": float(mean_reward),
        "final_std_reward": float(std_reward),
        "policy_type": policy_type,
        "training_completed": True,
        "wandb_project": "highway-distillation-baselines",
        "parallel_envs": 4,
        "ppo_hyperparams": wandb_config["ppo_config"]
    }

    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Final WandB logging
    final_summary = {
        "baseline_training_completed": True,
        "final_evaluation_mean_reward": float(mean_reward),
        "final_evaluation_std_reward": float(std_reward),
        "total_training_time_minutes": training_duration / 60,
        "model_save_path": str(save_dir / "final_model.zip"),
        "evaluation_episodes": 50,
        "final_timestamp": time.time()
    }

    wandb.log(final_summary)

    print(f"WandB: Final results logged to highway-distillation-baselines")
    print(f"Dashboard: https://wandb.ai/[your-username]/highway-distillation-baselines")

    train_env.close()

    return metadata


def main():
    """Main function to run focused baseline training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train focused baseline with comprehensive WandB charts")
    parser.add_argument("--env", type=str, default="highway-v0",
                       choices=["highway-v0", "merge-v0", "intersection-v0"],
                       help="Environment to train on")
    parser.add_argument("--obs", type=str, default="Lidar",
                       choices=["Lidar", "GrayscaleObservation", "Kinematics"],
                       help="Observation type")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Total training timesteps")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"], help="Training device")

    args = parser.parse_args()

    try:
        metadata = train_focused_baseline(
            env_name=args.env,
            obs_type=args.obs,
            timesteps=args.timesteps,
            device=args.device
        )

        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("📊 WandB Dashboard: Check for comprehensive performance charts")
        print("📈 Charts include: Learning curves, loss convergence, reward distributions")
        print("🎯 Performance metrics: Convergence analysis, stability scores")
        print("🔧 PPO internals: Value/policy losses, KL divergence, clip fractions")
        print("="*70)

    except Exception as e:
        print(f"\n[FAILED] Training failed: {e}")
        import traceback
        traceback.print_exc()

        # Log error to WandB if initialized
        if wandb.run is not None:
            wandb.log({
                "training_failed": True,
                "error_message": str(e),
                "error_timestamp": time.time()
            })

        sys.exit(1)


if __name__ == "__main__":
    main()
