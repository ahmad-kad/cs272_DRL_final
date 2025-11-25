#!/usr/bin/env python3
"""
Fine-tune Intersection Performance
Continues training from the most recent advanced curriculum model with enhanced rewards
and domain randomization focused specifically on intersection scenarios.
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import wandb
from datetime import datetime

class IntersectionFineTuner:
    """Fine-tunes intersection performance with enhanced rewards and domain randomization."""

    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path
        self.start_time = time.time()

    def create_intersection_env(self, episode_seed: int = None):
        """Create intersection environment with enhanced rewards and domain randomization."""

        # Set random seed for reproducible randomization
        if episode_seed is not None:
            random.seed(episode_seed)
            np.random.seed(episode_seed)

        env = gym.make("intersection-v0", render_mode=None)
        unwrapped_env = env.unwrapped

        # Enhanced intersection configuration
        base_config = {
            "env_name": "intersection-v0",
            "observation": {
                "type": "LidarObservation",
                "cells": 32,
                "row_anchor": [0.5, 0.5],
                "features": ["presence", "distance", "speed"],
                "features_range": {"distance": [0, 50], "speed": [-30, 30]}
            },
            "action": {"type": "DiscreteMetaAction"},
            "duration": 35,  # Shorter episodes for more learning iterations
            "collision_reward": -25.0,  # High collision penalty
            "high_speed_reward": 0.3,   # Conservative speed reward
            "reward_speed_range": [10, 20],  # Very conservative speeds
            "arrived_reward": 6.0,      # High success bonus
            "progress_reward": 0.25,    # Reward for continuing safely
            "safe_distance_reward": 0.6, # Reward for maintaining distance
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "vehicles_count": 15,       # Base traffic density
            "initial_vehicle_count": 10, # Base initial traffic
        }

        # Apply domain randomization
        config = base_config.copy()

        # Randomize traffic density
        base_vehicles = config.get("vehicles_count", 15)
        config["vehicles_count"] = random.randint(max(10, base_vehicles-3), base_vehicles+3)

        base_initial = config.get("initial_vehicle_count", 10)
        config["initial_vehicle_count"] = random.randint(max(6, base_initial-2), base_initial+2)

        # Randomize timing (affects physics and NPC behavior)
        config["simulation_frequency"] = random.randint(12, 18)

        # Randomize reward weights slightly for robustness
        config["collision_reward"] = config.get("collision_reward", -25.0) * random.uniform(0.95, 1.05)
        config["arrived_reward"] = config.get("arrived_reward", 6.0) * random.uniform(0.95, 1.05)
        config["progress_reward"] = config.get("progress_reward", 0.25) * random.uniform(0.9, 1.1)

        # Remove env_name before configuring
        del config["env_name"]
        unwrapped_env.configure(config)

        env.reset()
        return env

    def create_finetune_callback(self, save_dir: Path):
        """Create callback for fine-tuning with detailed intersection metrics."""

        class IntersectionFineTuneCallback(BaseCallback):
            def __init__(self, save_dir: Path, log_freq: int = 250):
                super().__init__(verbose=0)
                self.save_dir = save_dir
                self.log_freq = log_freq
                self.episode_rewards = []
                self.episode_lengths = []
                self.collision_count = []
                self.success_count = []
                self.start_time = time.time()

                # Detailed intersection metrics
                self.episodes_completed = 0
                self.total_collisions = 0
                self.total_successes = 0
                self.episode_times = []

            def _on_step(self) -> bool:
                # Collect episode data
                dones = self.locals.get('dones', np.array([]))
                rewards = self.locals.get('rewards', np.array([]))

                if len(dones) > 0 and len(rewards) > 0:
                    for env_idx in range(len(dones)):
                        if env_idx >= len(rewards):
                            continue

                        env_key = f'env_{env_idx}'
                        if not hasattr(self, env_key):
                            setattr(self, env_key, {
                                'current_reward': 0.0,
                                'current_length': 0,
                                'collisions': 0,
                                'start_time': time.time()
                            })

                        env_data = getattr(self, env_key)
                        env_data['current_reward'] += rewards[env_idx]
                        env_data['current_length'] += 1

                        # Check for collisions
                        infos = self.locals.get('infos', [])
                        if env_idx < len(infos) and infos[env_idx]:
                            if infos[env_idx].get('crashed', False):
                                env_data['collisions'] += 1

                        # Episode completed
                        if dones[env_idx]:
                            episode_start = env_data['start_time']
                            episode_time = time.time() - episode_start

                            self.episode_rewards.append(env_data['current_reward'])
                            self.episode_lengths.append(env_data['current_length'])
                            self.episode_times.append(episode_time)
                            self.collision_count.append(env_data['collisions'])

                            # Success criteria: positive reward and no collisions
                            success = (env_data['current_reward'] > 0 and env_data['collisions'] == 0)
                            self.success_count.append(1 if success else 0)

                            self.episodes_completed += 1
                            self.total_collisions += env_data['collisions']
                            self.total_successes += (1 if success else 0)

                            # Reset for next episode
                            env_data['current_reward'] = 0.0
                            env_data['current_length'] = 0
                            env_data['collisions'] = 0
                            env_data['start_time'] = time.time()

                # Log at regular intervals
                if self.n_calls % self.log_freq == 0 and len(self.episode_rewards) > 0:
                    recent_rewards = self.episode_rewards[-20:]
                    recent_lengths = self.episode_lengths[-20:]
                    recent_collisions = self.collision_count[-20:]
                    recent_successes = self.success_count[-20:]
                    recent_times = self.episode_times[-20:]

                    metrics = {
                        "fine_tune_timesteps": self.num_timesteps,
                        "episodes_completed": self.episodes_completed,

                        # Performance metrics
                        "avg_reward": np.mean(recent_rewards),
                        "reward_std": np.std(recent_rewards) if len(recent_rewards) > 1 else 0,
                        "avg_episode_length": np.mean(recent_lengths),
                        "avg_episode_time": np.mean(recent_times),
                        "crash_rate": np.mean(recent_collisions),
                        "success_rate": np.mean(recent_successes),

                        # Cumulative metrics
                        "total_episodes": len(self.episode_rewards),
                        "overall_crash_rate": self.total_collisions / max(1, len(self.episode_rewards)),
                        "overall_success_rate": self.total_successes / max(1, len(self.episode_rewards)),

                        # Learning progress
                        "learning_progress": np.mean(recent_rewards) - (np.mean(self.episode_rewards[:10]) if len(self.episode_rewards) > 10 else 0),

                        # Safety metrics
                        "episodes_without_crash": sum(1 for c in recent_collisions if c == 0),
                        "avg_crashes_per_episode": np.mean(recent_collisions),
                    }

                    # PPO training metrics
                    if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                        logs = self.model.logger.name_to_value
                        ppo_metrics = {
                            'ppo_value_loss': 'train/value_loss',
                            'ppo_policy_loss': 'train/policy_gradient_loss',
                            'ppo_entropy_loss': 'train/entropy_loss',
                            'ppo_approx_kl': 'train/approx_kl',
                            'ppo_clip_fraction': 'train/clip_fraction',
                            'ppo_learning_rate': 'train/learning_rate'
                        }
                        for metric_name, log_key in ppo_metrics.items():
                            if log_key in logs:
                                metrics[metric_name] = logs[log_key]

                    wandb.log(metrics, step=self.num_timesteps)

                    # Progress logging
                    if self.n_calls % 1000 == 0:  # Log every 1000 steps
                        print(f"[FINE-TUNE] Step {self.num_timesteps:,} | "
                              f"Success: {metrics['success_rate']:.1%} | "
                              f"Crash: {metrics['crash_rate']:.1%} | "
                              f"Avg Reward: {metrics['avg_reward']:.2f}")

                return True

        return IntersectionFineTuneCallback(save_dir, log_freq=250)

    def fine_tune_intersection(self, total_timesteps: int = 50000, learning_rate: float = 1e-5):
        """Fine-tune the model on intersection scenarios."""

        print(">>> INTERSECTION FINE-TUNING")
        print("=" * 80)
        print(f"Base Model: {self.base_model_path}")
        print(f"Fine-tuning Steps: {total_timesteps:,}")
        print(f"Learning Rate: {learning_rate}")
        print("Focus: Enhanced intersection performance with domain randomization")
        print("=" * 80)

        # Load base model
        if not os.path.exists(self.base_model_path):
            print(f"[ERROR] Base model not found: {self.base_model_path}")
            return None

        print(f"Loading base model...")
        base_env = self.create_intersection_env()
        model = PPO.load(self.base_model_path, env=base_env, device='cpu')

        # Update model for fine-tuning with lower learning rate
        model.learning_rate = learning_rate
        print(f"Updated learning rate to: {learning_rate}")

        # Setup directories
        timestamp = int(time.time())
        save_dir = Path(f"outputs/models/finetune_intersection_{timestamp}")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=total_timesteps // 4,
            save_path=str(save_dir),
            name_prefix="finetune_intersection"
        )

        finetune_callback = self.create_finetune_callback(save_dir)
        callback_list = CallbackList([checkpoint_callback, finetune_callback])

        # Initialize wandb
        run = wandb.init(
            project="highway-foundation-v2",
            name=f"finetune_intersection_{timestamp}",
            config={
                "fine_tune_type": "intersection_specialization",
                "base_model": self.base_model_path,
                "total_timesteps": total_timesteps,
                "learning_rate": learning_rate,
                "enhanced_rewards": True,
                "domain_randomization": True,
                "focus": "intersection_performance"
            }
        )

        try:
            print(f"\nStarting fine-tuning for {total_timesteps:,} timesteps...")
            print("Each episode uses different randomized intersection conditions for robustness.")

            # Fine-tuning with domain randomization
            trained_timesteps = 0
            check_interval = 5000

            while trained_timesteps < total_timesteps:
                remaining_steps = min(check_interval, total_timesteps - trained_timesteps)

                # Create fresh environment with randomization for each segment
                episode_seed = int(time.time() * 1000) % 1000000
                env = self.create_intersection_env(episode_seed)
                model.set_env(env)

                model.learn(
                    total_timesteps=remaining_steps,
                    callback=callback_list,
                    reset_num_timesteps=False
                )

                trained_timesteps += remaining_steps

            fine_tune_time = time.time() - self.start_time

            # Save final fine-tuned model
            final_model_path = save_dir / "finetune_intersection_final.zip"
            model.save(str(final_model_path))

            # Save metadata
            metadata = {
                "base_model": self.base_model_path,
                "fine_tune_timesteps": trained_timesteps,
                "fine_tune_time_seconds": fine_tune_time,
                "learning_rate": learning_rate,
                "final_model_path": str(final_model_path),
                "enhanced_rewards": True,
                "domain_randomization": True,
                "focus": "intersection_performance",
                "timestamp": datetime.now().isoformat()
            }

            with open(save_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"\n[SUCCESS] FINE-TUNING COMPLETED!")
            print("=" * 80)
            print(f"Base Model: {os.path.basename(self.base_model_path)}")
            print(f"Fine-tuned Model: {final_model_path}")
            print(f"Training Time: {fine_tune_time:.1f} seconds")
            print(f"Total Steps: {trained_timesteps:,}")
            print("Features:")
            print("  + Enhanced intersection rewards (-25 collision, +6 success)")
            print("  + Domain randomization for robustness")
            print("  + Conservative speed limits (10-20 km/h)")
            print("  + Episode-level environment variation")
            print("=" * 80)

            wandb.log({
                "fine_tune_completed": True,
                "total_fine_tune_time": fine_tune_time,
                "final_model_path": str(final_model_path)
            })

            env.close()
            return str(final_model_path)

        except Exception as e:
            print(f"[ERROR] Fine-tuning failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            wandb.finish()

def find_most_recent_model():
    """Find the most recent trained model for fine-tuning."""

    # Check advanced curriculum models first (most recent)
    advanced_dir = Path("outputs/models/curriculum_advanced")
    if advanced_dir.exists():
        # Look for expert_intersection first, then hard_intersection
        for phase in ["expert_intersection", "hard_intersection", "medium_intersection"]:
            phase_dir = advanced_dir / phase
            if phase_dir.exists():
                model_files = list(phase_dir.glob("*.zip"))
                if model_files:
                    # Get most recent by modification time
                    most_recent = max(model_files, key=lambda x: x.stat().st_mtime)
                    print(f"Found most recent advanced curriculum model: {most_recent}")
                    return str(most_recent)

    # Fall back to other curriculum models
    curriculum_dir = Path("outputs/models/curriculum")
    if curriculum_dir.exists():
        model_files = list(curriculum_dir.rglob("*.zip"))
        if model_files:
            most_recent = max(model_files, key=lambda x: x.stat().st_mtime)
            print(f"Found most recent curriculum model: {most_recent}")
            return str(most_recent)

    # Last resort: multi-env models
    multi_env_dir = Path("outputs/models/multi_env")
    if multi_env_dir.exists():
        model_files = list(multi_env_dir.rglob("*.zip"))
        if model_files:
            most_recent = max(model_files, key=lambda x: x.stat().st_mtime)
            print(f"Found most recent multi-env model: {most_recent}")
            return str(most_recent)

    return None

def main():
    """Main fine-tuning execution."""

    print(">>> INTERSECTION FINE-TUNING")
    print("=" * 80)

    # Find the most recent model
    base_model_path = find_most_recent_model()

    if not base_model_path:
        print("[ERROR] No trained models found for fine-tuning!")
        print("Please run curriculum training first.")
        return

    print(f"[BASE MODEL] {base_model_path}")

    # Create fine-tuner and run
    fine_tuner = IntersectionFineTuner(base_model_path)

    # Fine-tune for 50k steps with very low learning rate for stability
    final_model_path = fine_tuner.fine_tune_intersection(
        total_timesteps=50000,
        learning_rate=1e-5  # Very low LR to avoid catastrophic forgetting
    )

    if final_model_path:
        print(f"\n[SUCCESS] Fine-tuning completed successfully!")
        print(f"Final Model: {final_model_path}")
        print("\nTo evaluate: python evaluate_enhanced_curriculum.py")
        print("(Update the model path in the evaluation script)")

if __name__ == "__main__":
    main()
