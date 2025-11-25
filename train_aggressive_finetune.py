#!/usr/bin/env python3
"""
Aggressive Intersection Fine-tuning
More intensive training with refined reward structures and longer duration.
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

class AggressiveIntersectionFinetuner:
    """Aggressive fine-tuning with refined rewards and longer training."""

    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path
        self.start_time = time.time()

    def create_intersection_env(self, episode_seed: int = None):
        """Create intersection environment with refined reward structure."""

        # Set random seed for reproducible randomization
        if episode_seed is not None:
            random.seed(episode_seed)
            np.random.seed(episode_seed)

        env = gym.make("intersection-v0", render_mode=None)
        unwrapped_env = env.unwrapped

        # Refined intersection configuration with better reward structure
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
            "duration": 40,  # Slightly longer episodes
            # Refined reward structure for better intersection behavior
            "collision_reward": -20.0,      # Less severe than -25 (encourage risk-taking)
            "high_speed_reward": 0.4,       # Moderate speed reward
            "reward_speed_range": [12, 22], # Slightly higher speed range
            "arrived_reward": 8.0,          # Higher success bonus
            "progress_reward": 0.3,         # Reward for safe continuation
            "safe_distance_reward": 0.5,    # Distance maintenance reward
            # New reward components
            "lane_change_reward": 0.2,      # Reward proper lane changes
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "vehicles_count": 15,
            "initial_vehicle_count": 10,
        }

        # Apply domain randomization
        config = base_config.copy()

        # Randomize traffic density (wider range for more challenge)
        base_vehicles = config.get("vehicles_count", 15)
        config["vehicles_count"] = random.randint(max(12, base_vehicles-4), base_vehicles+2)

        base_initial = config.get("initial_vehicle_count", 10)
        config["initial_vehicle_count"] = random.randint(max(8, base_initial-3), base_initial+1)

        # Randomize timing
        config["simulation_frequency"] = random.randint(14, 18)

        # Randomize rewards slightly for robustness
        config["collision_reward"] = config.get("collision_reward", -20.0) * random.uniform(0.95, 1.05)
        config["arrived_reward"] = config.get("arrived_reward", 8.0) * random.uniform(0.95, 1.05)

        # Remove env_name before configuring
        del config["env_name"]
        unwrapped_env.configure(config)

        env.reset()
        return env

    def create_aggressive_callback(self, save_dir: Path):
        """Create callback with aggressive fine-tuning metrics and early stopping."""

        class AggressiveFineTuneCallback(BaseCallback):
            def __init__(self, save_dir: Path, log_freq: int = 200):
                super().__init__(verbose=0)
                self.save_dir = save_dir
                self.log_freq = log_freq
                self.episode_rewards = []
                self.episode_lengths = []
                self.collision_count = []
                self.success_count = []
                self.start_time = time.time()

                # Performance tracking for early stopping
                self.best_performance = 0.0
                self.no_improvement_count = 0
                self.patience = 10  # Stop if no improvement for 10 evaluation periods

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
                            episode_time = time.time() - env_data['start_time']

                            self.episode_rewards.append(env_data['current_reward'])
                            self.episode_lengths.append(env_data['current_length'])
                            self.collision_count.append(env_data['collisions'])

                            # Success criteria: positive reward and no collisions
                            success = (env_data['current_reward'] > 0 and env_data['collisions'] == 0)
                            self.success_count.append(1 if success else 0)

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

                    metrics = {
                        "aggressive_finetune_timesteps": self.num_timesteps,
                        "episodes_completed": len(self.episode_rewards),

                        # Performance metrics
                        "avg_reward": np.mean(recent_rewards),
                        "reward_std": np.std(recent_rewards) if len(recent_rewards) > 1 else 0,
                        "avg_episode_length": np.mean(recent_lengths),
                        "crash_rate": np.mean(recent_collisions),
                        "success_rate": np.mean(recent_successes),

                        # Cumulative metrics
                        "total_episodes": len(self.episode_rewards),
                        "overall_crash_rate": self.collision_count.count(1) / max(1, len(self.episode_rewards)),
                        "overall_success_rate": sum(self.success_count) / max(1, len(self.episode_rewards)),

                        # Learning progress
                        "learning_progress": np.mean(recent_rewards) - (np.mean(self.episode_rewards[:10]) if len(self.episode_rewards) > 10 else 0),
                    }

                    # Calculate performance score
                    crash_rate = metrics['crash_rate']
                    success_rate = metrics['success_rate']
                    avg_reward = metrics['avg_reward']

                    # Intersection performance score
                    performance_score = min(1.0, max(0.0, (avg_reward + 30) / 30))
                    metrics['performance_score'] = performance_score

                    # Check for improvement
                    if performance_score > self.best_performance:
                        self.best_performance = performance_score
                        self.no_improvement_count = 0
                        metrics['new_best_performance'] = True
                    else:
                        self.no_improvement_count += 1
                        metrics['no_improvement_count'] = self.no_improvement_count

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
                    if self.n_calls % 1000 == 0:
                        print(f"[AGGRESSIVE FINETUNE] Step {self.num_timesteps:,} | "
                              f"Success: {metrics['success_rate']:.1%} | "
                              f"Crash: {metrics['crash_rate']:.1%} | "
                              f"Avg Reward: {metrics['avg_reward']:.2f} | "
                              f"Performance: {performance_score:.3f}")

                return True

        return AggressiveFineTuneCallback(save_dir, log_freq=200)

    def aggressive_finetune(self, total_timesteps: int = 100000, learning_rate: float = 5e-5):
        """Aggressive fine-tuning with refined rewards and longer training."""

        print("AGGRESSIVE INTERSECTION FINE-TUNING")
        print("=" * 80)
        print(f"Base Model: {self.base_model_path}")
        print(f"Aggressive Fine-tuning: {total_timesteps:,} timesteps")
        print(f"Learning Rate: {learning_rate} (higher than conservative 1e-5)")
        print("Features:")
        print("  + Refined reward structure (-20 collision, +8 success)")
        print("  + Longer episodes (40 timesteps)")
        print("  + Higher speed ranges (12-22 km/h)")
        print("  + Enhanced domain randomization")
        print("  + Progress and safety rewards")
        print("=" * 80)

        # Load base model
        if not os.path.exists(self.base_model_path):
            print(f"[ERROR] Base model not found: {self.base_model_path}")
            return None

        print("Loading base model...")
        base_env = self.create_intersection_env()
        model = PPO.load(self.base_model_path, env=base_env, device='cpu')

        # Update model for aggressive fine-tuning
        model.learning_rate = learning_rate
        print(f"Updated learning rate to: {learning_rate}")

        # Setup directories
        timestamp = int(time.time())
        save_dir = Path(f"outputs/models/aggressive_finetune_{timestamp}")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=total_timesteps // 5,  # More frequent checkpoints
            save_path=str(save_dir),
            name_prefix="aggressive_finetune"
        )

        aggressive_callback = self.create_aggressive_callback(save_dir)
        callback_list = CallbackList([checkpoint_callback, aggressive_callback])

        # Initialize wandb
        run = wandb.init(
            project="highway-foundation-v2",
            name=f"aggressive_finetune_{timestamp}",
            config={
                "fine_tune_type": "aggressive_intersection_specialization",
                "base_model": self.base_model_path,
                "total_timesteps": total_timesteps,
                "learning_rate": learning_rate,
                "refined_rewards": True,
                "domain_randomization": True,
                "extended_episodes": True,
                "focus": "intersection_performance_maximization"
            }
        )

        try:
            print(f"\nStarting aggressive fine-tuning for {total_timesteps:,} timesteps...")
            print("More intensive training with refined reward structures for intersection mastery.")

            # Aggressive fine-tuning with enhanced domain randomization
            trained_timesteps = 0
            check_interval = 5000  # More frequent environment changes

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

            aggressive_time = time.time() - self.start_time

            # Save final aggressively fine-tuned model
            final_model_path = save_dir / "aggressive_finetune_final.zip"
            model.save(str(final_model_path))

            # Save metadata
            metadata = {
                "base_model": self.base_model_path,
                "aggressive_finetune_timesteps": trained_timesteps,
                "aggressive_finetune_time_seconds": aggressive_time,
                "learning_rate": learning_rate,
                "final_model_path": str(final_model_path),
                "refined_rewards": True,
                "domain_randomization": True,
                "extended_episodes": True,
                "enhanced_traffic_randomization": True,
                "focus": "intersection_performance_maximization",
                "timestamp": datetime.now().isoformat()
            }

            with open(save_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            print("\n[SUCCESS] AGGRESSIVE FINE-TUNING COMPLETED!")
            print("=" * 80)
            print(f"Base Model: {os.path.basename(self.base_model_path)}")
            print(f"Aggressively Fine-tuned Model: {final_model_path}")
            print(f"Training Time: {aggressive_time:.1f} seconds")
            print(f"Total Steps: {trained_timesteps:,}")
            print("Aggressive Features:")
            print("  + Higher learning rate (5e-5) for faster adaptation")
            print("  + Refined reward structure for intersection behavior")
            print("  + Extended episodes (40 timesteps) for learning")
            print("  + Enhanced domain randomization")
            print("  + Progress and safety reward incentives")
            print("  + More frequent environment variation")
            print("=" * 80)

            wandb.log({
                "aggressive_fine_tune_completed": True,
                "total_aggressive_fine_tune_time": aggressive_time,
                "final_model_path": str(final_model_path)
            })

            env.close()
            return str(final_model_path)

        except Exception as e:
            print(f"[ERROR] Aggressive fine-tuning failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            wandb.finish()

def find_best_base_model():
    """Find the best available model for aggressive fine-tuning."""

    # Priority: Fine-tuned > Advanced Curriculum > Multi-env
    candidates = [
        ("outputs/models/finetune_intersection_*/finetune_intersection_final.zip", "Fine-tuned"),
        ("outputs/models/curriculum_advanced/*/advanced_curriculum_*_early_progression.zip", "Advanced Curriculum"),
        ("outputs/models/multi_env/*/final_model.zip", "Multi-env"),
    ]

    for pattern, model_type in candidates:
        import glob
        matches = glob.glob(pattern)
        if matches:
            # Get most recent by modification time
            best_match = max(matches, key=os.path.getmtime)
            print(f"Selected {model_type} model: {best_match}")
            return best_match

    return None

def main():
    """Main aggressive fine-tuning execution."""

    print("AGGRESSIVE INTERSECTION FINE-TUNING")
    print("=" * 80)

    # Find the best base model
    base_model_path = find_best_base_model()

    if not base_model_path:
        print("[ERROR] No suitable base models found!")
        print("Please run curriculum training or fine-tuning first.")
        return

    print(f"[BASE MODEL] {base_model_path}")

    # Create aggressive fine-tuner
    fine_tuner = AggressiveIntersectionFinetuner(base_model_path)

    # Aggressive fine-tuning: higher LR, longer training
    final_model_path = fine_tuner.aggressive_finetune(
        total_timesteps=100000,  # Double the previous training
        learning_rate=5e-5       # 5x higher than conservative approach
    )

    if final_model_path:
        print(f"\n[SUCCESS] Aggressive fine-tuning completed!")
        print(f"Final Model: {final_model_path}")
        print("\nTo evaluate: python evaluate_finetuned_model.py")
        print("(Update the model path in the evaluation script)")

if __name__ == "__main__":
    main()
