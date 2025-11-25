#!/usr/bin/env python3
"""
EWC (Elastic Weight Consolidation) Fine-tuning for Intersection Performance
Prevents catastrophic forgetting by protecting important weights from curriculum training.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
from datetime import datetime
import copy

class EWCFinetuner:
    """
    Elastic Weight Consolidation for fine-tuning without catastrophic forgetting.

    EWC protects important parameters learned during curriculum training by adding
    a regularization term that penalizes changes to parameters important for
    the original task (highway/merge performance).
    """

    def __init__(self, base_model_path: str, ewc_lambda: float = 1000.0):
        self.base_model_path = base_model_path
        self.ewc_lambda = ewc_lambda  # Strength of EWC regularization

        # Load base model
        print(f"Loading base model: {base_model_path}")
        self.base_env = self.create_base_env()
        self.base_model = PPO.load(base_model_path, env=self.base_env, device='cpu')

        # Store original parameters (theta_0)
        self.theta_0 = {}
        for name, param in self.base_model.policy.named_parameters():
            if param.requires_grad:
                self.theta_0[name] = param.data.clone()

        # Fisher Information Matrix (F)
        self.fisher_information = {}

        # EWC regularization will be computed after FIM calculation

    def create_base_env(self):
        """Create environment matching the base model's training."""
        env = gym.make("highway-v0", render_mode=None)
        unwrapped_env = env.unwrapped

        config = {
            "observation": {
                "type": "LidarObservation",
                "cells": 32,
                "row_anchor": [0.5, 0.5],
                "features": ["presence", "distance", "speed"],
                "features_range": {"distance": [0, 50], "speed": [-30, 30]}
            },
            "action": {"type": "DiscreteMetaAction"},
            "duration": 40,
            "collision_reward": -20.0,
            "right_lane_reward": 0.3,
            "high_speed_reward": 0.6,
            "reward_speed_range": [20, 30],
            "lane_change_reward": 0.1,
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "vehicles_count": 12,
            "lanes_count": 4,
        }

        unwrapped_env.configure(config)
        env.reset()
        return env

    def create_intersection_env(self, episode_seed: int = None):
        """Create intersection environment with refined rewards for fine-tuning."""

        if episode_seed is not None:
            np.random.seed(episode_seed)

        env = gym.make("intersection-v0", render_mode=None)
        unwrapped_env = env.unwrapped

        # Conservative intersection config for fine-tuning
        config = {
            "env_name": "intersection-v0",
            "observation": {
                "type": "LidarObservation",
                "cells": 32,
                "row_anchor": [0.5, 0.5],
                "features": ["presence", "distance", "speed"],
                "features_range": {"distance": [0, 50], "speed": [-30, 30]}
            },
            "action": {"type": "DiscreteMetaAction"},
            "duration": 35,
            # Conservative reward structure to avoid catastrophic forgetting
            "collision_reward": -22.0,  # Less aggressive than -25
            "high_speed_reward": 0.35,
            "reward_speed_range": [12, 22],
            "arrived_reward": 5.0,      # Moderate success bonus
            "progress_reward": 0.15,    # Small progress incentive
            "safe_distance_reward": 0.4,
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "vehicles_count": 14,       # Moderate traffic
            "initial_vehicle_count": 9,
        }

        # Light domain randomization
        config["vehicles_count"] = np.random.randint(12, 16)
        config["initial_vehicle_count"] = np.random.randint(8, 11)

        del config["env_name"]
        unwrapped_env.configure(config)

        env.reset()
        return env

    def compute_fisher_information(self, num_samples: int = 1000):
        """
        Compute Fisher Information Matrix on the base task (highway environment).

        The FIM measures parameter importance for the original task.
        """
        print(f"Computing Fisher Information Matrix with {num_samples} samples...")

        # Reset FIM
        self.fisher_information = {}
        for name, param in self.base_model.policy.named_parameters():
            if param.requires_grad:
                self.fisher_information[name] = torch.zeros_like(param)

        # Collect samples and compute gradients
        self.base_model.policy.train()  # Enable gradients

        for sample_idx in range(num_samples):
            if sample_idx % 200 == 0:
                print(f"  Processing sample {sample_idx}/{num_samples}...")

            # Sample from base environment
            obs, info = self.base_env.reset()

            # Convert to torch tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

            # Get action distribution (log probabilities)
            with torch.no_grad():
                actions, values, log_probs = self.base_model.policy(obs_tensor)

            # For discrete actions, compute score function estimator
            # Use the policy's action distribution
            dist = self.base_model.policy.get_distribution(obs_tensor)
            action_log_probs = dist.log_prob(actions)

            # Compute log probability of sampled action
            log_prob = action_log_probs  # Keep as tensor for gradient computation

            # Compute gradients of log probability w.r.t. parameters
            self.base_model.policy.zero_grad()
            log_prob.backward()

            # Accumulate squared gradients (Fisher Information)
            for name, param in self.base_model.policy.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_information[name] += param.grad.data ** 2

        # Normalize by number of samples
        for name in self.fisher_information:
            self.fisher_information[name] /= num_samples

        # Add small epsilon to avoid division by zero
        for name in self.fisher_information:
            self.fisher_information[name] += 1e-8

        print("Fisher Information Matrix computed successfully!")
        print(f"Parameters tracked: {len(self.fisher_information)}")

        # Store FIM for later use
        self.fim_computed = True

    def ewc_loss(self, current_params):
        """
        Compute EWC regularization loss.

        L_EWC = sum_i (F_i * (theta_i - theta_0_i)^2)
        """
        if not hasattr(self, 'fim_computed') or not self.fim_computed:
            raise ValueError("Fisher Information Matrix not computed. Call compute_fisher_information() first.")

        ewc_loss = 0.0

        for name, param in current_params:
            if name in self.theta_0 and name in self.fisher_information:
                theta_0 = self.theta_0[name]
                fisher = self.fisher_information[name]

                # EWC regularization: F * (theta - theta_0)^2
                param_diff = (param - theta_0) ** 2
                ewc_penalty = torch.sum(fisher * param_diff)

                ewc_loss += ewc_penalty

        return self.ewc_lambda * ewc_loss

    def create_ewc_callback(self, save_dir: Path):
        """Create callback that monitors EWC fine-tuning progress."""

        class EWCFinetuneCallback(BaseCallback):
            def __init__(self, ewc_finetuner: 'EWCFinetuner', save_dir: Path, log_freq: int = 250):
                super().__init__(verbose=0)
                self.ewc_finetuner = ewc_finetuner
                self.save_dir = save_dir
                self.log_freq = log_freq
                self.episode_rewards = []
                self.episode_lengths = []
                self.collision_count = []
                self.success_count = []
                self.ewc_losses = []
                self.start_time = time.time()

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
                        "ewc_finetune_timesteps": self.num_timesteps,
                        "episodes_completed": len(self.episode_rewards),

                        # Performance metrics
                        "avg_reward": np.mean(recent_rewards),
                        "reward_std": np.std(recent_rewards) if len(recent_rewards) > 1 else 0,
                        "avg_episode_length": np.mean(recent_lengths),
                        "crash_rate": np.mean(recent_collisions),
                        "success_rate": np.mean(recent_successes),

                        # EWC metrics
                        "ewc_lambda": self.ewc_finetuner.ewc_lambda,
                        "fim_parameters_tracked": len(self.ewc_finetuner.fisher_information),
                    }

                    # Compute current EWC loss
                    current_params = list(self.model.policy.named_parameters())
                    ewc_loss_value = self.ewc_finetuner.ewc_loss(current_params).item()
                    metrics["ewc_regularization_loss"] = ewc_loss_value

                    # Performance score
                    crash_rate = metrics['crash_rate']
                    success_rate = metrics['success_rate']
                    avg_reward = metrics['avg_reward']
                    performance_score = min(1.0, max(0.0, (avg_reward + 30) / 30))
                    metrics['performance_score'] = performance_score

                    wandb.log(metrics, step=self.num_timesteps)

                    # Apply EWC regularization to gradients
                    self.ewc_finetuner.apply_ewc_penalty(self.model, lambda_ewc=self.ewc_finetuner.ewc_lambda)

                    # Progress logging
                    if self.n_calls % 1000 == 0:
                        print(f"[EWC FINETUNE] Step {self.num_timesteps:,} | "
                              f"Success: {metrics['success_rate']:.1%} | "
                              f"Crash: {metrics['crash_rate']:.1%} | "
                              f"Avg Reward: {metrics['avg_reward']:.2f} | "
                              f"EWC Loss: {ewc_loss_value:.2f}")

                return True

        return EWCFinetuneCallback(self, save_dir, log_freq=250)

    def apply_ewc_penalty(self, model, lambda_ewc=500.0):
        """Apply EWC regularization by modifying gradients."""

        if not hasattr(self, 'fim_computed') or not self.fim_computed:
            return  # Skip if FIM not computed

        for name, param in model.policy.named_parameters():
            if param.grad is not None and name in self.theta_0 and name in self.fisher_information:
                # EWC penalty: lambda * F * (theta - theta_0)
                theta_0 = self.theta_0[name]
                fisher = self.fisher_information[name]
                param_diff = param.data - theta_0

                ewc_penalty = lambda_ewc * fisher * param_diff

                # Add EWC penalty to gradients
                param.grad.data += ewc_penalty

    def ewc_finetune(self, total_timesteps: int = 50000, learning_rate: float = 1e-5):
        """Fine-tune with EWC regularization to prevent catastrophic forgetting."""

        print(">>> EWC (Elastic Weight Consolidation) FINE-TUNING")
        print("=" * 80)
        print(f"Base Model: {self.base_model_path}")
        print(f"EWC Fine-tuning: {total_timesteps:,} timesteps")
        print(f"Learning Rate: {learning_rate} (conservative)")
        print(f"EWC Lambda: {self.ewc_lambda} (regularization strength)")
        print("Features:")
        print("  + Fisher Information Matrix computed on base task")
        print("  + EWC regularization prevents catastrophic forgetting")
        print("  + Conservative intersection rewards")
        print("  + Domain randomization for robustness")
        print("=" * 80)

        # Step 1: Compute Fisher Information Matrix on base task
        self.compute_fisher_information(num_samples=2000)

        # Step 2: Create intersection environment and model
        intersection_env = self.create_intersection_env()

        # Create new model for fine-tuning
        ewc_model = PPO(
            "MlpPolicy",
            intersection_env,
            learning_rate=learning_rate,
            batch_size=256,
            n_steps=2048,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device='cpu',
            verbose=1
        )

        # Copy parameters from base model to start fine-tuning
        base_state_dict = self.base_model.policy.state_dict()
        ewc_model.policy.load_state_dict(base_state_dict)

        # Setup directories
        timestamp = int(time.time())
        save_dir = Path(f"outputs/models/ewc_finetune_{timestamp}")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=total_timesteps // 4,
            save_path=str(save_dir),
            name_prefix="ewc_finetune"
        )

        ewc_callback = self.create_ewc_callback(save_dir)
        callback_list = CallbackList([checkpoint_callback, ewc_callback])

        # Initialize wandb
        run = wandb.init(
            project="highway-foundation-v2",
            name=f"ewc_finetune_{timestamp}",
            config={
                "fine_tune_type": "ewc_regularized_intersection",
                "base_model": self.base_model_path,
                "total_timesteps": total_timesteps,
                "learning_rate": learning_rate,
                "ewc_lambda": self.ewc_lambda,
                "fisher_samples": 2000,
                "focus": "intersection_performance_with_generalization_preservation"
            }
        )

        try:
            print(f"\nStarting EWC fine-tuning for {total_timesteps:,} timesteps...")
            print("EWC regularization will protect highway/merge skills while improving intersection performance.")

            # EWC fine-tuning with domain randomization
            trained_timesteps = 0
            check_interval = 5000

            while trained_timesteps < total_timesteps:
                remaining_steps = min(check_interval, total_timesteps - trained_timesteps)

                # Create fresh environment with randomization
                episode_seed = int(time.time() * 1000) % 1000000
                env = self.create_intersection_env(episode_seed)
                ewc_model.set_env(env)

                ewc_model.learn(
                    total_timesteps=remaining_steps,
                    callback=callback_list,
                    reset_num_timesteps=False
                )

                trained_timesteps += remaining_steps

            ewc_time = time.time() - self.start_time

            # Save final EWC fine-tuned model
            final_model_path = save_dir / "ewc_finetune_final.zip"
            ewc_model.save(str(final_model_path))

            # Save metadata
            metadata = {
                "base_model": self.base_model_path,
                "ewc_finetune_timesteps": trained_timesteps,
                "ewc_finetune_time_seconds": ewc_time,
                "learning_rate": learning_rate,
                "ewc_lambda": self.ewc_lambda,
                "fisher_samples": 2000,
                "final_model_path": str(final_model_path),
                "ewc_regularized": True,
                "domain_randomization": True,
                "focus": "intersection_performance_with_generalization_preservation",
                "timestamp": datetime.now().isoformat()
            }

            with open(save_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            print("\n[SUCCESS] EWC FINE-TUNING COMPLETED!")
            print("=" * 80)
            print(f"Base Model: {os.path.basename(self.base_model_path)}")
            print(f"EWC Fine-tuned Model: {final_model_path}")
            print(f"Training Time: {ewc_time:.1f} seconds")
            print(f"Total Steps: {trained_timesteps:,}")
            print("EWC Features:")
            print("  + Fisher Information Matrix computed on base task")
            print("  + EWC regularization prevents catastrophic forgetting")
            print("  + Conservative intersection rewards (-22 collision, +5 success)")
            print("  + Domain randomization for robustness")
            print("  + Protected highway/merge generalization")
            print("=" * 80)

            wandb.log({
                "ewc_fine_tune_completed": True,
                "total_ewc_fine_tune_time": ewc_time,
                "final_model_path": str(final_model_path)
            })

            intersection_env.close()
            self.base_env.close()
            return str(final_model_path)

        except Exception as e:
            print(f"[ERROR] EWC fine-tuning failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            wandb.finish()

def find_base_model():
    """Find the best base model for EWC fine-tuning."""

    # Priority: Advanced curriculum > regular fine-tune > multi-env
    candidates = [
        ("outputs/models/curriculum_advanced/*/advanced_curriculum_*_early_progression.zip", "Advanced Curriculum"),
        ("outputs/models/finetune_intersection_*/finetune_intersection_final.zip", "Fine-tuned"),
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
    """Main EWC fine-tuning execution."""

    print(">>> EWC (Elastic Weight Consolidation) FINE-TUNING")
    print("=" * 80)

    # Find the best base model
    base_model_path = find_base_model()

    if not base_model_path:
        print("[ERROR] No suitable base models found!")
        print("Please run curriculum training first.")
        return

    print(f"[BASE MODEL] {base_model_path}")

    # Create EWC fine-tuner
    ewc_finetuner = EWCFinetuner(
        base_model_path=base_model_path,
        ewc_lambda=500.0  # Moderate regularization strength
    )

    # EWC fine-tuning: conservative approach
    final_model_path = ewc_finetuner.ewc_finetune(
        total_timesteps=40000,  # Shorter than aggressive approach
        learning_rate=1e-5      # Conservative learning rate
    )

    if final_model_path:
        print(f"\n[SUCCESS] EWC fine-tuning completed!")
        print(f"Final Model: {final_model_path}")
        print("\nTo evaluate: python evaluate_finetuned_model.py")
        print("(Update the model path in the evaluation script)")

if __name__ == "__main__":
    main()
