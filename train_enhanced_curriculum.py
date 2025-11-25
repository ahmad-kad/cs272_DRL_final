#!/usr/bin/env python3
"""
Enhanced Curriculum Learning with Domain Randomization
Includes enhanced rewards, extended phases, adaptive scheduling, and domain randomization.
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
import torch

class CurriculumPhase:
    """Represents a single phase in the curriculum with its configuration and progression criteria."""

    def __init__(self, name: str, env_config: dict, progression_criteria: dict,
                 target_timesteps: int = 50000, domain_randomization: bool = False):
        self.name = name
        self.base_env_config = env_config
        self.progression_criteria = progression_criteria
        self.target_timesteps = target_timesteps
        self.completed_timesteps = 0
        self.best_performance = 0.0
        self.performance_history = []
        self.domain_randomization = domain_randomization

    def get_randomized_config(self) -> dict:
        """Apply domain randomization to the base config if enabled."""
        if not self.domain_randomization:
            return self.base_env_config.copy()

        config = self.base_env_config.copy()

        # Domain randomization for intersection phases
        if "intersection" in self.name:
            # Randomize traffic density
            base_vehicles = config.get("vehicles_count", 10)
            config["vehicles_count"] = random.randint(max(6, base_vehicles-3), base_vehicles+3)

            base_initial = config.get("initial_vehicle_count", 6)
            config["initial_vehicle_count"] = random.randint(max(3, base_initial-2), base_initial+2)

            # Randomize timing
            config["simulation_frequency"] = random.uniform(12, 18)

            # Randomize reward weights slightly
            config["collision_reward"] = config.get("collision_reward", -20.0) * random.uniform(0.9, 1.1)
            config["arrived_reward"] = config.get("arrived_reward", 4.0) * random.uniform(0.9, 1.1)

        return config

    def should_progress(self, current_metrics: dict) -> bool:
        """Check if phase progression criteria are met."""
        crash_rate = current_metrics.get('crash_rate', 1.0)
        success_rate = current_metrics.get('success_rate', 0.0)
        avg_reward = current_metrics.get('avg_reward', -float('inf'))

        criteria = self.progression_criteria
        return (crash_rate <= criteria.get('max_crash_rate', 0.5) and
                success_rate >= criteria.get('min_success_rate', 0.3) and
                avg_reward >= criteria.get('min_avg_reward', -5.0))

class EnhancedCurriculumTrainer:
    """Enhanced curriculum trainer with domain randomization and advanced rewards."""

    def __init__(self):
        self.phases = self._define_curriculum_phases()
        self.current_phase_idx = 0
        self.start_time = time.time()

    def _define_curriculum_phases(self) -> List[CurriculumPhase]:
        """Define the enhanced curriculum phases with domain randomization."""

        phases = [
            # Phase 1: Highway Foundation
            CurriculumPhase(
                name="highway_foundation",
                env_config={
                    "env_name": "highway-v0",
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
                },
                progression_criteria={
                    "max_crash_rate": 0.15,
                    "min_success_rate": 0.8,
                    "min_avg_reward": 18.0
                },
                target_timesteps=35000
            ),

            # Phase 2: Easy Merge
            CurriculumPhase(
                name="easy_merge",
                env_config={
                    "env_name": "merge-v0",
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
                    "high_speed_reward": 0.5,
                    "reward_speed_range": [18, 28],
                    "lane_change_reward": 0.2,
                    "simulation_frequency": 15,
                    "policy_frequency": 1,
                    "vehicles_count": 8,
                },
                progression_criteria={
                    "max_crash_rate": 0.2,
                    "min_success_rate": 0.7,
                    "min_avg_reward": 10.0
                },
                target_timesteps=40000
            ),

            # Phase 3: Hard Merge
            CurriculumPhase(
                name="hard_merge",
                env_config={
                    "env_name": "merge-v0",
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
                    "high_speed_reward": 0.5,
                    "reward_speed_range": [18, 28],
                    "lane_change_reward": 0.2,
                    "simulation_frequency": 15,
                    "policy_frequency": 1,
                    "vehicles_count": 15,
                },
                progression_criteria={
                    "max_crash_rate": 0.25,
                    "min_success_rate": 0.6,
                    "min_avg_reward": 8.0
                },
                target_timesteps=45000
            ),

            # Phase 4: Very Easy Intersection (with domain randomization)
            CurriculumPhase(
                name="very_easy_intersection",
                env_config={
                    "env_name": "intersection-v0",
                    "observation": {
                        "type": "LidarObservation",
                        "cells": 32,
                        "row_anchor": [0.5, 0.5],
                        "features": ["presence", "distance", "speed"],
                        "features_range": {"distance": [0, 50], "speed": [-30, 30]}
                    },
                    "action": {"type": "DiscreteMetaAction"},
                    "duration": 60,
                    "collision_reward": -25.0,
                    "high_speed_reward": 0.3,
                    "reward_speed_range": [10, 20],
                    "arrived_reward": 5.0,
                    "progress_reward": 0.1,
                    "safe_distance_reward": 0.3,
                    "simulation_frequency": 15,
                    "policy_frequency": 1,
                    "vehicles_count": 6,
                    "initial_vehicle_count": 3,
                },
                progression_criteria={
                    "max_crash_rate": 0.15,
                    "min_success_rate": 0.85,
                    "min_avg_reward": 3.0
                },
                target_timesteps=50000,
                domain_randomization=True
            ),

            # Phase 5: Easy Intersection (with domain randomization)
            CurriculumPhase(
                name="easy_intersection",
                env_config={
                    "env_name": "intersection-v0",
                    "observation": {
                        "type": "LidarObservation",
                        "cells": 32,
                        "row_anchor": [0.5, 0.5],
                        "features": ["presence", "distance", "speed"],
                        "features_range": {"distance": [0, 50], "speed": [-30, 30]}
                    },
                    "action": {"type": "DiscreteMetaAction"},
                    "duration": 50,
                    "collision_reward": -22.0,
                    "high_speed_reward": 0.35,
                    "reward_speed_range": [12, 22],
                    "arrived_reward": 4.5,
                    "progress_reward": 0.12,
                    "safe_distance_reward": 0.35,
                    "simulation_frequency": 15,
                    "policy_frequency": 1,
                    "vehicles_count": 10,
                    "initial_vehicle_count": 6,
                },
                progression_criteria={
                    "max_crash_rate": 0.25,
                    "min_success_rate": 0.7,
                    "min_avg_reward": 2.0
                },
                target_timesteps=55000,
                domain_randomization=True
            ),

            # Phase 6: Medium Intersection (with domain randomization)
            CurriculumPhase(
                name="medium_intersection",
                env_config={
                    "env_name": "intersection-v0",
                    "observation": {
                        "type": "LidarObservation",
                        "cells": 32,
                        "row_anchor": [0.5, 0.5],
                        "features": ["presence", "distance", "speed"],
                        "features_range": {"distance": [0, 50], "speed": [-30, 30]}
                    },
                    "action": {"type": "DiscreteMetaAction"},
                    "duration": 45,
                    "collision_reward": -20.0,
                    "high_speed_reward": 0.35,
                    "reward_speed_range": [12, 22],
                    "arrived_reward": 4.0,
                    "progress_reward": 0.15,
                    "safe_distance_reward": 0.4,
                    "simulation_frequency": 15,
                    "policy_frequency": 1,
                    "vehicles_count": 12,
                    "initial_vehicle_count": 8,
                },
                progression_criteria={
                    "max_crash_rate": 0.3,
                    "min_success_rate": 0.6,
                    "min_avg_reward": 1.5
                },
                target_timesteps=60000,
                domain_randomization=True
            ),

            # Phase 7: Hard Intersection (with domain randomization)
            CurriculumPhase(
                name="hard_intersection",
                env_config={
                    "env_name": "intersection-v0",
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
                    "high_speed_reward": 0.4,
                    "reward_speed_range": [15, 25],
                    "arrived_reward": 4.0,
                    "progress_reward": 0.2,
                    "safe_distance_reward": 0.5,
                    "simulation_frequency": 15,
                    "policy_frequency": 1,
                    "vehicles_count": 15,
                    "initial_vehicle_count": 10,
                },
                progression_criteria={
                    "max_crash_rate": 0.35,
                    "min_success_rate": 0.5,
                    "min_avg_reward": 1.0
                },
                target_timesteps=70000,
                domain_randomization=True
            ),

            # Phase 8: Expert Intersection (with maximum domain randomization)
            CurriculumPhase(
                name="expert_intersection",
                env_config={
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
                    "collision_reward": -25.0,
                    "high_speed_reward": 0.3,
                    "reward_speed_range": [10, 20],
                    "arrived_reward": 6.0,
                    "progress_reward": 0.25,
                    "safe_distance_reward": 0.6,
                    "simulation_frequency": 15,
                    "policy_frequency": 1,
                    "vehicles_count": 18,
                    "initial_vehicle_count": 12,
                },
                progression_criteria={
                    "max_crash_rate": 0.4,
                    "min_success_rate": 0.45,
                    "min_avg_reward": 0.5
                },
                target_timesteps=80000,
                domain_randomization=True
            )
        ]

        return phases

    def create_curriculum_env(self, phase: CurriculumPhase, episode_seed: int = None):
        """Create environment for the current curriculum phase with optional randomization."""

        # Set random seed for reproducible randomization
        if episode_seed is not None:
            random.seed(episode_seed)
            np.random.seed(episode_seed)

        config = phase.get_randomized_config()
        env_name = config["env_name"]
        env = gym.make(env_name, render_mode=None)
        unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env

        # Apply phase-specific configuration
        del config["env_name"]  # Remove env_name from config
        unwrapped_env.configure(config)

        env.reset()
        return env

    def create_enhanced_callback(self, phase: CurriculumPhase, save_dir: Path, progression_signal: dict):
        """Create enhanced callback with curriculum-aware metrics and adaptive progression."""

        class EnhancedCurriculumCallback(BaseCallback):
            def __init__(self, phase: CurriculumPhase, save_dir: Path, progression_signal: dict, log_freq: int = 500):
                super().__init__(verbose=0)
                self.phase = phase
                self.save_dir = save_dir
                self.progression_signal = progression_signal
                self.log_freq = log_freq
                self.episode_rewards = []
                self.episode_lengths = []
                self.collision_count = []
                self.success_count = []
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
                                'episode_count': 0
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
                            env_data['episode_count'] += 1

                # Log at regular intervals
                if self.n_calls % self.log_freq == 0 and len(self.episode_rewards) > 0:
                    recent_rewards = self.episode_rewards[-20:]
                    recent_lengths = self.episode_lengths[-20:]
                    recent_collisions = self.collision_count[-20:]
                    recent_successes = self.success_count[-20:]

                    metrics = {
                        "phase": self.phase.name,
                        "phase_progress": self.phase.completed_timesteps / self.phase.target_timesteps,
                        "timesteps": self.num_timesteps,
                        "episodes_completed": len(self.episode_rewards),

                        # Performance metrics
                        "avg_reward": np.mean(recent_rewards),
                        "reward_std": np.std(recent_rewards) if len(recent_rewards) > 1 else 0,
                        "avg_episode_length": np.mean(recent_lengths),
                        "crash_rate": np.mean(recent_collisions),
                        "success_rate": np.mean(recent_successes),

                        # Learning progress
                        "learning_progress": np.mean(recent_rewards) - (np.mean(self.episode_rewards[:10]) if len(self.episode_rewards) > 10 else 0),
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

                    # Check progression criteria
                    current_metrics = {
                        'crash_rate': metrics['crash_rate'],
                        'success_rate': metrics['success_rate'],
                        'avg_reward': metrics['avg_reward']
                    }

                    if self.phase.should_progress(current_metrics):
                        print(f"\n[PROGRESSION MET] for {self.phase.name}!")
                        print(f"   Crash Rate: {metrics['crash_rate']:.3f} (target: <={self.phase.progression_criteria['max_crash_rate']})")
                        print(f"   Success Rate: {metrics['success_rate']:.3f} (target: >={self.phase.progression_criteria['min_success_rate']})")
                        print(f"   Avg Reward: {metrics['avg_reward']:.2f} (target: >={self.phase.progression_criteria['min_avg_reward']})")
                        print("   TRIGGERING EARLY PROGRESSION TO NEXT PHASE!\n")

                        # Signal to training loop that progression criteria are met
                        self.progression_signal['met'] = True
                        self.progression_signal['timesteps'] = self.num_timesteps

                return True

        return EnhancedCurriculumCallback(phase, save_dir, progression_signal, log_freq=500)

    def train_curriculum_phase(self, phase: CurriculumPhase, prev_model_path: Optional[str] = None):
        """Train a single curriculum phase with adaptive scheduling and domain randomization."""

        print(f"\n[PHASE START] {phase.name}")
        print("=" * 60)
        print(f"Target: {phase.target_timesteps:,} timesteps")
        print(f"Domain Randomization: {'Enabled' if phase.domain_randomization else 'Disabled'}")
        print(f"Progression Criteria:")
        print(f"  - Max Crash Rate: ≤{phase.progression_criteria['max_crash_rate']}")
        print(f"  - Min Success Rate: ≥{phase.progression_criteria['min_success_rate']}")
        print(f"  - Min Avg Reward: ≥{phase.progression_criteria['min_avg_reward']}")

        # Create environment (without randomization for initial model loading)
        env = self.create_curriculum_env(phase)

        # Load or create model
        if prev_model_path and os.path.exists(prev_model_path):
            print(f"Loading previous model: {prev_model_path}")
            model = PPO.load(prev_model_path, env=env, device='cpu')
        else:
            print("Creating new model for this phase")
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=5e-4,
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

        # Setup directories
        save_dir = Path(f"outputs/models/enhanced_curriculum/{phase.name}")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=phase.target_timesteps // 4,
            save_path=str(save_dir),
            name_prefix=f"enhanced_curriculum_{phase.name}"
        )

        # Create progression signal for callback communication
        progression_signal = {'met': False, 'timesteps': 0}

        enhanced_callback = self.create_enhanced_callback(phase, save_dir, progression_signal)
        callback_list = CallbackList([checkpoint_callback, enhanced_callback])

        # Train the phase
        print(f"\nTraining {phase.name} for up to {phase.target_timesteps:,} timesteps...")
        phase_start_time = time.time()

        trained_timesteps = 0
        check_interval = 10000

        while trained_timesteps < phase.target_timesteps and not progression_signal['met']:
            remaining_steps = min(check_interval, phase.target_timesteps - trained_timesteps)

            # Create fresh environment with randomization for each training segment
            if phase.domain_randomization and trained_timesteps > 0:
                episode_seed = int(time.time() * 1000) % 1000000  # Random seed for reproducibility
                env = self.create_curriculum_env(phase, episode_seed)
                model.set_env(env)  # Update model environment

            model.learn(
                total_timesteps=remaining_steps,
                callback=callback_list,
                reset_num_timesteps=False
            )

            trained_timesteps += remaining_steps
            phase.completed_timesteps = trained_timesteps

            # Check if progression criteria were met during this training segment
            if progression_signal['met']:
                print(f"\n[EARLY PROGRESSION] Phase {phase.name} met criteria after {progression_signal['timesteps']:,} timesteps!")
                print("Advancing to next curriculum phase...")
                break

        phase_time = time.time() - phase_start_time

        # Determine completion type
        completion_type = "early_progression" if progression_signal['met'] else "full_completion"

        print(f"\n[PHASE COMPLETED] {phase.name}")
        print(f"   Completion Type: {completion_type}")
        print(f"   Trained for: {trained_timesteps:,} timesteps ({phase_time:.1f}s)")
        if progression_signal['met']:
            print(f"   Early progression at: {progression_signal['timesteps']:,} timesteps")

        # Save phase model
        phase_model_path = save_dir / f"enhanced_curriculum_{phase.name}_{completion_type}.zip"
        model.save(str(phase_model_path))

        # Save phase metadata
        metadata = {
            "phase_name": phase.name,
            "completed_timesteps": trained_timesteps,
            "training_time_seconds": phase_time,
            "final_model_path": str(phase_model_path),
            "progression_criteria": phase.progression_criteria,
            "env_config": phase.base_env_config,
            "completion_type": completion_type,
            "early_progression_timesteps": progression_signal['timesteps'] if progression_signal['met'] else None,
            "domain_randomization": phase.domain_randomization,
            "timestamp": datetime.now().isoformat()
        }

        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        env.close()

        return str(phase_model_path)

    def train_full_curriculum(self):
        """Train the complete enhanced curriculum."""

        print("🚀 ENHANCED CURRICULUM TRAINING WITH DOMAIN RANDOMIZATION")
        print("=" * 80)
        print("Enhanced Features:")
        print("  + Advanced reward shaping with intersection-specific incentives")
        print("  + Extended curriculum: 8 phases with gradual difficulty progression")
        print("  + Adaptive curriculum scheduler based on performance metrics")
        print("  + Domain randomization for robust intersection handling")
        print("  + Episode-level environment variation during training")
        print("=" * 80)

        # Initialize wandb
        run = wandb.init(
            project="highway-foundation-v2",
            name=f"enhanced_curriculum_{int(time.time())}",
            config={
                "curriculum_type": "enhanced_with_domain_randomization",
                "phases": [phase.name for phase in self.phases],
                "total_phases": len(self.phases),
                "features": ["enhanced_rewards", "extended_phases", "adaptive_scheduling", "domain_randomization"]
            }
        )

        try:
            prev_model_path = None

            for i, phase in enumerate(self.phases):
                print(f"\n[CURRICULUM PROGRESS] Phase {i+1}/{len(self.phases)}: {phase.name}")
                if phase.domain_randomization:
                    print("   🌪️  Domain randomization enabled")

                # Train current phase
                prev_model_path = self.train_curriculum_phase(phase, prev_model_path)

                # Phase completion logging
                wandb.log({
                    "curriculum_phase": i + 1,
                    "completed_phase": phase.name,
                    "total_phases": len(self.phases),
                    "phase_completion_time": time.time() - self.start_time,
                    "domain_randomization_active": phase.domain_randomization
                }, step=sum(p.completed_timesteps for p in self.phases[:i+1]))

                print(f"\n[PHASE COMPLETED] {i+1}: {phase.name}")

            # Final curriculum completion
            total_time = time.time() - self.start_time
            print(f"\n[SUCCESS] ENHANCED CURRICULUM TRAINING COMPLETED!")
            print("=" * 80)
            print("Summary:")
            print(f"  - Total phases: {len(self.phases)}")
            print(f"  - Total training time: {total_time:.1f} seconds")
            print(f"  - Final model: {prev_model_path}")
            print("Features achieved:")
            print("  + Advanced safety incentives through enhanced reward shaping")
            print("  + Progressive difficulty with 8-phase extended curriculum")
            print("  + Performance-based curriculum advancement")
            print("  + Domain randomization for robust intersection handling")
            print("=" * 80)

            wandb.log({
                "curriculum_completed": True,
                "total_training_time": total_time,
                "final_model_path": prev_model_path,
                "curriculum_phases_completed": len(self.phases)
            })

        except Exception as e:
            print(f"[ERROR] Curriculum training failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            wandb.finish()

        return prev_model_path

def main():
    """Main execution function."""
    trainer = EnhancedCurriculumTrainer()
    final_model_path = trainer.train_full_curriculum()
    return final_model_path

if __name__ == "__main__":
    main()
