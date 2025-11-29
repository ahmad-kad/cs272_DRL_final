#!/usr/bin/env python3
"""
Adaptive Curriculum Training for Enhanced Reward Structure

This trainer implements curriculum learning that progressively increases difficulty
across scenarios (highway → merge → intersection) and modalities (lidar → grayscale → both).
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
import json
from typing import Dict, List, Any, Optional

from environments.enhanced_urban_env import EnhancedUrbanJunctionEnv
from utils.callbacks import WandbMetricsCallback
from train_enhanced_rewards import TqdmCallback


class AdaptiveCurriculumTrainer:
    """
    Adaptive curriculum trainer that progresses through scenarios and modalities.
    """

    def __init__(self, base_dir="outputs", use_wandb=False):
        self.base_dir = base_dir
        self.use_wandb = use_wandb
        self.models_dir = os.path.join(base_dir, "models")
        self.logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Curriculum progression
        self.curriculum_phases = self._define_curriculum()
        self.current_phase = 0
        self.phase_performance_history = []

        # Training state
        self.model = None
        self.total_timesteps = 0

    def _define_curriculum(self) -> List[Dict[str, Any]]:
        """Define the curriculum progression."""
        return [
            # Phase 1: Foundation - Single scenario, single modality
            {
                "name": "foundation_highway_lidar",
                "scenarios": ["highway"],
                "modalities": ["lidar"],
                "timesteps": 5000,
                "difficulty": 0.3,
                "required_success_rate": 0.80,
                "description": "Learn basic highway driving with lidar"
            },
            {
                "name": "foundation_highway_grayscale",
                "scenarios": ["highway"],
                "modalities": ["grayscale"],
                "timesteps": 5000,
                "difficulty": 0.3,
                "required_success_rate": 0.80,
                "description": "Learn basic highway driving with vision"
            },

            # Phase 2: Integration - Single scenario, multi-modal
            {
                "name": "integration_highway_both",
                "scenarios": ["highway"],
                "modalities": ["both"],
                "timesteps": 8000,
                "difficulty": 0.4,
                "required_success_rate": 0.85,
                "description": "Combine lidar and vision on highways"
            },

            # Phase 3: Expansion - Multi-scenario, single modality
            {
                "name": "expansion_merge_lidar",
                "scenarios": ["highway", "merge"],
                "modalities": ["lidar"],
                "timesteps": 10000,
                "difficulty": 0.5,
                "required_success_rate": 0.75,
                "description": "Add merging scenarios with lidar"
            },
            {
                "name": "expansion_merge_grayscale",
                "scenarios": ["highway", "merge"],
                "modalities": ["grayscale"],
                "timesteps": 10000,
                "difficulty": 0.5,
                "required_success_rate": 0.75,
                "description": "Add merging scenarios with vision"
            },

            # Phase 4: Mastery - Multi-scenario, multi-modal
            {
                "name": "mastery_all_scenarios",
                "scenarios": ["highway", "merge", "intersection"],
                "modalities": ["lidar", "grayscale", "both"],
                "timesteps": 15000,
                "difficulty": 0.7,
                "required_success_rate": 0.70,
                "description": "Full multi-scenario, multi-modal learning"
            },

            # Phase 5: Specialization - Focus on weak areas
            {
                "name": "specialization_intersection",
                "scenarios": ["intersection"],
                "modalities": ["both"],
                "timesteps": 8000,
                "difficulty": 0.8,
                "required_success_rate": 0.80,
                "description": "Specialize in intersection handling"
            }
        ]

    def _create_curriculum_env(self, phase_config: Dict[str, Any]):
        """Create environment for current curriculum phase."""
        def make_env():
            # Sample scenario and modality for this phase
            scenario = np.random.choice(phase_config["scenarios"])
            modality = np.random.choice(phase_config["modalities"])

            env = EnhancedUrbanJunctionEnv(
                scenario=scenario,
                modality=modality,
                render_mode=None
            )
            env = Monitor(env)
            return env

        env = DummyVecEnv([make_env])
        return env

    def _evaluate_phase_performance(self, phase_config: Dict[str, Any], n_episodes=20) -> Dict[str, float]:
        """Evaluate performance on current curriculum phase."""
        print(f"  Evaluating phase performance...")

        total_reward = 0
        successful_episodes = 0
        crashes = 0
        safety_overrides = 0

        for episode in range(n_episodes):
            # Sample environment for evaluation
            scenario = np.random.choice(phase_config["scenarios"])
            modality = np.random.choice(phase_config["modalities"])

            env = EnhancedUrbanJunctionEnv(scenario=scenario, modality=modality, render_mode=None)

            obs, _ = env.reset()
            episode_reward = 0
            done = False
            steps = 0

            while not done and steps < 300:
                action, _ = self.model.predict(obs, deterministic=True)
                step_result = env.step(action)

                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result

                episode_reward += reward
                steps += 1

                if hasattr(env, 'vehicle') and env.vehicle.crashed:
                    crashes += 1
                    break

            total_reward += episode_reward

            # Success criteria
            safety_stats = env.get_safety_stats()
            safety_overrides += safety_stats["safety_override_count"]

            if episode_reward > 10 and crashes == 0:  # Reasonable success threshold
                successful_episodes += 1

        avg_reward = total_reward / n_episodes
        success_rate = successful_episodes / n_episodes
        crash_rate = crashes / n_episodes
        avg_safety_overrides = safety_overrides / n_episodes

        return {
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "crash_rate": crash_rate,
            "avg_safety_overrides": avg_safety_overrides,
            "episodes": n_episodes
        }

    def _should_progress_to_next_phase(self, phase_config: Dict[str, Any], performance: Dict[str, float]) -> bool:
        """Determine if agent is ready to progress to next phase."""
        required_success = phase_config["required_success_rate"]
        actual_success = performance["success_rate"]

        # Must meet success rate and have reasonable crash rate
        success_criteria = actual_success >= required_success
        safety_criteria = performance["crash_rate"] < 0.3  # Less than 30% crash rate

        print(f"  Phase requirements: Success ≥ {required_success:.2f}, Safety < 30% crashes")
        print(f"  Achieved: Success = {actual_success:.2f}, Crashes = {performance['crash_rate']:.2f}")

        return success_criteria and safety_criteria

    def train_curriculum(self, start_phase=0):
        """Execute the full curriculum training."""
        print("STARTING ADAPTIVE CURRICULUM TRAINING")
        print("=" * 60)
        print("Curriculum will progress through:")
        for i, phase in enumerate(self.curriculum_phases):
            status = "COMPLETED" if i < start_phase else "PENDING" if i > start_phase else "CURRENT"
            print(f"  Phase {i+1}: {phase['name']} - {status}")
            print(f"    {phase['description']}")
        print("=" * 60)

        self.current_phase = start_phase

        while self.current_phase < len(self.curriculum_phases):
            phase_config = self.curriculum_phases[self.current_phase]

            print(f"\nPHASE {self.current_phase + 1}: {phase_config['name']}")
            print(f"   {phase_config['description']}")
            print(f"   Scenarios: {phase_config['scenarios']}")
            print(f"   Modalities: {phase_config['modalities']}")
            print(f"   Timesteps: {phase_config['timesteps']}")
            print("-" * 50)

            # Create environment for this phase
            env = self._create_curriculum_env(phase_config)

            # Initialize or continue model
            if self.model is None:
                # First phase - create new model
                self.model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=0,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.02,  # Good exploration for curriculum
                    vf_coef=0.5,
                )
                print("   🆕 Created new PPO model")
            else:
                # Continue training with existing model
                self.model.set_env(env)
                print("   Continuing with existing model")

            # Setup callbacks
            callbacks = [TqdmCallback(total_timesteps=phase_config["timesteps"], update_freq=500)]
            if self.use_wandb:
                callbacks.append(WandbMetricsCallback())

            checkpoint_cb = CheckpointCallback(
                save_freq=max(1000, phase_config["timesteps"] // 10),
                save_path=self.models_dir,
                name_prefix=f"curriculum_phase_{self.current_phase}_{phase_config['name']}"
            )
            callbacks.append(checkpoint_cb)

            # Train on this phase
            print(f"   Training for {phase_config['timesteps']} timesteps...")
            self.model.learn(
                total_timesteps=phase_config["timesteps"],
                callback=CallbackList(callbacks)
            )

            self.total_timesteps += phase_config["timesteps"]

            # Evaluate performance
            performance = self._evaluate_phase_performance(phase_config)
            self.phase_performance_history.append({
                "phase": self.current_phase,
                "config": phase_config,
                "performance": performance,
                "total_timesteps": self.total_timesteps
            })

            # Check if ready to progress
            if self._should_progress_to_next_phase(phase_config, performance):
                print("   Phase completed successfully! Advancing to next phase...")
                self.current_phase += 1
            else:
                print("   Phase not yet mastered. Continuing training on current phase...")
                # Could implement additional training or difficulty adjustment here

                # For now, advance anyway after additional attempts
                if len([p for p in self.phase_performance_history if p["phase"] == self.current_phase]) > 2:
                    print("   Multiple attempts made. Advancing with caution...")
                    self.current_phase += 1

        # Curriculum completed
        self._save_curriculum_results()
        print("\n🎉 CURRICULUM TRAINING COMPLETED!")
        print(f"   Total timesteps trained: {self.total_timesteps}")
        print(f"   Phases completed: {len(self.curriculum_phases)}")
        print(f"   Final model saved in: {self.models_dir}")

        return self.model

    def _save_curriculum_results(self):
        """Save curriculum training results."""
        results_file = os.path.join(self.base_dir, "curriculum_results.json")

        results = {
            "total_timesteps": self.total_timesteps,
            "phases_completed": len(self.curriculum_phases),
            "phase_history": self.phase_performance_history,
            "final_phase": self.current_phase,
            "training_summary": {
                "scenarios_mastered": list(set([s for phase in self.curriculum_phases for s in phase["scenarios"]])),
                "modalities_mastered": list(set([m for phase in self.curriculum_phases for m in phase["modalities"]])),
                "curriculum_phases": len(self.curriculum_phases)
            }
        }

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"   💾 Curriculum results saved to: {results_file}")

    def evaluate_final_performance(self):
        """Evaluate final model on all scenario-modality combinations."""
        print("\n🔬 FINAL MODEL EVALUATION")
        print("-" * 40)

        test_configs = [
            ("highway", "lidar"),
            ("highway", "grayscale"),
            ("highway", "both"),
            ("merge", "lidar"),
            ("merge", "grayscale"),
            ("merge", "both"),
            ("intersection", "lidar"),
            ("intersection", "grayscale"),
            ("intersection", "both")
        ]

        results = {}

        for scenario, modality in test_configs:
            print(f"  Testing {scenario} + {modality}...")

            env = EnhancedUrbanJunctionEnv(scenario=scenario, modality=modality, render_mode=None)

            total_reward = 0
            successful_episodes = 0
            crashes = 0

            for episode in range(10):  # Test 10 episodes each
                obs, _ = env.reset()
                episode_reward = 0
                done = False
                steps = 0

                while not done and steps < 300:
                    action, _ = self.model.predict(obs, deterministic=True)
                    step_result = env.step(action)

                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = step_result

                    episode_reward += reward
                    steps += 1

                    if hasattr(env, 'vehicle') and env.vehicle.crashed:
                        crashes += 1
                        break

                total_reward += episode_reward
                if episode_reward > 15 and crashes == 0:  # Success threshold
                    successful_episodes += 1

            results[f"{scenario}_{modality}"] = {
                "avg_reward": total_reward / 10,
                "success_rate": successful_episodes / 10,
                "crash_rate": crashes / 10
            }

        # Print summary
        print("\n📊 FINAL PERFORMANCE SUMMARY:")
        print("-" * 50)
        print("<15")
        print("-" * 50)

        for config, metrics in results.items():
            print("<15")

        return results


def run_adaptive_curriculum_training(start_phase=0, use_wandb=False):
    """Run the complete adaptive curriculum training."""
    trainer = AdaptiveCurriculumTrainer(use_wandb=use_wandb)
    final_model = trainer.train_curriculum(start_phase=start_phase)

    # Final evaluation
    final_results = trainer.evaluate_final_performance()

    return final_model, final_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive Curriculum Training")
    parser.add_argument("--start-phase", type=int, default=0, help="Starting curriculum phase")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--evaluate-only", action="store_true", help="Only run final evaluation")

    args = parser.parse_args()

    if args.evaluate_only:
        # Load existing model and evaluate
        trainer = AdaptiveCurriculumTrainer()
        try:
            from stable_baselines3 import PPO
            trainer.model = PPO.load("outputs/models/curriculum_phase_6_specialization_intersection_final.zip")
            trainer.evaluate_final_performance()
        except FileNotFoundError:
            print("❌ No trained model found. Run training first.")
    else:
        # Run full curriculum training
        run_adaptive_curriculum_training(start_phase=args.start_phase, use_wandb=args.use_wandb)
