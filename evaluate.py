#!/usr/bin/env python3
"""
Evaluation script for autonomous driving models.
Tests models on highway, merge, and intersection environments.
"""

import argparse
import gymnasium as gym
import highway_env
import json
import os
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from typing import Dict, List, Any

from utils.config import get_curriculum_config


class ModelEvaluator:
    """Evaluates trained models on different driving scenarios."""

    def __init__(self, model_path: str, modality: str = "lidar", use_attention: bool = True):
        self.model_path = model_path
        self.modality = modality
        self.use_attention = use_attention
        self.model = None

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the PPO model from file."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        print(f"Loading model: {self.model_path}")
        self.model = PPO.load(self.model_path, device="cpu")
        print("Model loaded successfully")

    def _create_test_environment(self, scenario: str, difficulty: str = "vanilla") -> DummyVecEnv:
        """Create test environment for a specific scenario."""
        def make_env():
            if difficulty == "vanilla":
                # Use vanilla environment but configure with the correct observation type
                env = gym.make(f"{scenario}-v0", render_mode=None)

                # Configure with appropriate observation to match trained model
                if self.modality == "lidar":
                    obs_config = {
                        "type": "LidarObservation",
                        "cells": 32,
                        "row_anchor": [0.5, 0.5],
                        "features": ["presence", "distance", "speed"],
                        "features_range": {"distance": [0, 50], "speed": [-30, 30]}
                    }
                elif self.modality == "grayscale":
                    obs_config = {
                        "type": "GrayscaleObservation",
                        "observation_shape": (128, 64),
                        "stack_size": 4,
                        "weights": [0.2989, 0.5870, 0.1140],
                        "scaling": 1.75,
                    }
                else:
                    raise ValueError(f"Unknown modality: {self.modality}")

                env.unwrapped.configure({"observation": obs_config})
            else:
                # Get curriculum configuration for easy/medium/hard
                config = get_curriculum_config(f"{scenario}-v0", difficulty, self.modality)
                env = gym.make(f"{scenario}-v0", render_mode=None)
                env.unwrapped.configure(config)

            env.reset()
            return env

        return DummyVecEnv([make_env])

    def evaluate_scenario(self, scenario: str, n_episodes: int = 50,
                         difficulty: str = "vanilla") -> Dict[str, float]:
        """
        Evaluate model performance on a specific scenario.

        Args:
            scenario: "highway", "merge", or "intersection"
            n_episodes: Number of episodes to evaluate
            difficulty: Difficulty level ("easy", "medium", "hard")

        Returns:
            Dictionary with performance metrics
        """
        print(f"\nEvaluating on {scenario}-{difficulty}...")

        # Create environment for this scenario
        env = self._create_test_environment(scenario, difficulty)

        # Evaluation metrics
        success_count = 0
        total_reward = 0.0
        crash_count = 0
        episode_lengths = []
        episode_rewards = []

        for episode in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0.0
            episode_crashes = 0
            steps = 0

            while not done and steps < 200:  # Max episode length
                action, _ = self.model.predict(obs, deterministic=True)

                # Handle different action spaces for different environments
                # Highway/Merge: 5 actions (0-4), Intersection: 3 actions (0-2)
                if hasattr(env, 'action_space') and hasattr(env.action_space, 'n'):
                    max_action = env.action_space.n - 1
                    if isinstance(action, (list, tuple, np.ndarray)):
                        action_scalar = int(action[0]) if len(action) > 0 else 0
                    else:
                        action_scalar = int(action)
                    action_scalar = min(action_scalar, max_action)
                    # For DummyVecEnv, actions need to be in array format
                    action = [action_scalar]
                else:
                    # Ensure action is in correct format for vec env
                    if not isinstance(action, (list, tuple, np.ndarray)):
                        action = [int(action)]

                # Handle different Gymnasium API versions
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                elif len(step_result) == 4:
                    obs, reward, done, info = step_result
                    terminated = done
                    truncated = False
                else:
                    raise ValueError(f"Unexpected step result length: {len(step_result)}")

                episode_reward += float(reward)
                steps += 1

                # Check for crashes (handle different env info formats)
                if hasattr(info, '__iter__'):  # Handle vec env
                    for info_item in info:
                        if isinstance(info_item, dict) and info_item.get("crashed", False):
                            episode_crashes += 1
                elif isinstance(info, dict) and info.get("crashed", False):
                    episode_crashes += 1

            # Episode complete
            total_reward += episode_reward
            crash_count += episode_crashes
            episode_lengths.append(steps)
            episode_rewards.append(episode_reward)

            # Success criteria: positive reward and no crashes
            if episode_reward > 0 and episode_crashes == 0:
                success_count += 1

            if (episode + 1) % 10 == 0:
                reward_scalar = float(episode_reward) if hasattr(episode_reward, '__iter__') else episode_reward
                print(f"  Episode {episode + 1}/{n_episodes}: "
                      f"Reward={reward_scalar:.2f}, Crashes={episode_crashes}, Steps={steps}")

        # Calculate final metrics
        success_rate = success_count / n_episodes
        avg_reward = total_reward / n_episodes
        crash_rate = crash_count / n_episodes
        avg_episode_length = np.mean(episode_lengths)
        reward_std = np.std(episode_rewards)

        results = {
            "scenario": scenario,
            "difficulty": difficulty,
            "episodes": n_episodes,
            "success_rate": float(success_rate),
            "crash_rate": float(crash_rate),
            "avg_reward": float(avg_reward),
            "reward_std": float(reward_std),
            "avg_episode_length": float(avg_episode_length),
            "success_count": success_count,
            "total_crashes": crash_count
        }

        print(f"Results for {scenario}-{difficulty}:")
        print(".3f")
        print(".3f")
        print(".2f")
        print(f"  Total Successes: {success_count}/{n_episodes}")
        print(f"  Total Crashes: {crash_count}")

        env.close()
        return results

    def evaluate_all_scenarios(self, n_episodes: int = 50,
                              difficulty: str = "vanilla") -> Dict[str, Any]:
        """
        Evaluate model on all three scenarios: highway, merge, intersection.

        Returns:
            Dictionary with results for all scenarios
        """
        scenarios = ["highway", "merge", "intersection"]
        results = {
            "model_path": self.model_path,
            "modality": self.modality,
            "use_attention": self.use_attention,
            "scenarios": {},
            "summary": {}
        }

        all_success_rates = []
        all_crash_rates = []
        all_rewards = []

        for scenario in scenarios:
            scenario_results = self.evaluate_scenario(scenario, n_episodes, difficulty)
            results["scenarios"][scenario] = scenario_results

            all_success_rates.append(scenario_results["success_rate"])
            all_crash_rates.append(scenario_results["crash_rate"])
            all_rewards.append(scenario_results["avg_reward"])

        # Calculate summary statistics
        results["summary"] = {
            "overall_success_rate": float(np.mean(all_success_rates)),
            "overall_crash_rate": float(np.mean(all_crash_rates)),
            "overall_avg_reward": float(np.mean(all_rewards)),
            "success_rate_std": float(np.std(all_success_rates)),
            "crash_rate_std": float(np.std(all_crash_rates)),
            "avg_reward_std": float(np.std(all_rewards))
        }

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate autonomous driving models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate the adaptive lidar model on all scenarios
  python evaluate.py --model outputs/models/adaptive_lidar_final.zip --scenarios highway merge intersection

  # Evaluate with custom number of episodes
  python evaluate.py --model outputs/models/adaptive_lidar_final.zip --episodes 100

  # Evaluate specific scenarios
  python evaluate.py --model outputs/models/adaptive_lidar_final.zip --scenarios highway merge
        """
    )

    parser.add_argument("--model", type=str, required=True,
                       help="Path to the model file (.zip)")
    parser.add_argument("--scenarios", nargs="+",
                       choices=["highway", "merge", "intersection"],
                       default=["highway", "merge", "intersection"],
                       help="Scenarios to evaluate on")
    parser.add_argument("--episodes", type=int, default=50,
                       help="Number of episodes per scenario")
    parser.add_argument("--difficulty", type=str, default="vanilla",
                       choices=["vanilla", "easy", "medium", "hard"],
                       help="Difficulty level for evaluation (vanilla = default environment)")
    parser.add_argument("--modality", type=str, default="lidar",
                       choices=["lidar", "grayscale"],
                       help="Model modality")
    parser.add_argument("--attention", action="store_true",
                       help="Whether model uses attention")
    parser.add_argument("--output", type=str,
                       help="Output file for results (JSON)")

    args = parser.parse_args()

    # Create evaluator
    try:
        evaluator = ModelEvaluator(
            model_path=args.model,
            modality=args.modality,
            use_attention=args.attention
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Evaluate on specified scenarios
    if len(args.scenarios) == 3:
        # Evaluate all scenarios
        print("Evaluating on all scenarios...")
        results = evaluator.evaluate_all_scenarios(
            n_episodes=args.episodes,
            difficulty=args.difficulty
        )

        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(".3f")
        print(".3f")
        print(".2f")
        print(f"Model: {args.model}")
        print(f"Episodes per scenario: {args.episodes}")
        print(f"Difficulty: {args.difficulty}")

    else:
        # Evaluate specific scenarios
        results = {
            "model_path": args.model,
            "modality": args.modality,
            "use_attention": args.attention,
            "scenarios": {},
            "summary": {}
        }

        for scenario in args.scenarios:
            scenario_results = evaluator.evaluate_scenario(
                scenario, args.episodes, args.difficulty
            )
            results["scenarios"][scenario] = scenario_results

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    # Also save to default location
    default_output = f"results/{Path(args.model).stem}_evaluation.json"
    Path("results").mkdir(exist_ok=True)

    with open(default_output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results also saved to: {default_output}")


if __name__ == "__main__":
    main()
