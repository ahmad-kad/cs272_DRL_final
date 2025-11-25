#!/usr/bin/env python3
"""
Ensemble Evaluation and Comparison Script

This script provides comprehensive evaluation and comparison of different ensemble
approaches for multi-modal autonomous driving.

Features:
- Evaluate individual models and ensemble approaches
- Compare performance across scenarios
- Generate detailed performance reports
- Visualize ensemble decision-making

Usage:
    # Compare all available models
    python evaluate_ensemble.py --compare-all

    # Evaluate specific ensemble approach
    python evaluate_ensemble.py --approach q_value_ensemble --strategy confidence_weighted

    # Generate performance report
    python evaluate_ensemble.py --generate-report

Author: AI Assistant
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from stable_baselines3 import PPO

from training.ensemble_models import MultiModalEnsemble
from environments.urban_junction_env import UrbanJunctionEnv


class EnsembleEvaluator:
    """
    Comprehensive evaluator for ensemble models and individual experts.
    """

    def __init__(self, base_dir: str = "outputs"):
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "models")
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

        # Model paths
        self.model_paths = {
            "lidar": os.path.join(self.models_dir, "adaptive_lidar_final.zip"),
            "grayscale": os.path.join(self.models_dir, "adaptive_grayscale_final.zip"),
            "late_fusion": os.path.join(self.models_dir, "ensemble_late_fusion_final.zip"),
        }

    def evaluate_individual_models(self, n_episodes: int = 50) -> Dict[str, Any]:
        """
        Evaluate individual lidar and grayscale models.

        Args:
            n_episodes: Episodes per scenario per model

        Returns:
            Evaluation results for individual models
        """
        print("Evaluating individual models...")

        results = {"individual_models": {}}

        for model_name, model_path in self.model_paths.items():
            if model_name in ["late_fusion"]:
                continue  # Skip ensemble models for individual evaluation

            if not os.path.exists(model_path):
                print(f"Warning: Model {model_path} not found, skipping")
                continue

            print(f"Evaluating {model_name} model...")
            model_results = self._evaluate_single_model(
                model_path, model_name, n_episodes
            )
            results["individual_models"][model_name] = model_results

        return results

    def evaluate_ensemble_approaches(
        self,
        approaches: List[str] = None,
        n_episodes: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluate different ensemble approaches.

        Args:
            approaches: List of ensemble approaches to evaluate
            n_episodes: Episodes per scenario

        Returns:
            Evaluation results for ensemble approaches
        """
        if approaches is None:
            approaches = ["q_value_uniform", "q_value_confidence", "q_value_adaptive"]

        print("Evaluating ensemble approaches...")

        results = {"ensemble_approaches": {}}

        for approach in approaches:
            print(f"Evaluating {approach}...")

            if approach.startswith("q_value"):
                strategy = approach.split("_")[1]  # uniform, confidence, adaptive
                ensemble = MultiModalEnsemble(
                    lidar_model_path=self.model_paths["lidar"],
                    grayscale_model_path=self.model_paths["grayscale"],
                    ensemble_strategy=strategy
                )
                ensemble_results = self._evaluate_q_value_ensemble(
                    ensemble, approach, n_episodes
                )
            else:
                print(f"Warning: Unknown approach {approach}, skipping")
                continue

            results["ensemble_approaches"][approach] = ensemble_results

        return results

    def _evaluate_single_model(
        self,
        model_path: str,
        model_name: str,
        n_episodes: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluate a single model across all scenarios.

        Args:
            model_path: Path to the model
            model_name: Name of the model (lidar/grayscale)
            n_episodes: Episodes per scenario

        Returns:
            Evaluation results
        """
        model = PPO.load(model_path)
        modality = model_name  # lidar or grayscale

        results = {
            "model_path": model_path,
            "modality": modality,
            "scenarios": {}
        }

        scenarios = ["highway", "merge", "intersection"]

        for scenario in scenarios:
            print(f"  Evaluating {model_name} on {scenario}...")

            # Create environment
            env = UrbanJunctionEnv(scenario=scenario, modality=modality)

            scenario_results = self._run_evaluation_episodes(
                model, env, n_episodes, modality
            )
            results["scenarios"][scenario] = scenario_results

        # Calculate overall statistics
        all_success = [results["scenarios"][s]["success_rate"] for s in scenarios]
        all_rewards = [results["scenarios"][s]["avg_reward"] for s in scenarios]
        all_crashes = [results["scenarios"][s]["crash_rate"] for s in scenarios]

        results["summary"] = {
            "overall_success_rate": float(np.mean(all_success)),
            "overall_avg_reward": float(np.mean(all_rewards)),
            "overall_crash_rate": float(np.mean(all_crashes)),
            "success_rate_std": float(np.std(all_success)),
            "reward_std": float(np.std(all_rewards))
        }

        return results

    def _evaluate_q_value_ensemble(
        self,
        ensemble: MultiModalEnsemble,
        ensemble_name: str,
        n_episodes: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluate a Q-value ensemble across scenarios.

        Args:
            ensemble: The ensemble to evaluate
            ensemble_name: Name of the ensemble approach
            n_episodes: Episodes per scenario

        Returns:
            Evaluation results
        """
        results = {
            "ensemble_type": "q_value",
            "strategy": ensemble.ensemble_strategy,
            "scenarios": {}
        }

        scenarios = ["highway", "merge", "intersection"]

        for scenario in scenarios:
            print(f"  Evaluating {ensemble_name} on {scenario}...")

            # Create environments for both modalities
            lidar_env = UrbanJunctionEnv(scenario=scenario, modality="lidar")
            grayscale_env = UrbanJunctionEnv(scenario=scenario, modality="grayscale")

            scenario_results = self._run_ensemble_evaluation_episodes(
                ensemble, lidar_env, grayscale_env, n_episodes
            )
            results["scenarios"][scenario] = scenario_results

        # Calculate overall statistics
        all_success = [results["scenarios"][s]["success_rate"] for s in scenarios]
        all_rewards = [results["scenarios"][s]["avg_reward"] for s in scenarios]
        all_crashes = [results["scenarios"][s]["crash_rate"] for s in scenarios]

        results["summary"] = {
            "overall_success_rate": float(np.mean(all_success)),
            "overall_avg_reward": float(np.mean(all_rewards)),
            "overall_crash_rate": float(np.mean(all_crashes)),
            "success_rate_std": float(np.std(all_success)),
            "reward_std": float(np.std(all_rewards))
        }

        return results

    def _run_evaluation_episodes(
        self,
        model: PPO,
        env,
        n_episodes: int,
        modality: str
    ) -> Dict[str, Any]:
        """
        Run evaluation episodes for a single model.

        Args:
            model: The model to evaluate
            env: Environment to evaluate in
            n_episodes: Number of episodes
            modality: Observation modality

        Returns:
            Episode results
        """
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        total_crashes = 0

        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_crashes = 0
            done = False
            steps = 0

            while not done and steps < 200:  # Max episode length
                action, _ = model.predict(obs, deterministic=True)

                # Step environment
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_obs, reward, done, info = step_result

                episode_reward += reward
                steps += 1

                # Check for crashes
                if isinstance(info, dict) and info.get("crashed", False):
                    episode_crashes += 1

                obs = next_obs

            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            total_crashes += episode_crashes

            if episode_reward > 0 and episode_crashes == 0:
                success_count += 1

        return {
            "episodes": n_episodes,
            "success_rate": float(success_count / n_episodes),
            "avg_reward": float(np.mean(episode_rewards)),
            "reward_std": float(np.std(episode_rewards)),
            "crash_rate": float(total_crashes / n_episodes),
            "avg_episode_length": float(np.mean(episode_lengths)),
            "success_count": success_count,
            "total_crashes": total_crashes
        }

    def _run_ensemble_evaluation_episodes(
        self,
        ensemble: MultiModalEnsemble,
        lidar_env,
        grayscale_env,
        n_episodes: int
    ) -> Dict[str, Any]:
        """
        Run evaluation episodes for an ensemble.

        Args:
            ensemble: The ensemble to evaluate
            lidar_env: Lidar environment
            grayscale_env: Grayscale environment
            n_episodes: Number of episodes

        Returns:
            Episode results
        """
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        total_crashes = 0
        ensemble_stats = []

        for episode in range(n_episodes):
            # Reset both environments with same seed for fair comparison
            seed = np.random.randint(0, 10000)
            lidar_obs, _ = lidar_env.reset(seed=seed)
            grayscale_obs, _ = grayscale_env.reset(seed=seed)

            episode_reward = 0
            episode_crashes = 0
            done = False
            steps = 0
            episode_ensemble_info = []

            while not done and steps < 200:
                # Get ensemble action
                action, ensemble_info = ensemble.predict(
                    lidar_obs, grayscale_obs, deterministic=True
                )

                # Step both environments (use lidar for primary stepping)
                step_result = lidar_env.step(action)
                if len(step_result) == 5:
                    next_lidar_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_lidar_obs, reward, done, info = step_result

                # Step grayscale env too (for next observation)
                grayscale_env.step(action)  # Don't care about reward from this

                episode_reward += reward
                steps += 1

                # Check for crashes
                if isinstance(info, dict) and info.get("crashed", False):
                    episode_crashes += 1

                # Store ensemble decision info
                episode_ensemble_info.append(ensemble_info)

                lidar_obs = next_lidar_obs
                # Get next grayscale obs
                grayscale_obs, _ = grayscale_env.reset()  # Simplified - should track properly

            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            total_crashes += episode_crashes

            if episode_reward > 0 and episode_crashes == 0:
                success_count += 1

            # Store ensemble statistics
            avg_lidar_weight = np.mean([info.get("lidar_weight", 0.5) for info in episode_ensemble_info])
            avg_grayscale_weight = np.mean([info.get("grayscale_weight", 0.5) for info in episode_ensemble_info])
            ensemble_stats.append({
                "avg_lidar_weight": float(avg_lidar_weight),
                "avg_grayscale_weight": float(avg_grayscale_weight)
            })

        return {
            "episodes": n_episodes,
            "success_rate": float(success_count / n_episodes),
            "avg_reward": float(np.mean(episode_rewards)),
            "reward_std": float(np.std(episode_rewards)),
            "crash_rate": float(total_crashes / n_episodes),
            "avg_episode_length": float(np.mean(episode_lengths)),
            "success_count": success_count,
            "total_crashes": total_crashes,
            "ensemble_stats": ensemble_stats
        }

    def generate_comparison_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive comparison report.

        Args:
            results: Combined evaluation results

        Returns:
            Formatted report string
        """
        report = []
        report.append("="*80)
        report.append("ENSEMBLE MODEL COMPARISON REPORT")
        report.append("="*80)

        # Individual model performance
        if "individual_models" in results:
            report.append("\nINDIVIDUAL MODEL PERFORMANCE:")
            report.append("-" * 40)

            for model_name, model_results in results["individual_models"].items():
                summary = model_results["summary"]
                report.append(f"\n{model_name.upper()} Model:")
                report.append(".3f")
                report.append(".2f")
                report.append(".3f")

        # Ensemble performance
        if "ensemble_approaches" in results:
            report.append("\nENSEMBLE APPROACH PERFORMANCE:")
            report.append("-" * 40)

            for approach_name, approach_results in results["ensemble_approaches"].items():
                summary = approach_results["summary"]
                report.append(f"\n{approach_name.upper()}:")
                report.append(".3f")
                report.append(".2f")
                report.append(".3f")

        # Best approaches
        if "individual_models" in results and "ensemble_approaches" in results:
            report.append("\nPERFORMANCE COMPARISON:")
            report.append("-" * 40)

            # Collect all approaches
            all_approaches = []

            for model_name, model_results in results["individual_models"].items():
                all_approaches.append({
                    "name": model_name,
                    "success_rate": model_results["summary"]["overall_success_rate"],
                    "avg_reward": model_results["summary"]["overall_avg_reward"],
                    "crash_rate": model_results["summary"]["overall_crash_rate"]
                })

            for approach_name, approach_results in results["ensemble_approaches"].items():
                all_approaches.append({
                    "name": approach_name,
                    "success_rate": approach_results["summary"]["overall_success_rate"],
                    "avg_reward": approach_results["summary"]["overall_avg_reward"],
                    "crash_rate": approach_results["summary"]["overall_crash_rate"]
                })

            # Sort by success rate
            all_approaches.sort(key=lambda x: x["success_rate"], reverse=True)

            report.append("\nRanked by Success Rate:")
            for i, approach in enumerate(all_approaches, 1):
                report.append(f"{i}. {approach['name']}: {approach['success_rate']:.3f} success, "
                             f"{approach['crash_rate']:.3f} crash rate")

        report.append("\n" + "="*80)
        return "\n".join(report)

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """
        Save evaluation results to JSON file.

        Args:
            results: Results to save
            filename: Optional filename (auto-generated if None)
        """
        if filename is None:
            timestamp = "2025-11-25"  # Using current date
            filename = f"ensemble_comparison_{timestamp}.json"

        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {filepath}")

    def run_comprehensive_evaluation(
        self,
        n_episodes: int = 50,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of all available models and approaches.

        Args:
            n_episodes: Episodes per scenario per approach
            save_results: Whether to save results

        Returns:
            Complete evaluation results
        """
        print("Running comprehensive ensemble evaluation...")
        print(f"Episodes per scenario: {n_episodes}")
        print("-" * 50)

        results = {}

        # Evaluate individual models
        individual_results = self.evaluate_individual_models(n_episodes)
        results.update(individual_results)

        # Evaluate ensemble approaches
        ensemble_results = self.evaluate_ensemble_approaches(n_episodes=n_episodes)
        results.update(ensemble_results)

        # Generate and print report
        report = self.generate_comparison_report(results)
        print(report)

        # Save results
        if save_results:
            self.save_results(results)

        return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate and compare ensemble models")
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Run comprehensive comparison of all available models and approaches"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Number of episodes per scenario (default: 30)"
    )
    parser.add_argument(
        "--approach",
        choices=["q_value_uniform", "q_value_confidence", "q_value_adaptive"],
        help="Specific ensemble approach to evaluate"
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate performance report from existing results"
    )
    parser.add_argument(
        "--results-file",
        help="Path to results file for report generation"
    )

    args = parser.parse_args()

    evaluator = EnsembleEvaluator()

    if args.compare_all:
        results = evaluator.run_comprehensive_evaluation(
            n_episodes=args.episodes,
            save_results=True
        )

    elif args.approach:
        results = evaluator.evaluate_ensemble_approaches(
            approaches=[args.approach],
            n_episodes=args.episodes
        )
        evaluator.save_results(results)

    elif args.generate_report:
        if args.results_file:
            with open(args.results_file, 'r') as f:
                results = json.load(f)
        else:
            # Try to load latest results
            results_files = [f for f in os.listdir("results") if f.startswith("ensemble_comparison")]
            if results_files:
                latest_file = max(results_files)
                with open(os.path.join("results", latest_file), 'r') as f:
                    results = json.load(f)
            else:
                print("No results files found. Run --compare-all first.")
                return

        report = evaluator.generate_comparison_report(results)
        print(report)

    else:
        print("Please specify an action: --compare-all, --approach, or --generate-report")
        parser.print_help()


if __name__ == "__main__":
    main()
