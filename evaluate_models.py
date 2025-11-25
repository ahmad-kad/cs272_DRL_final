#!/usr/bin/env python3
"""
Model Evaluation Script

This script demonstrates how to use the standardized evaluation configurations
to evaluate trained models across different driving scenarios and modalities.

Features:
- Comprehensive evaluation across highway, merge, and intersection scenarios
- Support for both lidar and grayscale observation modalities
- Optional real-time visualization for human observation
- Detailed performance metrics and statistics
- JSON result export for analysis

Usage:
    python evaluate_models.py model.zip --visualize --episodes 10
    python evaluate_models.py model.zip --results-dir custom_results
"""

import os
import json
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import time
from typing import Dict, Any, List
from environments.urban_junction_env import UrbanJunctionEnv
from utils.evaluation_configs import (
    get_evaluation_config,
    get_all_evaluation_configs,
    EVALUATION_CONFIG_SUMMARY
)

def evaluate_model_on_scenario(
    model_path: str,
    scenario: str,
    modality: str,
    n_episodes: int = 50,
    deterministic: bool = True,
    seed: int = 42,
    visualize: bool = False,
    render_delay: float = 0.05
) -> Dict[str, Any]:
    """
    Evaluate a single model on a specific scenario and modality.

    Args:
        model_path: Path to the saved model
        scenario: Scenario to evaluate ("highway", "merge", "intersection")
        modality: Modality ("lidar", "grayscale")
        n_episodes: Number of evaluation episodes
        deterministic: Whether to use deterministic actions
        seed: Random seed for reproducibility
        visualize: Whether to render the environment for human viewing
        render_delay: Delay between frames when visualizing (seconds)

    Returns:
        Dictionary containing evaluation results
    """
    print(f"\nEvaluating {model_path} on {scenario} ({modality})...")
    if visualize:
        print(f"  Visualization enabled (render delay: {render_delay}s)")
        print("  Close the visualization window to continue evaluation...")

    # Load model
    try:
        model = PPO.load(model_path)
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {e}")

    # Get evaluation configuration
    config = get_evaluation_config(scenario, modality)

    # Create environment
    env = UrbanJunctionEnv(
        config=config,
        scenario=scenario,
        modality=modality
    )

    # Set seeds for reproducibility
    env.reset(seed=seed)
    model.set_random_seed(seed)

    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    crash_count = 0
    total_crashes = 0

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_crashes = 0
        done = False
        steps = 0

        while not done and steps < config["duration"] * 2:  # Safety limit
            action, _ = model.predict(obs, deterministic=deterministic)

            # Handle environment step
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
                total_crashes += 1

            # Render if visualization is enabled
            if visualize:
                env.render()
                time.sleep(render_delay)

            obs = next_obs

        # Record episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        crash_count += episode_crashes

        # Success criteria: positive reward and no crashes
        if episode_reward > 0 and episode_crashes == 0:
            success_count += 1

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{n_episodes}: Reward={episode_reward:.2f}, "
                  f"Crashes={episode_crashes}, Success={episode_reward > 0 and episode_crashes == 0}")

    # Calculate statistics
    results = {
        "scenario": scenario,
        "modality": modality,
        "episodes": n_episodes,
        "success_rate": float(success_count / n_episodes),
        "crash_rate": float(total_crashes / n_episodes),
        "avg_reward": float(np.mean(episode_rewards)),
        "reward_std": float(np.std(episode_rewards)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "success_count": success_count,
        "total_crashes": total_crashes,
        "model_path": model_path
    }

    env.close()
    return results

def evaluate_model_comprehensive(
    model_path: str,
    n_episodes_per_scenario: int = 50,
    save_results: bool = True,
    results_dir: str = "results",
    visualize: bool = False,
    render_delay: float = 0.05
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a model across all scenarios and modalities.

    Args:
        model_path: Path to the saved model
        n_episodes_per_scenario: Episodes to evaluate per scenario-modality pair
        save_results: Whether to save results to JSON file
        results_dir: Directory to save results
        visualize: Whether to render the environment for human viewing
        render_delay: Delay between frames when visualizing (seconds)

    Returns:
        Complete evaluation results dictionary
    """
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE MODEL EVALUATION")
    print(f"Model: {model_path}")
    if visualize:
        print(f"Visualization: ENABLED (delay: {render_delay}s)")
        print("Note: Visualization will open separate windows for each scenario")
    else:
        print("Visualization: DISABLED")
    print(f"{'='*60}")

    # Extract model info from path
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    modality = "grayscale" if "grayscale" in model_name.lower() else "lidar"

    results = {
        "model_path": model_path,
        "model_name": model_name,
        "modality": modality,
        "evaluation_config": EVALUATION_CONFIG_SUMMARY,
        "scenarios": {}
    }

    all_success_rates = []
    all_crash_rates = []
    all_rewards = []

    # Evaluate on each scenario
    for scenario in ["highway", "merge", "intersection"]:
        try:
            scenario_results = evaluate_model_on_scenario(
                model_path=model_path,
                scenario=scenario,
                modality=modality,
                n_episodes=n_episodes_per_scenario,
                visualize=visualize,
                render_delay=render_delay
            )
            results["scenarios"][scenario] = scenario_results

            # Collect for summary stats
            all_success_rates.append(scenario_results["success_rate"])
            all_crash_rates.append(scenario_results["crash_rate"])
            all_rewards.append(scenario_results["avg_reward"])

        except Exception as e:
            print(f"Error evaluating {scenario}: {e}")
            results["scenarios"][scenario] = {"error": str(e)}

    # Calculate overall statistics
    if all_success_rates:
        results["summary"] = {
            "overall_success_rate": float(np.mean(all_success_rates)),
            "overall_crash_rate": float(np.mean(all_crash_rates)),
            "overall_avg_reward": float(np.mean(all_rewards)),
            "success_rate_std": float(np.std(all_success_rates)),
            "crash_rate_std": float(np.std(all_crash_rates)),
            "avg_reward_std": float(np.std(all_rewards))
        }

        print("\nOVERALL RESULTS:")
        print(".3f")
        print(".3f")
        print(".2f")
    # Save results if requested
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f"{model_name}_evaluation.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    return results

def main():
    """Main evaluation function demonstrating usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained models using standardized configurations")
    parser.add_argument("model_path", help="Path to the trained model (.zip file)")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per scenario (default: 50)")
    parser.add_argument("--results-dir", default="results", help="Directory to save results")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization (render environment for human viewing)")
    parser.add_argument("--render-delay", type=float, default=0.05, help="Delay between frames when visualizing (seconds, default: 0.05)")

    args = parser.parse_args()

    # Run comprehensive evaluation
    results = evaluate_model_comprehensive(
        model_path=args.model_path,
        n_episodes_per_scenario=args.episodes,
        save_results=not args.no_save,
        results_dir=args.results_dir,
        visualize=args.visualize,
        render_delay=args.render_delay
    )

    print(f"\nEvaluation complete for {args.model_path}")

if __name__ == "__main__":
    main()
