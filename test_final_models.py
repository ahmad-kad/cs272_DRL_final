#!/usr/bin/env python3
"""
Test Final Models on Different Environments

This script evaluates all final trained models across different scenarios and modalities
to compare their performance and robustness.

Models tested:
- adaptive_both_final.zip: Multi-modal curriculum trained (lidar + grayscale)
- adaptive_grayscale_final.zip: Single-modal grayscale trained
- adaptive_lidar_final.zip: Single-modal lidar trained
- ensemble_late_fusion_final.zip: Late fusion ensemble

Scenarios tested: highway, merge, intersection
Modalities tested: lidar, grayscale, both

Author: AI Assistant
"""

import os
import json
import numpy as np
from stable_baselines3 import PPO
from typing import Dict, Any, List
import time

from environments.urban_junction_env import UrbanJunctionEnv
from utils.config import get_curriculum_config


def load_model(model_path: str, env) -> PPO:
    """Load a PPO model from path."""
    print(f"Loading model: {model_path}")
    try:
        model = PPO.load(model_path, env=env, device="cuda" if hasattr(env, 'device') and env.device == "cuda" else "cpu")
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None


def evaluate_model_on_scenario(
    model_path: str,
    scenario: str,
    modality: str,
    n_episodes: int = 20,
    deterministic: bool = True,
    visualize: bool = False,
    render_delay: float = 0.05
) -> Dict[str, Any]:
    """
    Evaluate a single model on a specific scenario and modality.

    Args:
        model_path: Path to the saved model
        scenario: Scenario to evaluate ("highway", "merge", "intersection")
        modality: Modality ("lidar", "grayscale", "both")
        n_episodes: Number of evaluation episodes
        deterministic: Whether to use deterministic actions
        visualize: Whether to render the environment
        render_delay: Delay between frames when visualizing

    Returns:
        Dictionary containing evaluation results
    """
    print(f"\nEvaluating {os.path.basename(model_path)} on {scenario} ({modality})...")

    # Get evaluation configuration (use hard difficulty for evaluation)
    config = get_curriculum_config(scenario, "hard", modality)

    # Create environment
    env = UrbanJunctionEnv(
        config=config,
        scenario=scenario,
        modality=modality
    )

    # Load model
    model = load_model(model_path, env)
    if model is None:
        return {"error": f"Failed to load model {model_path}"}

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

        while not done and steps < 200:
            action, _ = model.predict(obs, deterministic=deterministic)

            step_result = env.step(action)
            if isinstance(step_result, tuple) and len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                # Handle older gym API
                next_obs, reward, done, info = step_result

            episode_reward += reward
            steps += 1

            # Check for crashes
            if isinstance(info, dict) and info.get("crashed", False):
                episode_crashes += 1

            # Render if visualization is enabled
            if visualize:
                env.render()
                time.sleep(render_delay)

            obs = next_obs

        # Record episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        total_crashes += episode_crashes

        # Success criteria: positive reward and no crashes
        if episode_reward > 0 and episode_crashes == 0:
            success_count += 1

        if (episode + 1) % 5 == 0:
            print(f"  Episode {episode + 1}/{n_episodes}: Reward={episode_reward:.2f}, "
                  f"Crashes={episode_crashes}, Success={episode_reward > 0 and episode_crashes == 0}")

    env.close()

    results = {
        "episodes": n_episodes,
        "success_rate": float(success_count / n_episodes),
        "avg_reward": float(np.mean(episode_rewards)),
        "reward_std": float(np.std(episode_rewards)),
        "crash_rate": float(total_crashes / n_episodes),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "success_count": success_count,
        "total_crashes": total_crashes,
        "scenario": scenario,
        "modality": modality,
        "model": os.path.basename(model_path)
    }

    print(f"  Results: Success Rate = {results['success_rate']:.3f}, "
          f"Avg Reward = {results['avg_reward']:.2f}, Crash Rate = {results['crash_rate']:.3f}")

    return results


def test_all_models():
    """Test all final models on all scenarios and modalities."""

    # Define models to test
    models = {
        "Multi-Modal (Both)": "outputs/models/adaptive_both_final.zip",
        "Grayscale Only": "outputs/models/adaptive_grayscale_final.zip",
        "Lidar Only": "outputs/models/adaptive_lidar_final.zip",
        "Late Fusion": "outputs/models/ensemble_late_fusion_final.zip"
    }

    # Define test scenarios and modalities
    scenarios = ["highway", "merge", "intersection"]
    modalities = ["lidar", "grayscale", "both"]

    # Store results
    all_results = {}
    summary_results = {}

    print("="*80)
    print("FINAL MODEL COMPARISON ACROSS ENVIRONMENTS")
    print("="*80)

    # Test each model
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"WARNING:  Model not found: {model_path}")
            continue

        print(f"\n[*] Testing {model_name}")
        print("-" * 50)

        model_results = {}

        # Test on each scenario
        for scenario in scenarios:
            scenario_results = {}

            # Test on each modality (but only compatible ones)
            for modality in modalities:
                # Only test compatible modality combinations
                if model_name == "Grayscale Only" and modality != "grayscale":
                    continue
                if model_name == "Lidar Only" and modality != "lidar":
                    continue
                if model_name == "Multi-Modal (Both)" and modality != "both":
                    continue
                if model_name == "Late Fusion" and modality != "both":
                    continue

                try:
                    results = evaluate_model_on_scenario(
                        model_path=model_path,
                        scenario=scenario,
                        modality=modality,
                        n_episodes=args.episodes,
                        visualize=False
                    )

                    if "error" not in results:
                        scenario_results[modality] = results
                    else:
                        print(f"ERROR: Error testing {model_name} on {scenario} ({modality}): {results['error']}")

                except Exception as e:
                    print(f"ERROR: Exception testing {model_name} on {scenario} ({modality}): {e}")
                    continue

            if scenario_results:
                model_results[scenario] = scenario_results

        if model_results:
            all_results[model_name] = model_results

            # Calculate summary statistics for this model
            all_success_rates = []
            all_rewards = []
            all_crashes = []

            for scenario, scenario_data in model_results.items():
                for modality, modality_data in scenario_data.items():
                    all_success_rates.append(modality_data['success_rate'])
                    all_rewards.append(modality_data['avg_reward'])
                    all_crashes.append(modality_data['crash_rate'])

            summary_results[model_name] = {
                "overall_success_rate": float(np.mean(all_success_rates)),
                "overall_avg_reward": float(np.mean(all_rewards)),
                "overall_crash_rate": float(np.mean(all_crashes)),
                "success_std": float(np.std(all_success_rates)),
                "reward_std": float(np.std(all_rewards)),
                "scenarios_tested": len(model_results),
                "total_tests": sum(len(s) for s in model_results.values())
            }

    # Print comprehensive results
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    print("\n[STATS] Overall Performance by Model:")
    print("-" * 60)

    # Sort models by success rate
    sorted_models = sorted(summary_results.items(),
                          key=lambda x: x[1]['overall_success_rate'],
                          reverse=True)

    for model_name, stats in sorted_models:
        print(f"\n[WIN] {model_name}")
        print(".3f")
        print(".2f")
        print(".3f")
        print(f"   Scenarios Tested: {stats['scenarios_tested']}")
        print(f"   Total Test Combinations: {stats['total_tests']}")

    # Detailed breakdown
    print("\n[DETAILS] Detailed Results by Scenario and Modality:")
    print("-" * 60)

    for model_name, model_data in all_results.items():
        print(f"\n[*] {model_name}:")
        for scenario, scenario_data in model_data.items():
            print(f"  [SCENARIO] {scenario.upper()}:")
            for modality, results in scenario_data.items():
                success_rate = results['success_rate']
                avg_reward = results['avg_reward']
                crash_rate = results['crash_rate']

                # Color coding for performance
                if success_rate >= 0.8:
                    status = "EXCELLENT"
                elif success_rate >= 0.6:
                    status = "GOOD"
                elif success_rate >= 0.4:
                    status = "FAIR"
                else:
                    status = "POOR"

                print(".3f")

    # Save results
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save detailed results
    detailed_file = f"results/final_model_comparison_{timestamp}.json"
    with open(detailed_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "models_tested": list(all_results.keys()),
            "detailed_results": all_results,
            "summary_results": summary_results
        }, f, indent=2)

    print(f"\n[SAVE] Detailed results saved to: {detailed_file}")

    # Save summary table
    summary_file = f"results/final_model_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("FINAL MODEL COMPARISON SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write("Overall Performance Rankings:\n\n")

        for i, (model_name, stats) in enumerate(sorted_models, 1):
            f.write(f"{i}. {model_name}\n")
            f.write(".3f")
            f.write(".2f")
            f.write(".3f")
            f.write(f"   Scenarios Tested: {stats['scenarios_tested']}\n")
            f.write(f"   Total Combinations: {stats['total_tests']}\n\n")

    print(f"[SAVE] Summary table saved to: {summary_file}")

    return all_results, summary_results


def visualize_best_model():
    """Visualize the best performing model on a highway scenario."""
    print("\n[VISUALIZE] Visualizing Best Model on Highway Scenario...")

    # Determine best model from summary (this would need to be passed in)
    best_model_path = "outputs/models/adaptive_both_final.zip"  # Assume multi-modal is best

    try:
        results = evaluate_model_on_scenario(
            model_path=best_model_path,
            scenario="highway",
            modality="both",
            n_episodes=3,  # Just a few episodes for visualization
            visualize=True,
            render_delay=0.1
        )

        print("Visualization completed!")
        return results

    except Exception as e:
        print(f"ERROR: Visualization failed: {e}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test final models on different environments")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization for best model")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes per test")

    args = parser.parse_args()

    # Run comprehensive testing
    all_results, summary_results = test_all_models()

    # Optional visualization
    if args.visualize:
        visualize_best_model()

    print("\n[SUCCESS] Model testing completed!")
    print("\n[INSIGHT] Key Insights:")
    print("   - Multi-modal models should perform best on 'both' modality")
    print("   - Single-modal models excel on their native modality")
    print("   - Late fusion combines modalities at the policy level")
    print("   - Curriculum-trained models should be more robust across scenarios")
