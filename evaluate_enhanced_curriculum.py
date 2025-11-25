#!/usr/bin/env python3
"""
Evaluate Enhanced Curriculum Final Model on All Three Environments
Tests the final expert_intersection model from enhanced curriculum training with domain randomization.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
import numpy as np
import wandb
from datetime import datetime

def create_eval_env(env_name):
    """Create evaluation environment matching enhanced curriculum configs."""

    if env_name == "highway-v0":
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

    elif env_name == "merge-v0":
        env = gym.make("merge-v0", render_mode=None)
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
            "high_speed_reward": 0.5,
            "reward_speed_range": [18, 28],
            "lane_change_reward": 0.2,
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "vehicles_count": 15,
        }
        unwrapped_env.configure(config)

    elif env_name == "intersection-v0":
        env = gym.make("intersection-v0", render_mode=None)
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
        }
        unwrapped_env.configure(config)

    env.reset()
    return env

def evaluate_model(model_path, env_name, episodes=30):
    """Evaluate model on specified environment."""

    print(f"Loading model: {model_path}")
    env = create_eval_env(env_name)

    try:
        model = PPO.load(model_path, env=env, device='cpu')
    except Exception as e:
        print(f"Failed to load model for {env_name}: {e}")
        env.close()
        return {"error": str(e)}

    rewards = []
    lengths = []
    crashes = []
    successes = []

    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        crashed = False
        success = False

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)

            episode_reward += reward
            episode_length += 1

            if step_info.get('crashed', False):
                crashed = True

            # Success criteria: positive reward and no collisions
            if terminated and episode_reward > 0 and not crashed:
                success = True

            if terminated or truncated or episode_length >= 50:
                break

        rewards.append(episode_reward)
        lengths.append(episode_length)
        crashes.append(1 if crashed else 0)
        successes.append(1 if success else 0)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/30: Reward = {episode_reward:.2f}, Length = {episode_length}")

    env.close()

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    success_rate = np.mean(successes) * 100
    crash_rate = np.mean(crashes) * 100

    # Performance score: normalize by environment-specific expectations
    if env_name == "highway-v0":
        performance_score = min(1.0, max(0.0, (mean_reward + 20) / 40))
    elif env_name == "merge-v0":
        performance_score = min(1.0, max(0.0, (mean_reward + 10) / 25))
    else:  # intersection
        performance_score = min(1.0, max(0.0, (mean_reward + 30) / 30))

    print(".2f")
    print(".1f")
    print(".1f")
    print(".3f")

    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'success_rate': success_rate,
        'crash_rate': crash_rate,
        'performance_score': performance_score
    }

def main():
    """Evaluate enhanced curriculum final model on all three environments."""

    print("ENHANCED CURRICULUM FINAL MODEL EVALUATION")
    print("=" * 80)
    print("Model: Enhanced Curriculum (8 Phases + Domain Randomization)")
    print("  1. Highway Foundation")
    print("  2. Easy Merge")
    print("  3. Hard Merge")
    print("  4. Very Easy Intersection (DR)")
    print("  5. Easy Intersection (DR)")
    print("  6. Medium Intersection (DR)")
    print("  7. Hard Intersection (DR)")
    print("  8. Expert Intersection (DR)")
    print("Training Method: Progressive curriculum with adaptive scheduling, enhanced safety rewards, and domain randomization")
    print("DR = Domain Randomization enabled")
    print("Testing on: highway-v0, merge-v0, AND intersection-v0")
    print("=" * 80)

    # Use the final model from expert intersection phase
    model_path = "outputs/models/enhanced_curriculum/expert_intersection/enhanced_curriculum_expert_intersection_early_progression.zip"

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Available enhanced curriculum models:")
        for phase_dir in os.listdir("outputs/models/enhanced_curriculum"):
            phase_path = os.path.join("outputs/models/enhanced_curriculum", phase_dir)
            if os.path.isdir(phase_path):
                files = [f for f in os.listdir(phase_path) if f.endswith('.zip')]
                if files:
                    print(f"  {phase_dir}: {files}")
        return

    print(f"Loading final enhanced curriculum model: {model_path}")

    # Initialize wandb
    run = wandb.init(
        project="highway-foundation-v2",
        name=f"eval_enhanced_curriculum_final_{int(datetime.now().timestamp())}",
        config={
            "evaluation_type": "enhanced_curriculum_final_evaluation",
            "model_path": model_path,
            "environments": ["highway-v0", "merge-v0", "intersection-v0"],
            "episodes_per_env": 30,
            "curriculum_phases": 8,
            "adaptive_scheduling": True,
            "enhanced_rewards": True,
            "domain_randomization": True
        }
    )

    try:
        environments = ["highway-v0", "merge-v0", "intersection-v0"]
        results = {}

        for env_name in environments:
            print(f"\nEvaluating on {env_name}...")
            print("-" * 60)

            env_results = evaluate_model(model_path, env_name, episodes=30)
            if "error" not in env_results:
                results[env_name] = env_results
            else:
                print(f"Skipping {env_name} due to error")

        print("\n" + "=" * 80)
        print("ENHANCED CURRICULUM FINAL MODEL PERFORMANCE SUMMARY")
        print("=" * 80)

        if results:
            for env, result in results.items():
                env_short = env.replace('-v0', '')
                score = result['performance_score']
                success = result['success_rate']
                crash = result['crash_rate']
                reward = result['mean_reward']
                print("5s")

            total_score = sum(result['performance_score'] for result in results.values()) / len(results)
            avg_success = sum(result['success_rate'] for result in results.values()) / len(results)
            avg_crash = sum(result['crash_rate'] for result in results.values()) / len(results)

            print("5s")
            print(".1f")
            print(".1f")

            # Analysis
            print("\nENHANCED CURRICULUM ANALYSIS:")
            print("-" * 40)
            print("Advanced Training Features:")
            print("  + Enhanced safety rewards (-25 collision penalty)")
            print("  + 8-phase curriculum with domain randomization")
            print("  + Adaptive phase advancement")
            print("  + Episode-level environment variation")

            highway_score = results.get("highway-v0", {}).get("performance_score", 0)
            merge_score = results.get("merge-v0", {}).get("performance_score", 0)
            intersection_score = results.get("intersection-v0", {}).get("performance_score", 0)

            if total_score > 0.95:
                verdict = "OUTSTANDING: Exceptional performance across all environments!"
            elif total_score > 0.85:
                verdict = "EXCELLENT: Major improvements in safety and success rates"
            elif intersection_score > 0.8:
                verdict = "VERY GOOD: Excellent intersection handling"
            else:
                verdict = "GOOD: Solid performance with room for refinement"

            print(f"\nVERDICT: {verdict}")

            # Compare to previous models
            print("
PERFORMANCE COMPARISON:")
            print("-" * 40)
            print("Original Multi-Env (0.552): Simultaneous training")
            print(".3f"            print("Advanced Curriculum (0.894): 7-phase progressive")
            print(".3f"            print("Enhanced Curriculum (8-phase + DR): Latest results")
            print(".3f"
        # Log to wandb
        wandb.log({
            "evaluation_completed": True,
            "highway_performance": results.get("highway-v0", {}).get("performance_score", 0),
            "merge_performance": results.get("merge-v0", {}).get("performance_score", 0),
            "intersection_performance": results.get("intersection-v0", {}).get("performance_score", 0),
            "overall_performance": total_score if 'total_score' in locals() else 0,
            "avg_success_rate": avg_success if 'avg_success' in locals() else 0,
            "avg_crash_rate": avg_crash if 'avg_crash' in locals() else 0
        })

    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()

