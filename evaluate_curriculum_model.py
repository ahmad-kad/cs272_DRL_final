#!/usr/bin/env python3
"""
Evaluate Curriculum Transfer Model: Highway → Merge → Intersection
Tests the curriculum model across all three environments separately.
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
    """Create evaluation environment with curriculum-compatible configs."""
    if env_name == "highway-v0":
        env = gym.make("highway-v0", render_mode=None)
        unwrapped_env = env.unwrapped

        # Curriculum-compatible highway config
        highway_config = {
            "observation": {
                "type": "LidarObservation",
                "cells": 32,  # Match curriculum model's 32 cells
                "row_anchor": [0.5, 0.5],
                "features": ["presence", "distance", "speed"],
                "features_range": {"distance": [0, 50], "speed": [-30, 30]}
            },
            "action": {"type": "DiscreteMetaAction"},
            "duration": 40,
            "collision_reward": -5,
            "right_lane_reward": 0.2,
            "high_speed_reward": 0.4,
            "reward_speed_range": [20, 30],
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "vehicles_count": 15,
            "lanes_count": 4,
        }
        unwrapped_env.configure(highway_config)

    elif env_name == "merge-v0":
        env = gym.make("merge-v0", render_mode=None)
        unwrapped_env = env.unwrapped

        # Curriculum-compatible merge config
        merge_config = {
            "observation": {
                "type": "LidarObservation",
                "cells": 32,  # Match curriculum model's 32 cells
                "row_anchor": [0.5, 0.5],
                "features": ["presence", "distance", "speed"],
                "features_range": {"distance": [0, 50], "speed": [-30, 30]}
            },
            "action": {"type": "DiscreteMetaAction"},
            "duration": 40,
            "collision_reward": -5,
            "high_speed_reward": 0.4,
            "reward_speed_range": [20, 30],
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "vehicles_count": 15,
        }
        unwrapped_env.configure(merge_config)

    elif env_name == "intersection-v0":
        env = gym.make("intersection-v0", render_mode=None)
        unwrapped_env = env.unwrapped

        # Curriculum-compatible intersection config (same as training)
        intersection_config = {
            "observation": {
                "type": "LidarObservation",
                "cells": 32,  # Match foundation model's 32 cells
                "row_anchor": [0.5, 0.5],
                "features": ["presence", "distance", "speed"],
                "features_range": {"distance": [0, 50], "speed": [-30, 30]}
            },
            "action": {"type": "DiscreteMetaAction"},
            "duration": 40,
            "collision_reward": -5,
            "high_speed_reward": 0.4,
            "reward_speed_range": [20, 30],
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "vehicles_count": 15,
            "initial_vehicle_count": 10,
        }
        unwrapped_env.configure(intersection_config)

    env.reset()
    return env

def evaluate_model(model_path, env_name, episodes=30):
    """Evaluate model on specified environment."""
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
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            if info.get('crashed', False):
                crashed = True

            # Success criteria: positive reward and no crash
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
        # Highway: expect full episode completion with positive reward
        performance_score = min(1.0, max(0.0, (mean_reward + 20) / 40))  # Scale from -20 to +20
    elif env_name == "merge-v0":
        # Merge: expect merge completion within reasonable time
        performance_score = min(1.0, max(0.0, (mean_reward + 10) / 25))  # Scale from -10 to +15
    else:  # intersection
        # Intersection: very challenging, scale appropriately
        performance_score = min(1.0, max(0.0, (mean_reward + 30) / 30))  # Scale from -30 to 0

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
    """Evaluate curriculum model on all three environments."""
    print("CURRICULUM MODEL EVALUATION")
    print("=" * 70)
    print("Model: Highway -> Merge -> Intersection Curriculum Transfer")
    print("Testing on: highway-v0, merge-v0, AND intersection-v0")
    print("The ultimate test: does curriculum learning enable cross-scenario generalization?")
    print("=" * 70)

    # Model path
    model_path = "outputs/models/curriculum/highway_merge_intersection_intersection_lidar/highway_merge_intersection_intersection_lidar_final.zip"

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please run curriculum training first: python transfer_merge_to_intersection.py")
        return

    print(f"Loading model from: {model_path}")

    # Initialize wandb
    run = wandb.init(
        project="highway-foundation-v2",
        name=f"eval_curriculum_three_{int(datetime.now().timestamp())}",
        config={
            "evaluation_type": "curriculum_three_evaluation",
            "model_path": model_path,
            "environments": ["highway-v0", "merge-v0", "intersection-v0"],
            "episodes_per_env": 30,
            "curriculum_phase": "highway_merge_intersection"
        }
    )

    try:
        environments = ["highway-v0", "merge-v0", "intersection-v0"]
        results = {}

        for env_name in environments:
            print(f"\nEvaluating on {env_name}...")
            print("-" * 50)

            env_results = evaluate_model(model_path, env_name, episodes=30)
            if "error" not in env_results:
                results[env_name] = env_results
            else:
                print(f"Skipping {env_name} due to error")

        print("\n" + "=" * 70)
        print("CURRICULUM MODEL PERFORMANCE SUMMARY")
        print("=" * 70)

        if results:
            for env, result in results.items():
                env_short = env.replace('-v0', '')
                score = result['performance_score']
                print(".3f")

            total_score = sum(result['performance_score'] for result in results.values()) / len(results)
            print(".3f")

            # Analysis
            highway_score = results.get("highway-v0", {}).get("performance_score", 0)
            merge_score = results.get("merge-v0", {}).get("performance_score", 0)
            intersection_score = results.get("intersection-v0", {}).get("performance_score", 0)

            print("\nCURRICULUM TRANSFER ANALYSIS:")
            print("----------------------------------------")
            print(".3f")
            print(".3f")
            print(".3f")

            if highway_score > 0.5 and merge_score > 0.5 and intersection_score > 0:
                print("VERDICT: Curriculum learning shows promising generalization!")
            elif highway_score > 0.7 and merge_score > 0.7:
                print("VERDICT: Excellent on highway/merge, intersection remains challenging")
            else:
                print("VERDICT: Limited generalization - may need curriculum improvements")

        # Log to wandb
        wandb.log({
            "evaluation_completed": True,
            "highway_performance": results.get("highway-v0", {}).get("performance_score", 0),
            "merge_performance": results.get("merge-v0", {}).get("performance_score", 0),
            "intersection_performance": results.get("intersection-v0", {}).get("performance_score", 0),
            "overall_performance": total_score if 'total_score' in locals() else 0
        })

    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        wandb.finish()

if __name__ == "__main__":
    main()
