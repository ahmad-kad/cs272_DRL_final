#!/usr/bin/env python3
"""
Evaluate Fine-tuned Intersection Model on All Three Environments
Tests the intersection fine-tuned model across highway, merge, and intersection scenarios.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
import numpy as np
import wandb
from datetime import datetime

def create_eval_env(env_name):
    """Create evaluation environment matching the fine-tuned model's expectations."""

    if env_name == "highway-v0":
        env = gym.make("highway-v0", render_mode=None)
        unwrapped_env = env.unwrapped

        # Match highway foundation config
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

        # Match merge config
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

        # Match fine-tuned intersection config
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
            "vehicles_count": 15,
            "initial_vehicle_count": 10,
        }
        unwrapped_env.configure(config)

    env.reset()
    return env

def evaluate_model(model_path, env_name, episodes=30):
    """Evaluate model on specified environment."""

    print(f"Loading fine-tuned model: {model_path}")
    env = create_eval_env(env_name)

    try:
        model = PPO.load(model_path, env=env, device='cpu')
    except Exception as e:
        print(f"Failed to load model: {e}")
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

def find_finetuned_model():
    """Find the most recent fine-tuned intersection model."""

    finetune_dir = Path("outputs/models")

    # Check for safety fine-tune models first (highest priority)
    safety_models = list(finetune_dir.glob("safety_finetune_*/safety_finetune_final.zip"))
    if safety_models:
        most_recent = max(safety_models, key=lambda x: x.stat().st_mtime)
        print(f"Found safety fine-tuned model: {most_recent}")
        return str(most_recent)

    # Check for contrastive fine-tune models
    contrastive_models = list(finetune_dir.glob("contrastive_finetune_*/contrastive_finetune_final.zip"))
    if contrastive_models:
        most_recent = max(contrastive_models, key=lambda x: x.stat().st_mtime)
        print(f"Found contrastive fine-tuned model: {most_recent}")
        return str(most_recent)

    # Check for EWC fine-tune models
    ewc_models = list(finetune_dir.glob("ewc_finetune_*/ewc_finetune_final.zip"))
    if ewc_models:
        most_recent = max(ewc_models, key=lambda x: x.stat().st_mtime)
        print(f"Found EWC fine-tuned model: {most_recent}")
        return str(most_recent)

    # Check for aggressive fine-tune models
    aggressive_models = list(finetune_dir.glob("aggressive_finetune_*/aggressive_finetune_final.zip"))
    if aggressive_models:
        most_recent = max(aggressive_models, key=lambda x: x.stat().st_mtime)
        print(f"Found aggressive fine-tuned model: {most_recent}")
        return str(most_recent)

    # Check for regular fine-tune models
    finetune_models = list(finetune_dir.glob("finetune_intersection_*/finetune_intersection_final.zip"))
    if finetune_models:
        most_recent = max(finetune_models, key=lambda x: x.stat().st_mtime)
        print(f"Found fine-tuned model: {most_recent}")
        return str(most_recent)

    # Fallback to advanced curriculum models
    advanced_dir = Path("outputs/models/curriculum_advanced")
    if advanced_dir.exists():
        for phase in ["expert_intersection", "hard_intersection", "medium_intersection"]:
            phase_dir = advanced_dir / phase
            if phase_dir.exists():
                model_files = list(phase_dir.glob("*.zip"))
                if model_files:
                    most_recent = max(model_files, key=lambda x: x.stat().st_mtime)
                    print(f"Using advanced curriculum model: {most_recent}")
                    return str(most_recent)

    return None

def main():
    """Evaluate fine-tuned model on all three environments."""

    print("SAFETY FINE-TUNED INTERSECTION MODEL - CROSS-ENVIRONMENT EVALUATION")
    print("=" * 80)
    print("Model: Safety Fine-tuned Intersection Model")
    print("Training: Multi-phase safety curriculum + harsh crash penalties + safety-first rewards")
    print("Testing on: highway-v0, merge-v0, AND intersection-v0")
    print("Goal: Assess crash reduction and safety improvements")
    print("=" * 80)

    # Find the fine-tuned model
    model_path = find_finetuned_model()

    if not model_path:
        print("❌ No fine-tuned or advanced curriculum models found!")
        print("Please run fine-tuning first: python finetune_intersection.py")
        return

    print(f"[MODEL] {model_path}")

    # Initialize wandb
    run = wandb.init(
        project="highway-foundation-v2",
        name=f"eval_finetuned_cross_env_{int(datetime.now().timestamp())}",
        config={
            "evaluation_type": "finetuned_cross_environment",
            "model_path": model_path,
            "environments": ["highway-v0", "merge-v0", "intersection-v0"],
            "episodes_per_env": 30,
            "fine_tuned": True,
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
        print("FINETUNED MODEL CROSS-ENVIRONMENT PERFORMANCE SUMMARY")
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
            print("\nSAFETY FINE-TUNING ANALYSIS:")
            print("-" * 40)
            print("Safety Fine-tuning Effects:")
            print("  + Multi-phase safety curriculum with harsh crash penalties")
            print("  + Safety-first reward structure prioritizes crash avoidance")
            print("  + Progressive safety training from ultra-safe to balanced")
            print("  + Reduced progress incentives to prevent risky behavior")
            print("  + Safety bonuses for maintaining safe distances")
            print("  + Multi-objective optimization balancing safety vs. completion")

            highway_score = results.get("highway-v0", {}).get("performance_score", 0)
            merge_score = results.get("merge-v0", {}).get("performance_score", 0)
            intersection_score = results.get("intersection-v0", {}).get("performance_score", 0)

            # Compare to previous models
            print("\nPERFORMANCE COMPARISON:")
            print("-" * 40)
            print("Original Multi-Env (0.552): Simultaneous training")
            print(".3f")
            print("Advanced Curriculum (0.894): Progressive learning")
            print(".3f")
            print("Fine-tuned Intersection: Specialized training")
            print(".3f")

            if intersection_score > 0.8:
                specialization = "EXCELLENT: Significant intersection improvement achieved!"
            elif intersection_score > 0.7:
                specialization = "GOOD: Solid intersection performance gains"
            elif intersection_score > merge_score and intersection_score > highway_score:
                specialization = "SPECIALIZED: Best at intersection (as intended)"
            else:
                specialization = "MIXED: Intersection focus may have impacted generalization"

            print(f"\n[SPECIALIZATION ASSESSMENT] {specialization}")

            # Fine-tuning impact analysis
            print("\n[FINE-TUNING IMPACT]")
            intersection_improvement = intersection_score - 0.682  # vs advanced curriculum
            if intersection_improvement > 0:
                print(".3f")
            elif intersection_improvement > -0.1:
                print(".3f")
            else:
                print(".3f")
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
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()
