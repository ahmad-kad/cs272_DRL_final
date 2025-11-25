#!/usr/bin/env python3
"""
Evaluate Fine-tuned Intersection Model
Tests the fine-tuned model specifically on intersection scenarios with detailed metrics.
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
import json

def create_intersection_env(seed=None):
    """Create intersection environment matching fine-tuning configuration."""

    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)

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
        "vehicles_count": 15,
        "initial_vehicle_count": 10,
    }

    unwrapped_env.configure(config)
    env.reset()
    return env

def evaluate_intersection_model(model_path, episodes=50):
    """Evaluate model specifically on intersection with detailed metrics."""

    print(f"Loading fine-tuned model: {model_path}")
    env = create_intersection_env()

    try:
        model = PPO.load(model_path, env=env, device='cpu')
    except Exception as e:
        print(f"Failed to load model: {e}")
        env.close()
        return {"error": str(e)}

    # Detailed tracking
    rewards = []
    lengths = []
    crashes = []
    successes = []
    completion_times = []
    near_misses = []
    traffic_density = []

    print(f"\nEvaluating on intersection scenarios for {episodes} episodes...")
    print("Each episode tests different traffic conditions for robustness.")

    for episode in range(episodes):
        # Randomize traffic conditions for each episode
        seed = episode * 42  # Deterministic but varied seeds
        env = create_intersection_env(seed)

        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        crashed = False
        success = False
        near_miss = False
        start_time = datetime.now()

        # Track traffic density
        initial_traffic = len([v for v in env.unwrapped.road.vehicles if v is not env.unwrapped.vehicle])

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)

            episode_reward += reward
            episode_length += 1

            if step_info.get('crashed', False):
                crashed = True
                # Check if it was a "near miss" (crashed late in episode)
                if episode_length > 10:  # Survived at least 10 steps
                    near_miss = True

            # Success criteria: positive reward and no collisions
            if terminated and episode_reward > 0 and not crashed:
                success = True
                end_time = datetime.now()
                completion_time = (end_time - start_time).total_seconds()
                completion_times.append(completion_time)

            if terminated or truncated or episode_length >= 50:
                break

        rewards.append(episode_reward)
        lengths.append(episode_length)
        crashes.append(1 if crashed else 0)
        successes.append(1 if success else 0)
        near_misses.append(1 if near_miss else 0)
        traffic_density.append(initial_traffic)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/50: Reward = {episode_reward:.2f}, "
                  f"Length = {episode_length}, Success = {success}, Crash = {crashed}")

    env.close()

    # Calculate comprehensive metrics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    success_rate = np.mean(successes) * 100
    crash_rate = np.mean(crashes) * 100
    near_miss_rate = np.mean(near_misses) * 100

    # Performance score
    performance_score = min(1.0, max(0.0, (mean_reward + 30) / 30))

    # Safety metrics
    safe_episodes = sum(1 for c in crashes if c == 0)
    safety_score = safe_episodes / len(crashes)

    # Efficiency metrics
    if completion_times:
        avg_completion_time = np.mean(completion_times)
        std_completion_time = np.std(completion_times)
    else:
        avg_completion_time = 0
        std_completion_time = 0

    # Traffic analysis
    avg_traffic_density = np.mean(traffic_density)

    print("
" + "="*60)
    print("INTERSECTION PERFORMANCE RESULTS")
    print("="*60)

    print("
🎯 OVERALL PERFORMANCE")
    print(".2f"    print(".1f"    print(".1f"    print(".3f"

    print("
🛡️ SAFETY METRICS")
    print(".1f"    print(".1f"    print(".1f"

    print("
⚡ EFFICIENCY METRICS")
    print(f"Average Traffic Density: {avg_traffic_density:.1f} vehicles")
    print(f"Average Episode Length: {np.mean(lengths):.1f} steps")
    if completion_times:
        print(".2f"        print(".2f"
    else:
        print("No successful completions for timing analysis")

    print("
📊 DISTRIBUTION ANALYSIS")
    print("Reward Distribution:")
    print(f"  Min: {np.min(rewards):.2f}")
    print(f"  25th percentile: {np.percentile(rewards, 25):.2f}")
    print(f"  Median: {np.median(rewards):.2f}")
    print(f"  75th percentile: {np.percentile(rewards, 75):.2f}")
    print(f"  Max: {np.max(rewards):.2f}")

    # Performance assessment
    if success_rate >= 85 and crash_rate <= 10:
        assessment = "OUTSTANDING: Exceptional intersection handling!"
        grade = "A+"
    elif success_rate >= 75 and crash_rate <= 15:
        assessment = "EXCELLENT: Very strong intersection performance"
        grade = "A"
    elif success_rate >= 65 and crash_rate <= 20:
        assessment = "GOOD: Solid intersection capabilities"
        grade = "B+"
    elif success_rate >= 50 and crash_rate <= 30:
        assessment = "FAIR: Adequate intersection performance"
        grade = "B"
    else:
        assessment = "NEEDS IMPROVEMENT: Further fine-tuning required"
        grade = "C"

    print(f"\n🎓 PERFORMANCE GRADE: {grade}")
    print(f"📋 ASSESSMENT: {assessment}")

    # Comparison to baseline expectations
    print("
📈 COMPARISON TO TARGETS")
    print("Target Success Rate: >80%"    print(".1f"    print("Target Crash Rate: <15%"    print(".1f"
    success_diff = success_rate - 80
    crash_diff = crash_rate - 15

    if success_diff >= 0 and crash_diff <= 0:
        print("✅ TARGETS ACHIEVED: Exceeds expectations!")
    elif success_diff >= -5 and crash_diff <= 5:
        print("⚪ CLOSE TO TARGET: Minor adjustments needed")
    else:
        print("🔄 TARGETS MISSED: Additional fine-tuning recommended")

    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'success_rate': success_rate,
        'crash_rate': crash_rate,
        'near_miss_rate': near_miss_rate,
        'performance_score': performance_score,
        'safety_score': safety_score,
        'avg_completion_time': avg_completion_time,
        'avg_traffic_density': avg_traffic_density,
        'grade': grade,
        'assessment': assessment,
        'reward_distribution': {
            'min': np.min(rewards),
            'q25': np.percentile(rewards, 25),
            'median': np.median(rewards),
            'q75': np.percentile(rewards, 75),
            'max': np.max(rewards)
        }
    }

def find_finetuned_model():
    """Find the most recent fine-tuned intersection model."""

    finetune_dir = Path("outputs/models")
    finetune_models = list(finetune_dir.glob("finetune_intersection_*/finetune_intersection_final.zip"))

    if finetune_models:
        most_recent = max(finetune_models, key=lambda x: x.stat().st_mtime)
        print(f"Found fine-tuned model: {most_recent}")
        return str(most_recent)

    # If no fine-tuned models, check for advanced curriculum models
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
    """Evaluate fine-tuned intersection model."""

    print("🎯 FINE-TUNED INTERSECTION MODEL EVALUATION")
    print("=" * 80)
    print("Testing intersection performance with enhanced rewards and domain randomization")
    print("=" * 80)

    # Find the model to evaluate
    model_path = find_finetuned_model()

    if not model_path:
        print("❌ No fine-tuned or advanced curriculum models found!")
        print("Please run fine-tuning first: python finetune_intersection.py")
        return

    print(f"📥 Model: {model_path}")

    # Initialize wandb
    run = wandb.init(
        project="highway-foundation-v2",
        name=f"eval_finetuned_intersection_{int(datetime.now().timestamp())}",
        config={
            "evaluation_type": "finetuned_intersection_evaluation",
            "model_path": model_path,
            "episodes": 50,
            "enhanced_rewards": True,
            "domain_randomization": True
        }
    )

    try:
        results = evaluate_intersection_model(model_path, episodes=50)

        if "error" not in results:
            # Log detailed results to wandb
            wandb.log({
                "evaluation_completed": True,
                "success_rate": results['success_rate'],
                "crash_rate": results['crash_rate'],
                "performance_score": results['performance_score'],
                "safety_score": results['safety_score'],
                "grade": results['grade'],
                "mean_reward": results['mean_reward'],
                "near_miss_rate": results['near_miss_rate']
            })

            print(f"\n🎉 Evaluation completed! Results logged to Weights & Biases.")
            print(f"Model: {os.path.basename(model_path)}")
            print(f"Grade: {results['grade']} - {results['assessment']}")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()

