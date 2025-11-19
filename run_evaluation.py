#!/usr/bin/env python3
"""
Evaluation and Plotting for Highway Distillation Project

Generates all required plots for the project submission:
- Learning Curves
- Performance Violin Plots
For Highway, Merge, Intersection, and Custom environments.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import gymnasium as gym
import highway_env
import wandb
import time

# Import custom environment
try:
    from highway_distillation.environments.urban_junction_env import UrbanJunctionEnv
    from highway_distillation.config import PATHS
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from highway_distillation.environments.urban_junction_env import UrbanJunctionEnv
    from highway_distillation.config import PATHS

# Ensure directories exist
os.makedirs(PATHS.get('plots', 'highway_distillation/outputs/plots'), exist_ok=True)
os.makedirs(PATHS.get('data', 'highway_distillation/outputs/data'), exist_ok=True)

def load_monitor_data(log_dir):
    """Load learning curve data from monitor logs."""
    all_dfs = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".monitor.csv"):
                try:
                    df = pd.read_csv(os.path.join(root, file), skiprows=1)
                    all_dfs.append(df)
                except:
                    pass
    
    if not all_dfs:
        return None
        
    combined_df = pd.concat(all_dfs)
    combined_df['r_smooth'] = combined_df['r'].rolling(window=100).mean()
    return combined_df

def plot_learning_curve(agent_type, env_name, exp_id, log_dir):
    """Generate Learning Curve Plot."""
    data = load_monitor_data(log_dir)
    
    plt.figure(figsize=(10, 6))
    if data is not None and len(data) > 0:
        plt.plot(data['l'].cumsum(), data['r_smooth'])
        plt.title(f"ID {exp_id}: Learning Curve - {env_name} ({agent_type.capitalize()})")
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Reward (100 ep moving avg)")
    else:
        plt.text(0.5, 0.5, "No Training Data (Run training first)", ha='center')
        plt.title(f"ID {exp_id}: Learning Curve - {env_name} ({agent_type.capitalize()})")
    
    save_path = os.path.join(PATHS.get('plots', 'plots'), f"ID{exp_id}_{env_name}_{agent_type}_learning.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {save_path}")

def evaluate_performance(model, env, num_episodes=500):
    """Run evaluation episodes."""
    rewards = []
    obs = env.reset()
    
    print(f"Evaluating {num_episodes} episodes...")
    for ep in range(num_episodes):
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
        rewards.append(episode_reward)
        obs = env.reset()
        if (ep + 1) % 50 == 0:
            print(f"  ... {ep+1}/{num_episodes} complete")
        
    return rewards

def plot_violin(rewards, agent_type, env_name, exp_id):
    """Generate Violin Plot."""
    plt.figure(figsize=(10, 6))
    sns.violinplot(y=rewards)
    plt.title(f"ID {exp_id}: Performance Test - {env_name} ({agent_type.capitalize()})")
    plt.ylabel("Episode Return")
    plt.xlabel(f"Distribution (n={len(rewards)})")
    
    # Add statistics
    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    plt.text(0.05, 0.95, f'Mean: {mean_r:.2f}\nStd: {std_r:.2f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    save_path = os.path.join(PATHS.get('plots', 'plots'), f"ID{exp_id}_{env_name}_{agent_type}_violin.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {save_path}")

def create_eval_env(env_id, agent_type):
    """Create evaluation environment."""
    if env_id == "custom":
        config = UrbanJunctionEnv.default_config()
        config["use_lidar_only"] = (agent_type == 'lidar')
        config["use_grayscale_only"] = (agent_type == 'grayscale')
        env = UrbanJunctionEnv(config=config, 
                               use_lidar_only=(agent_type == 'lidar'),
                               use_grayscale_only=(agent_type == 'grayscale'))
    else:
        env = gym.make(env_id)
        config = env.unwrapped.config
        if agent_type == 'lidar':
            config["observation"] = {
                "type": "LidarObservation",
                "cells": 32,
                "normalize": True
            }
        elif agent_type == 'grayscale':
            config["observation"] = {
                "type": "GrayscaleObservation",
                "observation_shape": (64, 64),
                "stack_size": 4
            }
        env.configure(config)
        
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
    return env

def main():
    # Initialize WandB
    wandb_config = {
        "evaluation_type": "performance_testing",
        "experiments": [
            "ID1: Highway Lidar", "ID3: Highway Grayscale",
            "ID5: Merge Lidar", "ID7: Merge Grayscale",
            "ID9: Intersection Lidar", "ID11: Intersection Grayscale",
            "ID13: Custom Lidar", "ID14: Custom Grayscale"
        ],
        "num_episodes": 100,  # Per evaluation
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S")
    }

    wandb.init(
        project="highway-distillation",
        name=f"evaluation_run_{int(time.time())}",
        config=wandb_config,
        notes="Comprehensive performance evaluation of trained generalist agents"
    )

    # Experiments matching project requirements
    experiments = [
        (1, "Highway", "lidar", "highway-v0"),
        (3, "Highway", "grayscale", "highway-v0"),
        (5, "Merge", "lidar", "merge-v0"),
        (7, "Merge", "grayscale", "merge-v0"),
        (9, "Intersection", "lidar", "intersection-v0"),
        (11, "Intersection", "grayscale", "intersection-v0"),
        (13, "Custom", "lidar", "custom"),
        (14, "Custom", "grayscale", "custom"),
    ]

    models = {
        "lidar": os.path.join(PATHS.get('models', 'highway_distillation/outputs/models'),
                             "lidar_generalist", "final_model"),
        "grayscale": os.path.join(PATHS.get('models', 'highway_distillation/outputs/models'),
                                 "gray_generalist", "final_model")
    }

    print("\n" + "="*70)
    print("EVALUATION AND PLOTTING")
    print("="*70)
    print(f"WandB Project: highway-distillation")
    print("="*70)

    evaluation_start_time = time.time()
    evaluation_results = []
    
    for exp_id, env_name, agent_type, gym_id in experiments:
        print(f"\n[ID {exp_id}] {env_name} - {agent_type.upper()}")

        experiment_start_time = time.time()

        # 1. Plot Learning Curve
        log_dir = os.path.join(PATHS.get('logs', 'highway_distillation/outputs/logs'), "monitor")
        plot_learning_curve(agent_type, env_name, exp_id, log_dir)

        # 2. Performance Test
        model_path = models[agent_type]
        if not os.path.exists(model_path + ".zip"):
            print(f"[SKIP] Model not found: {model_path}")

            # Log skipped experiment
            wandb.log({
                f"exp_{exp_id}_skipped": True,
                f"exp_{exp_id}_reason": "model_not_found",
                f"exp_{exp_id}_timestamp": time.time()
            })
            continue

        try:
            env = create_eval_env(gym_id, agent_type)
            model = PPO.load(model_path, env=env)

            rewards = evaluate_performance(model, env, num_episodes=100)  # 100 for speed, 500 for final

            # Calculate statistics
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            median_reward = np.median(rewards)
            min_reward = np.min(rewards)
            max_reward = np.max(rewards)
            success_rate = np.mean([r > 0 for r in rewards])  # Basic success metric

            perf_id = exp_id + 1 if gym_id != "custom" else exp_id
            plot_violin(rewards, agent_type, env_name, perf_id)

            env.close()

            experiment_time = time.time() - experiment_start_time

            # Log results to WandB
            experiment_metrics = {
                f"exp_{exp_id}_completed": True,
                f"exp_{exp_id}_environment": env_name,
                f"exp_{exp_id}_agent_type": agent_type,
                f"exp_{exp_id}_gym_id": gym_id,
                f"exp_{exp_id}_mean_reward": mean_reward,
                f"exp_{exp_id}_std_reward": std_reward,
                f"exp_{exp_id}_median_reward": median_reward,
                f"exp_{exp_id}_min_reward": min_reward,
                f"exp_{exp_id}_max_reward": max_reward,
                f"exp_{exp_id}_success_rate": success_rate,
                f"exp_{exp_id}_evaluation_time": experiment_time,
                f"exp_{exp_id}_timestamp": time.time()
            }

            # Log the full reward distribution as a histogram
            wandb.log({
                f"exp_{exp_id}_reward_distribution": wandb.Histogram(rewards),
                **experiment_metrics
            })

            evaluation_results.append({
                "exp_id": exp_id,
                "env_name": env_name,
                "agent_type": agent_type,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "success_rate": success_rate,
                **experiment_metrics
            })

            print(f"[WANDB] Logged results for ID {exp_id}")

        except Exception as e:
            experiment_time = time.time() - experiment_start_time
            print(f"[ERROR] {env_name}: {e}")

            # Log error to WandB
            wandb.log({
                f"exp_{exp_id}_failed": True,
                f"exp_{exp_id}_error": str(e),
                f"exp_{exp_id}_evaluation_time": experiment_time,
                f"exp_{exp_id}_timestamp": time.time()
            })

    total_evaluation_time = time.time() - evaluation_start_time

    # Calculate summary statistics
    if evaluation_results:
        all_mean_rewards = [r["mean_reward"] for r in evaluation_results]
        all_success_rates = [r["success_rate"] for r in evaluation_results]

        summary_metrics = {
            "evaluation_completed": True,
            "total_evaluation_time_minutes": total_evaluation_time / 60,
            "experiments_completed": len(evaluation_results),
            "experiments_total": len(experiments),

            # Overall performance metrics
            "overall_mean_reward": np.mean(all_mean_rewards),
            "overall_std_reward": np.std(all_mean_rewards),
            "overall_mean_success_rate": np.mean(all_success_rates),
            "best_experiment_mean_reward": np.max(all_mean_rewards),
            "worst_experiment_mean_reward": np.min(all_mean_rewards),

            # Per-agent type performance
            "lidar_experiments": len([r for r in evaluation_results if r["agent_type"] == "lidar"]),
            "grayscale_experiments": len([r for r in evaluation_results if r["agent_type"] == "grayscale"]),

            "final_timestamp": time.time()
        }

        if summary_metrics["lidar_experiments"] > 0:
            lidar_rewards = [r["mean_reward"] for r in evaluation_results if r["agent_type"] == "lidar"]
            summary_metrics.update({
                "lidar_mean_reward": np.mean(lidar_rewards),
                "lidar_std_reward": np.std(lidar_rewards),
                "lidar_best_reward": np.max(lidar_rewards)
            })

        if summary_metrics["grayscale_experiments"] > 0:
            gray_rewards = [r["mean_reward"] for r in evaluation_results if r["agent_type"] == "grayscale"]
            summary_metrics.update({
                "grayscale_mean_reward": np.mean(gray_rewards),
                "grayscale_std_reward": np.std(gray_rewards),
                "grayscale_best_reward": np.max(gray_rewards)
            })

        wandb.log(summary_metrics)

        # Log summary table as a WandB table
        if evaluation_results:
            import pandas as pd
            df = pd.DataFrame(evaluation_results)
            wandb.log({"evaluation_summary_table": wandb.Table(dataframe=df)})

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Plots saved to: {PATHS.get('plots', 'plots')}")
    print(f"WandB Dashboard: https://wandb.ai/{wandb.run.entity}/highway-distillation")
    print(f"Experiments completed: {len(evaluation_results)}/{len(experiments)}")
    print(".1f")

if __name__ == "__main__":
    main()
