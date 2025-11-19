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
    
    for exp_id, env_name, agent_type, gym_id in experiments:
        print(f"\n[ID {exp_id}] {env_name} - {agent_type.upper()}")
        
        # 1. Plot Learning Curve
        log_dir = os.path.join(PATHS.get('logs', 'highway_distillation/outputs/logs'), "monitor")
        plot_learning_curve(agent_type, env_name, exp_id, log_dir)
        
        # 2. Performance Test
        model_path = models[agent_type]
        if not os.path.exists(model_path + ".zip"):
            print(f"[SKIP] Model not found: {model_path}")
            continue
            
        try:
            env = create_eval_env(gym_id, agent_type)
            model = PPO.load(model_path, env=env)
            
            rewards = evaluate_performance(model, env, num_episodes=100)  # 100 for speed, 500 for final
            
            perf_id = exp_id + 1 if gym_id != "custom" else exp_id
            plot_violin(rewards, agent_type, env_name, perf_id)
            
            env.close()
        except Exception as e:
            print(f"[ERROR] {env_name}: {e}")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Plots saved to: {PATHS.get('plots', 'plots')}")

if __name__ == "__main__":
    main()
