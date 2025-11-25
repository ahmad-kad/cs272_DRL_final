import gymnasium as gym
import numpy as np

def get_curriculum_config(env_name, difficulty="easy", modality="lidar"):
    """
    Returns environment configuration based on difficulty and modality.
    
    Difficulties:
    - easy: Low traffic, lenient rewards (Learning dynamics)
    - medium: Moderate traffic, standard rewards (Learning interaction)
    - hard: Dense traffic, strict penalties (Mastery & Safety)
    """
    
    # 1. Observation Configuration
    if modality == "lidar":
        obs_config = {
            "type": "LidarObservation",
            "cells": 32,
            "row_anchor": [0.5, 0.5],
            "features": ["presence", "distance", "speed"],
            "features_range": {"distance": [0, 50], "speed": [-30, 30]}
        }
    elif modality == "grayscale":
        obs_config = {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],
            "scaling": 1.75,
        }
    else:
        raise ValueError(f"Unknown modality: {modality}")

    # 2. Difficulty Parameters
    if difficulty == "easy":
        density = 10
        duration = 40
        collision_reward = -2.0
        high_speed_reward = 0.2
        reward_speed_range = [10, 30]
    elif difficulty == "medium":
        density = 20
        duration = 40
        collision_reward = -5.0
        high_speed_reward = 0.5
        reward_speed_range = [20, 30]
    elif difficulty == "hard":
        density = 30
        duration = 60
        collision_reward = -10.0
        high_speed_reward = 0.8
        reward_speed_range = [25, 35]
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    # 3. Base Configuration
    config = {
        "observation": obs_config,
        "action": {"type": "DiscreteMetaAction"},
        "duration": duration,
        "collision_reward": collision_reward,
        "high_speed_reward": high_speed_reward,
        "reward_speed_range": reward_speed_range,
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "vehicles_count": density,
        "lanes_count": 4,
        "screen_width": 600,
        "screen_height": 150,
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
    }

    # 4. Scenario Specifics
    if "merge" in env_name:
        config["vehicles_count"] = max(5, density // 2)
        config["merging_vehicle_probability"] = 0.3 if difficulty == "easy" else 0.6
    elif "intersection" in env_name:
        config["vehicles_count"] = max(5, density // 3)
        config["spawn_probability"] = 0.2 if difficulty == "easy" else 0.5
        config["arrived_reward"] = 6.0 # Bonus for intersection completion
        
    return config

