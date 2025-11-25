"""
Evaluation Configurations for Autonomous Driving Environments

This module provides standardized default configurations for evaluating models
across different driving scenarios (highway, merge, intersection) and observation
modalities (lidar, grayscale). These configurations are optimized for model
benchmarking rather than training.
"""

from typing import Dict, Any
import numpy as np

# Standardized observation configurations for evaluation
EVALUATION_OBS_CONFIGS = {
    "lidar": {
        "type": "LidarObservation",
        "cells": 32,  # Match trained model configuration
        "maximum_range": 50,
        "normalize": True,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
    },
    "grayscale": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],  # Standard RGB to grayscale weights
        "scaling": 1.75,
    }
}

# Scenario-specific evaluation parameters
SCENARIO_EVALUATION_PARAMS = {
    "highway": {
        "description": "Standard highway cruising with moderate traffic",
        "duration": 60,  # Longer episodes for highway evaluation
        "vehicles_count": 15,  # Moderate traffic density
        "lanes_count": 4,
        "reward_speed_range": [20, 30],  # Optimal speed range
        "collision_reward": -1.0,
        "high_speed_reward": 0.4,
        "arrived_reward": 0.0,  # No specific arrival goal for highway
        "right_lane_reward": 0.1,  # Slight preference for right lane
        "normalize_reward": True,
        "offroad_terminal": True,
    },
    "merge": {
        "description": "Highway merge scenario with merging traffic",
        "duration": 40,  # Shorter episodes focused on merge completion
        "vehicles_count": 12,  # Moderate traffic with merging vehicles
        "lanes_count": 4,
        "reward_speed_range": [20, 30],
        "collision_reward": -1.0,
        "high_speed_reward": 0.3,
        "arrived_reward": 0.5,  # Reward for successful merge onto main road
        "right_lane_reward": 0.1,
        "normalize_reward": True,
        "offroad_terminal": True,
        "merging_vehicle_probability": 0.4,  # Probability of vehicles merging
    },
    "intersection": {
        "description": "Urban intersection with cross traffic and traffic lights",
        "duration": 30,  # Shorter episodes due to complexity
        "vehicles_count": 10,  # Lower traffic density but more complex interactions
        "lanes_count": 4,
        "reward_speed_range": [15, 25],  # Slower speeds in urban environment
        "collision_reward": -1.0,
        "high_speed_reward": 0.2,  # Lower speed reward in intersection
        "arrived_reward": 1.0,  # Significant reward for completing intersection
        "right_lane_reward": 0.05,  # Minimal lane preference in intersection
        "normalize_reward": True,
        "offroad_terminal": True,
        "spawn_probability": 0.3,  # Probability of new vehicles spawning
    }
}

# Common evaluation settings
EVALUATION_COMMON_CONFIG = {
    "action": {
        "type": "DiscreteMetaAction",  # Stable discrete actions for evaluation
    },
    "simulation_frequency": 15,
    "policy_frequency": 1,
    "screen_width": 600,
    "screen_height": 150,
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,  # Disable for evaluation performance
}

def get_evaluation_config(scenario: str, modality: str) -> Dict[str, Any]:
    """
    Get standardized evaluation configuration for a specific scenario and modality.

    Args:
        scenario: Driving scenario ("highway", "merge", "intersection")
        modality: Observation modality ("lidar", "grayscale")

    Returns:
        Complete environment configuration dictionary for evaluation

    Raises:
        ValueError: If scenario or modality is not supported
    """
    if scenario not in SCENARIO_EVALUATION_PARAMS:
        raise ValueError(f"Unsupported scenario '{scenario}'. Must be one of: {list(SCENARIO_EVALUATION_PARAMS.keys())}")

    if modality not in EVALUATION_OBS_CONFIGS:
        raise ValueError(f"Unsupported modality '{modality}'. Must be one of: {list(EVALUATION_OBS_CONFIGS.keys())}")

    # Build complete configuration
    config = EVALUATION_COMMON_CONFIG.copy()
    config["observation"] = EVALUATION_OBS_CONFIGS[modality].copy()
    config.update(SCENARIO_EVALUATION_PARAMS[scenario].copy())

    return config

def get_all_evaluation_configs() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Get all evaluation configurations for systematic model evaluation.

    Returns:
        Nested dictionary: scenario -> modality -> config
    """
    configs = {}
    for scenario in SCENARIO_EVALUATION_PARAMS.keys():
        configs[scenario] = {}
        for modality in EVALUATION_OBS_CONFIGS.keys():
            configs[scenario][modality] = get_evaluation_config(scenario, modality)
    return configs

# Pre-defined configurations for convenience
HIGHWAY_LIDAR_CONFIG = get_evaluation_config("highway", "lidar")
HIGHWAY_GRAYSCALE_CONFIG = get_evaluation_config("highway", "grayscale")
MERGE_LIDAR_CONFIG = get_evaluation_config("merge", "lidar")
MERGE_GRAYSCALE_CONFIG = get_evaluation_config("merge", "grayscale")
INTERSECTION_LIDAR_CONFIG = get_evaluation_config("intersection", "lidar")
INTERSECTION_GRAYSCALE_CONFIG = get_evaluation_config("intersection", "grayscale")

# Summary of evaluation configurations
EVALUATION_CONFIG_SUMMARY = {
    "scenarios": list(SCENARIO_EVALUATION_PARAMS.keys()),
    "modalities": list(EVALUATION_OBS_CONFIGS.keys()),
    "total_configurations": len(SCENARIO_EVALUATION_PARAMS) * len(EVALUATION_OBS_CONFIGS),
    "evaluation_episodes_per_config": 50,  # Standard evaluation episodes
    "description": "Standardized configurations for model evaluation across driving scenarios"
}
