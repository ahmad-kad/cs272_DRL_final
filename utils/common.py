"""
Common utilities for the autonomous driving RL system.

This module contains shared functions and utilities to reduce code duplication
and improve maintainability across the codebase.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np
from stable_baselines3 import PPO


def load_model_safely(model_path: str, verbose: bool = True) -> Optional[PPO]:
    """
    Load a PPO model with safe error handling and standardized logging.

    Args:
        model_path: Path to the model file (.zip)
        verbose: Whether to print status messages

    Returns:
        Loaded PPO model, or None if loading failed

    Example:
        model = load_model_safely("results/models/highway_expert.zip")
        if model is None:
            print("Failed to load model")
    """
    try:
        if verbose:
            print(f"[LOAD] Loading model from: {model_path}")
        model = PPO.load(model_path)
        if verbose:
            print(f"[OK] Model loaded successfully")
        return model
    except FileNotFoundError:
        if verbose:
            print(f"[ERROR] Model file not found: {model_path}")
    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to load model: {e}")
    return None


def save_model_safely(model: PPO, model_path: str, verbose: bool = True) -> bool:
    """
    Save a PPO model with safe error handling and standardized logging.

    Args:
        model: PPO model to save
        model_path: Path where to save the model (.zip)
        verbose: Whether to print status messages

    Returns:
        True if saving succeeded, False otherwise

    Example:
        success = save_model_safely(model, "results/models/trained_model.zip")
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if verbose:
            print(f"[SAVE] Saving model to: {model_path}")
        model.save(model_path)
        if verbose:
            print(f"[OK] Model saved successfully")
        return True
    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to save model: {e}")
        return False


def evaluate_episode(model: PPO, env, max_steps: int = 1000,
                    deterministic: bool = True) -> Dict[str, Any]:
    """
    Evaluate a model on a single episode with standardized metrics.

    Args:
        model: PPO model to evaluate
        env: Environment to evaluate on
        max_steps: Maximum steps per episode
        deterministic: Whether to use deterministic actions

    Returns:
        Dictionary containing episode metrics:
        - reward: Total episode reward
        - steps: Number of steps taken
        - crashed: Whether the vehicle crashed
        - completed: Whether the episode was completed successfully

    Example:
        metrics = evaluate_episode(model, env, max_steps=500)
        print(f"Episode reward: {metrics['reward']:.2f}")
    """
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    crashed = False
    terminated = False
    truncated = False

    while not (terminated or truncated) and steps < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)

        # Handle both old and new gym API
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
        else:
            next_obs, reward, terminated, truncated = step_result[:4]
            info = step_result[4] if len(step_result) > 4 else {}

        total_reward += reward
        obs = next_obs
        steps += 1

        # Check for crash (environment-specific)
        if hasattr(env, 'vehicle') and env.vehicle.crashed:
            crashed = True
            break

    # Determine completion based on scenario
    completed = False
    if hasattr(env, 'current_scenario'):
        scenario = env.current_scenario
        if scenario == "highway" and steps >= 150 and not crashed:
            completed = True
        elif scenario == "merge" and steps >= 100 and not crashed:
            completed = True
        elif scenario == "intersection" and steps >= 80 and not crashed:
            completed = True

    return {
        "reward": total_reward,
        "steps": steps,
        "crashed": crashed,
        "completed": completed,
        "success": completed and not crashed
    }


def format_model_loading_message(model_path: str, modality: str = "") -> str:
    """
    Format a standardized model loading message.

    Args:
        model_path: Path to the model file
        modality: Optional modality description (lidar, grayscale, etc.)

    Returns:
        Formatted loading message string

    Example:
        msg = format_model_loading_message("models/lidar.zip", "lidar")
        print(msg)  # "Loading lidar model from: models/lidar.zip"
    """
    modality_str = f"{modality} " if modality else ""
    return f"Loading {modality_str}model from: {model_path}"


def format_model_saving_message(model_path: str) -> str:
    """
    Format a standardized model saving message.

    Args:
        model_path: Path where the model will be saved

    Returns:
        Formatted saving message string

    Example:
        msg = format_model_saving_message("results/models/trained.zip")
        print(msg)  # "Saving model to: results/models/trained.zip"
    """
    return f"Saving model to: {model_path}"


def calculate_performance_metrics(episodes_data: list) -> Dict[str, float]:
    """
    Calculate standardized performance metrics from episode data.

    Args:
        episodes_data: List of episode result dictionaries from evaluate_episode()

    Returns:
        Dictionary containing aggregated metrics:
        - avg_reward: Average episode reward
        - success_rate: Percentage of successful episodes
        - crash_rate: Percentage of crashed episodes
        - completion_rate: Percentage of completed episodes
        - avg_steps: Average episode length

    Example:
        episodes = [evaluate_episode(model, env) for _ in range(10)]
        metrics = calculate_performance_metrics(episodes)
        print(f"Success rate: {metrics['success_rate']:.1%}")
    """
    if not episodes_data:
        return {
            "avg_reward": 0.0,
            "success_rate": 0.0,
            "crash_rate": 0.0,
            "completion_rate": 0.0,
            "avg_steps": 0.0
        }

    rewards = [ep["reward"] for ep in episodes_data]
    successes = [ep["success"] for ep in episodes_data]
    crashes = [ep["crashed"] for ep in episodes_data]
    completions = [ep["completed"] for ep in episodes_data]
    steps = [ep["steps"] for ep in episodes_data]

    return {
        "avg_reward": np.mean(rewards),
        "success_rate": np.mean(successes),
        "crash_rate": np.mean(crashes),
        "completion_rate": np.mean(completions),
        "avg_steps": np.mean(steps)
    }


def ensure_directory_exists(dir_path: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        dir_path: Path to the directory

    Returns:
        True if directory exists or was created successfully

    Example:
        if ensure_directory_exists("results/models"):
            print("Directory ready")
    """
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def validate_environment_config(config: Dict[str, Any]) -> bool:
    """
    Validate that an environment configuration contains required keys.

    Args:
        config: Environment configuration dictionary

    Returns:
        True if configuration is valid

    Example:
        config = {"scenario": "highway", "modality": "lidar"}
        if validate_environment_config(config):
            env = create_environment(config)
    """
    required_keys = ["scenario", "modality"]
    return all(key in config for key in required_keys)


def safe_file_operation(operation_func, *args, **kwargs):
    """
    Execute a file operation with safe error handling.

    Args:
        operation_func: Function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Tuple of (success: bool, result: any, error: str)

    Example:
        success, result, error = safe_file_operation(
            lambda: open("file.txt", "r").read()
        )
    """
    try:
        result = operation_func(*args, **kwargs)
        return True, result, ""
    except Exception as e:
        return False, None, str(e)
