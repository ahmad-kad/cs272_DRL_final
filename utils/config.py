import gymnasium as gym
import numpy as np
from config import ObservationConfigFactory, BaseEnvironmentConfig, RewardConfig, ScenarioConfigFactory
from constants import TrainingConstants


def get_curriculum_config(env_name: str, difficulty: str = "easy", modality: str = "lidar") -> dict:
    """
    Returns environment configuration based on difficulty and modality.

    Difficulties:
    - easy: Low traffic, lenient rewards (Learning dynamics)
    - medium: Moderate traffic, standard rewards (Learning interaction)
    - hard: Dense traffic, strict penalties (Mastery & Safety)

    Args:
        env_name: Name of the environment (e.g., "highway-v0", "merge-v0")
        difficulty: Difficulty level ("easy", "medium", "hard")
        modality: Observation modality ("lidar", "grayscale", "both")

    Returns:
        Complete environment configuration dictionary
    """
    if difficulty not in TrainingConstants.DIFFICULTIES:
        raise ValueError(f"Unknown difficulty: {difficulty}. Must be one of {TrainingConstants.DIFFICULTIES}")

    # Build configuration from components
    config = BaseEnvironmentConfig.get_config(
        duration_override=BaseEnvironmentConfig.get_duration_for_difficulty(difficulty)
    )

    # Add reward configuration
    config.update(RewardConfig.get_by_difficulty(difficulty))

    # Add scenario-specific configuration
    config.update(ScenarioConfigFactory.create_config(env_name, difficulty))

    # Add observation configuration
    config["observation"] = ObservationConfigFactory.create_config(modality, use_curriculum_features=True)

    return config

