"""
Base environment configurations for autonomous driving RL environments.

This module provides the foundational environment configuration.
"""

from typing import Dict, Any
from constants import EnvironmentConstants, TrainingConstants


class BaseEnvironmentConfig:
    """Base environment configuration settings."""

    @staticmethod
    def get_config(duration_override: int = None) -> Dict[str, Any]:
        """Get base environment configuration."""
        config = {
            "action": {
                "type": "DiscreteMetaAction",  # More stable than Continuous for merging
            },
            "simulation_frequency": EnvironmentConstants.SIMULATION_FREQUENCY,
            "policy_frequency": EnvironmentConstants.POLICY_FREQUENCY,
            "duration": duration_override or EnvironmentConstants.DEFAULT_DURATION,
            "lanes_count": EnvironmentConstants.DEFAULT_LANE_COUNT,
            "normalize_reward": True,
            "offroad_terminal": False,  # allow recovery at first
            "screen_width": 1000,
            "screen_height": 900,
            "centering_position": [0.4, 0.35],
            "scaling": 6,
        }
        return config

    @staticmethod
    def get_duration_for_difficulty(difficulty: str) -> int:
        """Get episode duration based on difficulty level."""
        return TrainingConstants.EPISODE_DURATION.get(
            difficulty, EnvironmentConstants.DEFAULT_DURATION
        )
