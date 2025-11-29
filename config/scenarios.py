"""
Scenario configurations for autonomous driving RL environments.

This module handles scenario-specific configuration logic.
"""

from typing import Dict, Any
from constants import TrainingConstants, ScenarioConstants


class HighwayScenarioConfig:
    """Configuration for highway scenarios."""

    @staticmethod
    def get_config(difficulty: str) -> Dict[str, Any]:
        """Get highway scenario configuration."""
        return {
            "vehicles_count": TrainingConstants.TRAFFIC_DENSITY[difficulty]
        }


class MergeScenarioConfig:
    """Configuration for merge scenarios."""

    @staticmethod
    def get_config(difficulty: str) -> Dict[str, Any]:
        """Get merge scenario configuration."""
        return {
            "vehicles_count": max(5, TrainingConstants.TRAFFIC_DENSITY[difficulty] // 2),
            "merging_vehicle_probability": ScenarioConstants.MERGE_VEHICLE_PROBABILITY[difficulty]
        }


class IntersectionScenarioConfig:
    """Configuration for intersection scenarios."""

    @staticmethod
    def get_config(difficulty: str) -> Dict[str, Any]:
        """Get intersection scenario configuration."""
        return {
            "vehicles_count": max(5, TrainingConstants.TRAFFIC_DENSITY[difficulty] // 3),
            "spawn_probability": ScenarioConstants.INTERSECTION_SPAWN_PROBABILITY[difficulty],
            "arrived_reward": ScenarioConstants.INTERSECTION_COMPLETION_BONUS
        }


class ScenarioConfigFactory:
    """Factory for creating scenario-specific configurations."""

    @staticmethod
    def create_config(env_name: str, difficulty: str) -> Dict[str, Any]:
        """Create scenario configuration based on environment name."""
        scenario_map = {
            "merge": MergeScenarioConfig.get_config,
            "intersection": IntersectionScenarioConfig.get_config,
        }

        # Check for scenario keywords in environment name
        for scenario_key, config_func in scenario_map.items():
            if scenario_key in env_name:
                return config_func(difficulty)

        # Default to highway configuration
        return HighwayScenarioConfig.get_config(difficulty)
