"""
Reward configurations for autonomous driving RL environments.

This module handles all reward-related configuration logic.
"""

from typing import Dict, Any
from constants import RewardConstants, TrainingConstants


class RewardConfig:
    """Handles reward configuration based on difficulty level."""

    @staticmethod
    def get_easy_rewards() -> Dict[str, Any]:
        """Get reward configuration for easy difficulty."""
        return {
            "collision_reward": RewardConstants.COLLISION_EASY,
            "high_speed_reward": RewardConstants.HIGH_SPEED_EASY,
            "reward_speed_range": TrainingConstants.REWARD_SPEED_RANGE["easy"],
            "on_road_reward": RewardConstants.ON_ROAD_REWARD,
            "offroad_penalty": RewardConstants.OFFROAD_PENALTY,
        }

    @staticmethod
    def get_medium_rewards() -> Dict[str, Any]:
        """Get reward configuration for medium difficulty."""
        return {
            "collision_reward": RewardConstants.COLLISION_MEDIUM,
            "high_speed_reward": RewardConstants.HIGH_SPEED_MEDIUM,
            "reward_speed_range": TrainingConstants.REWARD_SPEED_RANGE["medium"],
            "on_road_reward": RewardConstants.ON_ROAD_REWARD,
            "offroad_penalty": RewardConstants.OFFROAD_PENALTY,
        }

    @staticmethod
    def get_hard_rewards() -> Dict[str, Any]:
        """Get reward configuration for hard difficulty."""
        return {
            "collision_reward": RewardConstants.COLLISION_HARD,
            "high_speed_reward": RewardConstants.HIGH_SPEED_HARD,
            "reward_speed_range": TrainingConstants.REWARD_SPEED_RANGE["hard"],
            "on_road_reward": RewardConstants.ON_ROAD_REWARD,
            "offroad_penalty": RewardConstants.OFFROAD_PENALTY,
        }

    @staticmethod
    def get_by_difficulty(difficulty: str) -> Dict[str, Any]:
        """Get reward configuration based on difficulty level."""
        difficulty_map = {
            "easy": RewardConfig.get_easy_rewards,
            "medium": RewardConfig.get_medium_rewards,
            "hard": RewardConfig.get_hard_rewards,
        }

        if difficulty not in difficulty_map:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        return difficulty_map[difficulty]()
