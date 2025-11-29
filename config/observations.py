"""
Observation configurations for autonomous driving RL environments.

This module handles all observation-related configuration logic.
"""

from typing import Dict, Any
from constants import ObservationConstants


class LidarObservationConfig:
    """Handles Lidar observation configurations."""

    @staticmethod
    def get_basic_config() -> Dict[str, Any]:
        """Get basic lidar observation configuration."""
        return {
            "type": "LidarObservation",
            "cells": ObservationConstants.LIDAR_CELLS,
            "maximum_range": ObservationConstants.LIDAR_MAX_RANGE,
            "normalize": True,
            "features": ["presence", "x"],
        }

    @staticmethod
    def get_curriculum_config() -> Dict[str, Any]:
        """Get lidar configuration optimized for curriculum learning."""
        return {
            "type": "LidarObservation",
            "cells": ObservationConstants.LIDAR_CELLS,
            "row_anchor": ObservationConstants.LIDAR_ROW_ANCHOR,
            "features": ["presence", "distance", "speed"],
            "features_range": {
                "distance": [0, ObservationConstants.LIDAR_MAX_RANGE],
                "speed": [-30, 30]
            }
        }


class GrayscaleObservationConfig:
    """Handles Grayscale observation configurations."""

    @staticmethod
    def get_basic_config() -> Dict[str, Any]:
        """Get basic grayscale observation configuration."""
        return {
            "type": "GrayscaleObservation",
            "observation_shape": ObservationConstants.GRAYSCALE_OBSERVATION_SHAPE,
            "stack_size": ObservationConstants.GRAYSCALE_STACK_SIZE,
            "weights": ObservationConstants.GRAYSCALE_WEIGHTS,
            "scaling": ObservationConstants.GRAYSCALE_SCALING,
        }


class ObservationConfigFactory:
    """Factory for creating observation configurations."""

    @staticmethod
    def create_lidar_config(use_curriculum_features: bool = False) -> Dict[str, Any]:
        """Create lidar observation configuration."""
        if use_curriculum_features:
            return LidarObservationConfig.get_curriculum_config()
        return LidarObservationConfig.get_basic_config()

    @staticmethod
    def create_grayscale_config() -> Dict[str, Any]:
        """Create grayscale observation configuration."""
        return GrayscaleObservationConfig.get_basic_config()

    @staticmethod
    def create_combined_config() -> Dict[str, Any]:
        """Create combined observation configuration (lidar base for environment handling)."""
        return LidarObservationConfig.get_curriculum_config()

    @staticmethod
    def create_config(modality: str, use_curriculum_features: bool = False) -> Dict[str, Any]:
        """Create observation configuration based on modality."""
        modality_map = {
            "lidar": lambda: ObservationConfigFactory.create_lidar_config(use_curriculum_features),
            "grayscale": ObservationConfigFactory.create_grayscale_config,
            "both": ObservationConfigFactory.create_combined_config,
        }

        if modality not in modality_map:
            raise ValueError(f"Unknown modality: {modality}")

        return modality_map[modality]()
