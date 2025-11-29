"""
Configuration module for autonomous driving RL environments.

This package provides shared configuration utilities and constants
for consistent environment setup across the project.
"""

from .base import BaseEnvironmentConfig
from .observations import (
    ObservationConfigFactory,
    LidarObservationConfig,
    GrayscaleObservationConfig
)
from .rewards import RewardConfig
from .scenarios import ScenarioConfigFactory

__all__ = [
    "BaseEnvironmentConfig",
    "ObservationConfigFactory",
    "LidarObservationConfig",
    "GrayscaleObservationConfig",
    "RewardConfig",
    "ScenarioConfigFactory",
]
