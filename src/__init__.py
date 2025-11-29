"""
Enhanced Autonomous Driving RL System

Core package for safe and effective autonomous driving through
reinforcement learning with safety constraints and curriculum learning.
"""

from .environments import EnhancedUrbanJunctionEnv
from .training import AdaptiveCurriculumTrainer

__version__ = "1.0.0"
__author__ = "AI Assistant"

__all__ = [
    "EnhancedUrbanJunctionEnv",
    "AdaptiveCurriculumTrainer"
]
