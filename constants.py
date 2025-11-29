"""
Centralized constants for the autonomous driving RL project.

This file contains all reward values, configuration constants, and magic numbers
used throughout the project to ensure consistency and maintainability.
"""

# No additional imports needed


class RewardConstants:
    """Reward values for different difficulty levels and scenarios."""

    # Collision penalties by difficulty
    COLLISION_EASY = -2.0
    COLLISION_MEDIUM = -5.0
    COLLISION_HARD = -10.0
    COLLISION_DEFAULT = -4.0

    # Speed rewards
    HIGH_SPEED_EASY = 0.2
    HIGH_SPEED_MEDIUM = 0.5
    HIGH_SPEED_HARD = 0.8

    # Other rewards
    ON_ROAD_REWARD = 0.3
    ARRIVED_INTERSECTION = 1.0
    RIGHT_LANE_BASE = 0.1

    # Penalties
    OFFROAD_PENALTY = -4.0


class EnvironmentConstants:
    """Constants for environment configuration."""

    # Simulation settings
    SIMULATION_FREQUENCY = 15
    POLICY_FREQUENCY = 1

    # Episode settings
    DEFAULT_DURATION = 40  # seconds
    MAX_EPISODE_STEPS_FALLBACK = 1000

    # Road network settings
    DEFAULT_LANE_COUNT = 4
    DEFAULT_HIGHWAY_LENGTH = 2500
    DEFAULT_SPEED_LIMIT = 30

    # Traffic settings
    HIGHWAY_TRAFFIC_DENSITY = 10
    URBAN_TRAFFIC_DENSITY = 5


class ObservationConstants:
    """Constants for observation configurations."""

    # Lidar settings
    LIDAR_CELLS = 32
    LIDAR_MAX_RANGE = 50
    LIDAR_ROW_ANCHOR = [0.5, 0.5]

    # Grayscale settings
    GRAYSCALE_OBSERVATION_SHAPE = (128, 64)
    GRAYSCALE_STACK_SIZE = 4
    GRAYSCALE_WEIGHTS = [0.2989, 0.5870, 0.1140]  # RGB to grayscale conversion
    GRAYSCALE_SCALING = 1.75

    # Normalization
    GRAYSCALE_NORMALIZATION_FACTOR = 127.5
    GRAYSCALE_MEAN = 1.0


class TrainingConstants:
    """Constants for training configuration."""

    # Difficulty parameters
    DIFFICULTIES = ["easy", "medium", "hard"]

    # Traffic density by difficulty
    TRAFFIC_DENSITY = {
        "easy": 10,
        "medium": 20,
        "hard": 30
    }

    # Episode duration by difficulty
    EPISODE_DURATION = {
        "easy": 40,
        "medium": 40,
        "hard": 60
    }

    # Reward speed ranges by difficulty
    REWARD_SPEED_RANGE = {
        "easy": [10, 30],
        "medium": [20, 30],
        "hard": [25, 35]
    }


class ScenarioConstants:
    """Constants for different driving scenarios."""

    SCENARIOS = ["highway", "merge", "intersection"]
    MODALITIES = ["lidar", "grayscale", "both"]

    # Scenario-specific settings
    MERGE_VEHICLE_PROBABILITY = {
        "easy": 0.3,
        "medium": 0.6,
        "hard": 0.8
    }

    INTERSECTION_SPAWN_PROBABILITY = {
        "easy": 0.2,
        "medium": 0.5,
        "hard": 0.7
    }

    # Special rewards
    INTERSECTION_COMPLETION_BONUS = 6.0
