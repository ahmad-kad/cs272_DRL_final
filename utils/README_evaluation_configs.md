# Evaluation Configurations

This document describes the standardized evaluation configurations for benchmarking autonomous driving models across different scenarios and observation modalities.

## Overview

The evaluation configurations provide standardized settings for evaluating models on three driving scenarios (highway, merge, intersection) with two observation modalities (lidar, grayscale). These configurations are optimized for model benchmarking rather than training.

## Scenarios

### Highway
- **Description**: Standard highway cruising with moderate traffic
- **Duration**: 60 seconds (longer episodes for highway evaluation)
- **Traffic**: 15 vehicles (moderate density)
- **Focus**: Lane keeping, speed control, traffic flow navigation

### Merge
- **Description**: Highway merge scenario with merging traffic
- **Duration**: 40 seconds (focused on merge completion)
- **Traffic**: 12 vehicles (moderate with merging vehicles)
- **Focus**: Safe merging, traffic gap detection, lane changes

### Intersection
- **Description**: Urban intersection with cross traffic
- **Duration**: 30 seconds (complex interactions)
- **Traffic**: 10 vehicles (lower density, higher complexity)
- **Focus**: Right-of-way, collision avoidance, traffic light navigation

## Modalities

### Lidar
- **Type**: LidarObservation
- **Cells**: 64 (higher resolution for evaluation)
- **Range**: 50 meters
- **Features**: presence, x, y, vx, vy, cos_h, sin_h
- **Normalization**: True

### Grayscale
- **Type**: GrayscaleObservation
- **Shape**: (128, 64)
- **Stack Size**: 4 (temporal frames)
- **Weights**: [0.2989, 0.5870, 0.1140] (RGB to grayscale)
- **Scaling**: 1.75

## Usage

### Basic Usage
```python
from utils.evaluation_configs import get_evaluation_config

# Get configuration for highway scenario with lidar
config = get_evaluation_config("highway", "lidar")

# Create environment
env = UrbanJunctionEnv(config=config, scenario="highway", modality="lidar")
```

### Pre-defined Configurations
```python
from utils.evaluation_configs import (
    HIGHWAY_LIDAR_CONFIG,
    HIGHWAY_GRAYSCALE_CONFIG,
    MERGE_LIDAR_CONFIG,
    MERGE_GRAYSCALE_CONFIG,
    INTERSECTION_LIDAR_CONFIG,
    INTERSECTION_GRAYSCALE_CONFIG
)
```

### Comprehensive Evaluation
```python
from utils.evaluation_configs import get_all_evaluation_configs

# Get all configurations
configs = get_all_evaluation_configs()
# Returns: scenario -> modality -> config dict
```

### Model Evaluation Script
Use the provided evaluation script for comprehensive model testing:

```bash
python evaluate_models.py path/to/model.zip --episodes 50
```

## Configuration Parameters

### Common Settings
- **Action Type**: DiscreteMetaAction (stable for evaluation)
- **Simulation Frequency**: 15 Hz
- **Policy Frequency**: 1 Hz
- **Reward Normalization**: True (for PPO stability)

### Scenario-Specific Parameters
- **Collision Reward**: -1.0 (standardized penalty)
- **Speed Rewards**: Optimized for each scenario
- **Arrival Rewards**: Scenario-specific completion bonuses
- **Traffic Probabilities**: Realistic spawning rates

## Evaluation Metrics

Models are evaluated on:
- **Success Rate**: Episodes with positive reward and no crashes
- **Crash Rate**: Average crashes per episode
- **Average Reward**: Mean episode reward
- **Episode Length**: Average steps per episode

## Best Practices

1. **Reproducibility**: Use fixed random seeds for consistent evaluation
2. **Episode Count**: Use at least 50 episodes per configuration
3. **Deterministic Evaluation**: Use deterministic actions for final evaluation
4. **Cross-Scenario**: Evaluate models across all scenarios for comprehensive assessment

## File Structure

```
utils/
├── evaluation_configs.py      # Main configuration definitions
├── README_evaluation_configs.md # This documentation
└── ../evaluate_models.py      # Evaluation script
```
