# Urban Junction Environment

Highway environment for autonomous driving research with procedural scenario generation and adversarial traffic.

## Features

- **Procedural Generation**: Varied road layouts, traffic patterns, and environmental conditions
- **Multi-Scenario Support**: Highway, merge, and intersection scenarios
- **Adversarial Traffic**: Intelligent vehicles that challenge the ego vehicle
- **Multi-modal Observations**: Lidar and visual observations available
- **Highway-Env Integration**: Built on the established highway-env framework

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- highway-env>=1.8.2
- gymnasium>=0.29.0
- numpy>=1.24.0

## Usage

```python
from highway_distillation.environments.urban_junction_env import UrbanJunctionEnv

# Create environment
env = UrbanJunctionEnv(
    config={
        'scenario': 'highway',  # 'highway', 'merge', or 'intersection'
        'observation_type': 'lidar',  # 'lidar' or 'grayscale'
        'duration': 60,
        'vehicles_count': 10
    }
)

# Reset and step
obs, info = env.reset()
action = env.action_space.sample()  # Your policy here
obs, reward, terminated, truncated, info = env.step(action)
```

## Environment Configuration

- **scenario**: Type of driving scenario (`'highway'`, `'merge'`, `'intersection'`)
- **observation_type**: Sensor modality (`'lidar'`, `'grayscale'`)
- **duration**: Episode length in seconds
- **vehicles_count**: Number of other vehicles
- **lidar_rays**: Number of lidar rays (for lidar observations)
- **visual_height/width**: Image dimensions (for grayscale observations)

## Scenarios

- **Highway**: Straight road with varying lane counts and traffic
- **Merge**: Highway with on-ramp merging scenarios
- **Intersection**: Urban intersection with crossing traffic

## Research Applications

This environment is designed for:
- Reinforcement learning research
- Autonomous driving algorithm development
- Multi-modal sensor fusion studies
- Adversarial scenario testing
