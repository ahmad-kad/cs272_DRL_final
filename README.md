# Urban Junction Environment

A highway environment for autonomous driving research with procedural scenario generation and adversarial traffic.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from highway_distillation.environments.urban_junction_env import UrbanJunctionEnv

# Create environment
env = UrbanJunctionEnv(config={
    'scenario': 'highway',        # 'highway', 'merge', or 'intersection'
    'observation_type': 'lidar',  # 'lidar' or 'grayscale'
    'duration': 60,               # episode length in seconds
    'vehicles_count': 10          # number of other vehicles
})

# Use the environment
obs, info = env.reset()
action = env.action_space.sample()  # 5 discrete actions
obs, reward, terminated, truncated, info = env.step(action)
```

## Environment Details

### Actions (5 discrete)
- `LANE_LEFT`, `LANE_RIGHT`, `FASTER`, `SLOWER`, `IDLE`

### Observations
- **Lidar**: 8 nearby vehicles (position, speed, presence) + optional context = 40-43 features
- **Grayscale**: Visual input with configurable dimensions
- **Frame stacking**: Optional 2-frame history

### Rewards
- Good speed (20-30 mph): +0.4
- Bad speed: -0.3
- Progress: +0.02 × speed
- Collision: -1.0 (episode ends)
- Red light violation: -0.4
- Off-road: -0.3
- Stage complete: +0.5
- Episode success: +2.0

### Scenarios
- **Highway**: Straight road with varying lane counts and traffic
- **Merge**: Highway with on-ramp merging scenarios
- **Intersection**: Urban intersection with crossing traffic

## Training

Train baseline agents:

```bash
# Quick test
python train_all_baselines.py --mode quick --env highway-v0 --obs Lidar --device cpu

# Full training across all environments and observation types
python train_all_baselines.py --all --mode standard --device cuda
```

## Project Structure

```
├── highway_distillation/
│   ├── environments/          # Urban Junction environment
│   ├── training/              # Training utilities
│   └── tests/                 # Test suites
├── train_*.py                 # Training scripts
├── run_*.py                   # Evaluation and utility scripts
└── requirements.txt           # Dependencies
```
