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

## End-to-End Traffic Flow Learning

This project features an advanced approach where **single agents automatically learn to recognize and adapt to traffic flow patterns** across multiple driving scenarios without explicit scenario detection.

### Key Innovation

Instead of training separate models for highway, merge, and intersection scenarios, we train **one neural network** that learns traffic patterns end-to-end:

- **Highway Traffic**: Consistent forward/backward flow, lane-structured traffic
- **Merge Scenarios**: Side-approaching vehicles, speed differentials, lane changes
- **Intersection Dynamics**: Cross-traffic, perpendicular movement, complex interactions

### Enhanced Multi-Environment Training

```bash
# Train single agent on all three scenarios simultaneously
python train_highway_merge_intersection_multi_env.py
```

**Features:**
- **Automatic Adaptation**: Network learns scenario patterns from lidar data automatically
- **Comprehensive Metrics**: Episode-level tracking in WandB (rewards, collisions, survival, merge success)
- **No Catastrophic Forgetting**: Simultaneous training avoids transfer learning degradation
- **Real-time Charts**: Monitor learning progress across all scenarios

### Expected Learning Progression

1. **Early Training**: Basic highway driving (lane following, speed control)
2. **Mid Training**: Merge behavior discovery (yielding, gap finding)
3. **Late Training**: Intersection mastery (right-of-way, cross-traffic navigation)

### WandB Metrics Dashboard

The enhanced training provides detailed charts showing:
- Episode rewards over training time
- Collision rates and safety improvement
- Survival rates (episodes completed without crashes)
- Merge success rates in merge scenarios
- Speed adaptation across different traffic conditions
- Lane change behaviors and learning

## Project Structure

```
├── highway_distillation/
│   ├── environments/          # Urban Junction environment
│   ├── training/              # Training utilities
│   └── tests/                 # Test suites
├── train_*.py                 # Training scripts (including enhanced multi-env)
├── run_*.py                   # Evaluation and utility scripts
├── evaluate_*.py              # Comprehensive evaluation suites
└── requirements.txt           # Dependencies
```
