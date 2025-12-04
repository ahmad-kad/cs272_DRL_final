# Crazy Driver Environment Training

This directory contains scripts to train agents for the `crazy_driver_env` - a challenging continuous control driving environment where an agent must dodge oncoming traffic while avoiding pursuing police cars.

## Environment Overview

- **Environment ID**: `CopChase-v0`
- **Observation Space**: 105D kinematic features (15 vehicles × 7 features each)
  - Features per vehicle: `[presence, x, y, vx, vy, cos_h, sin_h]`
- **Action Space**: 2D continuous `[-1, 1]²` (acceleration, steering)
- **Reward Structure**:
  - +0.5 for near-misses with oncoming traffic
  - +0.2 for speed rewards (20-30 m/s range)
  - -10 for collisions with police cars
  - -4 for collisions with regular NPCs
  - -10 penalty for getting behind police cars
  - Off-road penalties

## Training Scripts

### 1. Baseline PPO (`train_crazy_driver_baseline.py`)

Standard PPO implementation using MLP policy on kinematic observations.

```bash
# Train baseline PPO
python train_crazy_driver_baseline.py --steps 500000 --n_envs 8

# Custom training
python train_crazy_driver_baseline.py --steps 1000000 --n_envs 4 --model_name my_baseline
```

**Architecture**:
- MLP Policy: `[128, 64]` hidden layers for both actor and critic
- Standard PPO hyperparameters
- Learning rate: 3e-4

### 2. I2A with World Model (`train_crazy_driver_i2a.py`)

Imagination-Augmented Agent that uses a learned world model to simulate future trajectories before making decisions.

```bash
# Train I2A with optional world model pre-training
python train_crazy_driver_i2a.py --steps 500000 --n_envs 4 --pretrain_world_model --world_model_steps 100000

# Without pre-training (world model learns online)
python train_crazy_driver_i2a.py --steps 500000 --n_envs 4
```

**Architecture**:
- **World Model**: Predicts `(obs, action) → (next_obs, reward, done)`
- **Imagination**: 8 random action sequences × 5 steps each per decision
- **Feature Combination**: MLP([obs_features, imagination_features]) → 128D
- **Policy**: Smaller MLP `[64, 64]` on top of rich features

## Key Differences

| Aspect | Baseline PPO | I2A + World Model |
|--------|-------------|-------------------|
| **Decision Making** | Reactive (current state only) | Proactive (imagines futures) |
| **Training Time** | ~2-3 hours (500k steps) | ~6-8 hours (500k steps) |
| **Computational Cost** | Standard | ~2-3x higher |
| **Sample Efficiency** | Standard | Potentially better |
| **Strategic Planning** | Limited | Multi-step lookahead |

## Expected Performance Comparison

### Baseline PPO
- **Strengths**: Fast training, stable, good at reactive control
- **Weaknesses**: May get stuck in local optima, poor long-term planning
- **Expected Performance**: Moderate success rate, frequent collisions

### I2A + World Model
- **Strengths**: Better planning, anticipates collisions, more strategic
- **Weaknesses**: Slower training, higher variance, complex architecture
- **Expected Performance**: Higher success rate, fewer collisions, better evasion strategies

## Evaluation

After training, evaluate both models:

```bash
# Visualize trained models
python visualize_crazy_driver.py --model outputs/models/crazy_driver_baseline_500k.zip
python visualize_crazy_driver.py --model outputs/models/crazy_driver_i2a_500k.zip
```

## W&B Logging

Both scripts support Weights & Biases logging:

```bash
# Enable W&B logging (requires wandb setup)
python train_crazy_driver_baseline.py --steps 500000
python train_crazy_driver_i2a.py --steps 500000
```

## Implementation Notes

### World Model Architecture
```python
WorldModel:
Input: (obs[105] + action[2]) = 107D
→ Encoder: Linear(107→256) → ReLU → Linear(256→256) → ReLU
→ Heads: Linear(256→105), Linear(256→1), Linear(256→1)
Output: (next_obs[105], reward[1], done_prob[1])
```

### Imagination Process
```python
For each decision:
1. Sample 8 random action sequences (5 steps each)
2. Use world model to simulate trajectories
3. Encode each trajectory into 128D features
4. Average imagination features
5. Combine with observation features
6. Feed to PPO policy
```

### Training Challenges
- **World Model**: Needs accurate dynamics prediction
- **Imagination**: Computational overhead per step
- **Stability**: Complex architecture may be less stable
- **Hyperparameters**: May need tuning for optimal performance

## Future Improvements

1. **Better World Model**: Use transformer architecture for sequence modeling
2. **Curriculum Learning**: Start with simple scenarios, increase complexity
3. **Attention Mechanisms**: Focus on most relevant vehicles
4. **Hierarchical Policies**: High-level planning + low-level control

## Requirements

- `stable-baselines3>=2.0.0`
- `torch>=2.0.0`
- `gymnasium>=0.29.0`
- `highway-env>=1.8.0`
- Optional: `wandb>=0.16.0` for logging
