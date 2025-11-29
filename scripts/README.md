# Scripts Directory

This directory contains the core scripts for training, evaluation, and visualization of the enhanced autonomous driving RL system.

## Core Scripts

### Training Scripts

#### `train_enhanced_rewards.py`
**Main enhanced training script with safety constraints and curriculum learning.**

```bash
# Train on specific scenario
python train_enhanced_rewards.py --scenario highway --timesteps 5000

# Compare enhanced vs baseline rewards
python train_enhanced_rewards.py --scenario highway --compare --timesteps 3000

# Train with custom reward weights
python train_enhanced_rewards.py --scenario merge --custom-rewards
```

**Features:**
- Enhanced reward structure with safety constraints
- Hard safety constraints (emergency braking, speed limiting)
- Proximity-based collision avoidance
- Scenario-aware speed optimization
- Curriculum learning progression

#### `train_ensemble.py`
**Specialized ensemble training for multi-modal models.**

```bash
# Q-Value averaging ensemble
python train_ensemble.py --approach q_value_ensemble \
    --lidar-model results/models/lidar_expert.zip \
    --grayscale-model results/models/grayscale_expert.zip

# Late fusion training
python train_ensemble.py --approach late_fusion --total-timesteps 50000

# Mixture of experts
python train_ensemble.py --approach mixture_experts \
    --lidar-model results/models/lidar_expert.zip \
    --grayscale-model results/models/grayscale_expert.zip
```

**Ensemble Approaches:**
1. **Q-Value Averaging**: Combine predictions from pretrained models
2. **Late Fusion**: Train new policy on concatenated observations
3. **Mixture of Experts**: Learn optimal model weighting

### Evaluation Scripts

#### `evaluate_all.py`
**Comprehensive evaluation system for all trained models.**

```bash
# Evaluate all discovered models
python evaluate_all.py

# Custom evaluation parameters
python evaluate_all.py --episodes 20 --scenarios highway merge
```

**Features:**
- Automatic model discovery
- Multi-scenario evaluation (highway, merge, intersection)
- Multi-modality support (lidar, grayscale, both)
- Comprehensive metrics (reward, success, safety, efficiency)
- Statistical analysis and ranking
- JSON results export

### Visualization Scripts

#### `visualize_results.py`
**Interactive visualization dashboard for evaluation results.**

```bash
# Generate full visualization suite
python visualize_results.py

# Custom visualization options
python visualize_results.py --dashboard-only --interactive
```

**Visualizations:**
- **Performance Heatmap**: Model performance across scenarios/modalities
- **Radar Charts**: Multi-dimensional performance profiles
- **Curriculum Progression**: Learning curves over training phases
- **Interactive Dashboard**: HTML-based detailed analysis
- **Scenario Comparisons**: Side-by-side performance analysis

## Usage Workflow

### 1. Training Phase
```bash
# Train enhanced agent
python train_enhanced_rewards.py --scenario highway --timesteps 10000

# Train ensemble (optional)
python train_ensemble.py --approach late_fusion --total-timesteps 20000
```

### 2. Evaluation Phase
```bash
# Comprehensive evaluation
python evaluate_all.py
```

### 3. Analysis Phase
```bash
# Generate visualizations
python visualize_results.py
```

## Output Structure

```
results/
├── evaluations/
│   ├── comprehensive_evaluation_*.json    # Detailed metrics
│   └── evaluation_report.txt               # Executive summary
└── visualizations/
    ├── performance_heatmap.png             # Performance overview
    ├── radar_performance.html              # Interactive profiles
    ├── curriculum_progression.png          # Learning curves
    ├── model_comparison_dashboard.png      # Comparative analysis
    └── interactive_dashboard.html          # Full dashboard
```

## Key Metrics

### Performance Metrics
- **Average Reward**: Episode reward normalized by length
- **Success Rate**: Percentage of successful episodes
- **Crash Rate**: Safety violations per episode
- **Completion Rate**: Task completion percentage

### Safety Metrics
- **Safety Score**: 1 - crash_rate
- **Constraint Violations**: Hard constraint activation frequency
- **Proximity Warnings**: Early collision detection events

### Efficiency Metrics
- **Episode Length**: Steps to completion
- **Reward Efficiency**: Reward per timestep
- **Training Stability**: Learning curve smoothness

## Configuration

### Reward Structure
```python
enhanced_rewards = {
    "collision_reward": -20.0,      # Strong crash penalty
    "proximity_penalty": -2.0,      # Early collision warning
    "speed_reward": 2.0,            # Scenario-aware speed optimization
    "safe_maneuver_bonus": 1.0,     # Collision avoidance rewards
    "lane_change_penalty": -0.05,   # Reduced for safety maneuvers
}
```

### Safety Constraints
```python
safety_config = {
    "emergency_brake_threshold": 3.0,    # Meters for emergency braking
    "max_speed_limit": 35.0,             # Km/h speed limit
    "lane_deviation_threshold": 4.0,     # Lane position limit
    "proximity_zones": [3.0, 7.0, 12.0] # Warning zones in meters
}
```

## Dependencies

- `stable-baselines3`: RL training and evaluation
- `numpy`: Numerical computations
- `pandas`: Data analysis and manipulation
- `matplotlib`: Static plotting
- `plotly`: Interactive visualizations
- `tqdm`: Progress bars
- `gymnasium`: Environment interface

## Troubleshooting

### Common Issues

**Model Loading Errors:**
```bash
# Check model exists
ls results/models/*.zip

# Verify model compatibility
python -c "from stable_baselines3 import PPO; PPO.load('model.zip')"
```

**Evaluation Timeouts:**
```bash
# Reduce evaluation episodes
python evaluate_all.py --episodes 5

# Test single scenario
python evaluate_all.py --scenarios highway
```

**Visualization Errors:**
```bash
# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"

# Use non-interactive backend
export MPLBACKEND=Agg
```

### Performance Optimization

**For Large-Scale Evaluation:**
```bash
# Reduce episodes per evaluation
python evaluate_all.py --episodes 10

# Evaluate subset of models
# Manually specify model paths in script
```

**For Fast Iteration:**
```bash
# Quick training test
python train_enhanced_rewards.py --scenario highway --timesteps 1000

# Quick evaluation
python evaluate_all.py --episodes 3
```

## Integration

These scripts integrate with the main system through:

- **Environment**: `environments/enhanced_urban_env.py`
- **Training**: `training/adaptive_curriculum_trainer.py`
- **Configuration**: `config/` directory
- **Results**: `results/` directory structure

For full system integration, see the main `run_complete_system.py` orchestrator.
