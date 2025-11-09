# Urban Junction RL Agent

Multi-modal autonomous driving agent trained with curriculum learning for urban scenarios (highway, merge, intersection) with adversarial traffic.

**Key Features:**
- Multi-modal observations: kinematics + lidar + visual
- Context-aware policies for scenario adaptation
- Curriculum learning with adaptive difficulty
- Comprehensive validation suite

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup
python highway_distillation/tests/test_all_scenarios.py

# Train curriculum (4 phases)
python highway_distillation/training/phase1_training.py  # Multi-modal foundation
python highway_distillation/training/phase2_training.py  # Context-aware policies
python highway_distillation/training/phase3_training.py  # Curriculum learning
python highway_distillation/training/phase4_validation.py  # Validation
```

## Training Phases

| Phase | Focus | Timesteps | Key Feature |
|-------|-------|-----------|-------------|
| 1 | Multi-modal foundation | 1M | Sensor fusion (kinematics + lidar + visual) |
| 2 | Context awareness | 2M | Scenario-specific behaviors |
| 3 | Curriculum learning | 6M | Adaptive difficulty + adversarial traffic |
| 4 | Validation | - | Comprehensive testing suite |

## Key Components

- **Environment**: Urban scenarios (highway/merge/intersection) with antagonistic vehicles
- **Observations**: Multi-modal (kinematics + lidar + visual) or context-aware
- **Policies**: Custom neural architectures for sensor fusion and context adaptation
- **Logging**: Insight-focused logging (minimal but crucial information)
- **Validation**: Multi-level testing (annoyance levels, generalization, sensor ablation)

## Usage

```python
# Basic training
from highway_distillation.training.phase1_training import train_phase1
train_phase1()

# Plot results
from highway_distillation.plot_convergence import plot_convergence
plot_convergence("outputs/data/phase1_convergence_data.csv")
```

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- Stable Baselines3
- NumPy, Pandas, Matplotlib

## Troubleshooting

**PyTorch CUDA issues**: Ensure compatible CUDA version
**Gymnasium conflicts**: `pip install "gymnasium>=1.0.0,<2.0.0"`
**Memory issues**: Reduce batch size or model dimensions
