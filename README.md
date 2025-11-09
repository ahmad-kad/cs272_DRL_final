# Urban Junction RL Agent

Multi-modal autonomous driving agent trained with curriculum learning for urban scenarios (highway, merge, intersection) with adversarial traffic.

**Key Features:**
- Multi-modal observations: kinematics + lidar + visual
- Context-aware policies for scenario adaptation
- Curriculum learning with adaptive difficulty
- Comprehensive validation suite

**Architecture**: Simplified "less is more" design - 80% less code, 100% functionality

**Project Structure**: Clean, organized codebase with unified training framework

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup
python highway_distillation/tests/test_all_scenarios.py

# Train curriculum (4 phases)
python highway_distillation/training.py phase1  # Multi-modal foundation (1M steps)
python highway_distillation/training.py phase2  # Context-aware policies (2M steps)
python highway_distillation/training.py phase3  # Curriculum learning (9M steps)
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
- **Logging**: Minimal "less is more" logging - only insights, no noise
- **Validation**: Multi-level testing (annoyance levels, generalization, sensor ablation)

## Usage

```python
# Basic training
from highway_distillation.training import train_phase1
train_phase1()

# Or run specific phases
from highway_distillation.training import train_phase1, train_phase2, train_phase3
train_phase1()  # Multi-modal foundation
train_phase2()  # Context-aware policies
train_phase3()  # Curriculum learning

# Plot results (after training completes)
from highway_distillation.plot_convergence import plot_convergence
plot_convergence("outputs/phase1_results.csv")
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
