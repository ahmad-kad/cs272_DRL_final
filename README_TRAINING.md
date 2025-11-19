# Highway Distillation - Generalist Agent Training

## Project Goal
Train a **single agent** to beat multiple highway-env environments:
- Highway (high-speed driving)
- Merge (merging behavior)
- Intersection (complex navigation)

## Strategy: Generalist Specialist Approach
Instead of training 6 separate agents (3 envs × 2 sensors), we train **2 generalist agents**:
1. **Lidar Generalist** - Trains on mixed scenarios using only lidar (32 rays)
2. **Grayscale Generalist** - Trains on mixed scenarios using only vision (64x64 images)

Both agents learn universal driving patterns that transfer across environments.

## Quick Start

### 1. Validate Setup (2-3 minutes)
```bash
python test_setup.py --agent lidar --steps 2000
```

### 2. Production Training (1.5-2 hours)
```bash
python run_training.py --agent all
```
This trains:
- Lidar Agent: 300K steps, ~20-30 minutes
- Grayscale Agent: 500K steps, ~45-60 minutes

### 3. Generate Evaluation Plots
```bash
python run_evaluation.py
```
Produces 12+ plots for project submission.

## Project Structure
```
cs272_DRL_final/
├── run_training.py          # Main training entry point
├── run_evaluation.py        # Generate plots
├── test_setup.py            # Quick validation
└── highway_distillation/
    ├── training.py          # Training logic with WandB
    ├── config.py            # Hyperparameters
    ├── environments/
    │   └── urban_junction_env.py  # Custom mixed environment
    └── outputs/
        ├── models/          # Saved agents
        ├── logs/            # Training logs
        └── plots/           # Evaluation plots
```

## Training Configuration

**Lidar Agent:**
- Observation: 32 lidar rays (physics-only)
- Policy: MLP [256, 256]
- Parallel Envs: 8
- Timesteps: 300,000
- Speed: Very fast (~1500 steps/sec)

**Grayscale Agent:**
- Observation: 64x64 grayscale images
- Policy: CnnPolicy (NatureCNN)
- Parallel Envs: 4
- Timesteps: 500,000
- Speed: Moderate (~500 steps/sec)

## Monitoring

**WandB Dashboard:**
https://wandb.ai/[username]/highway-distillation-generalist

**Key Metrics:**
- `episode_avg_reward` - Rolling average (should increase)
- `learning_improvement` - Net gain from start
- `value_loss` - Should decrease
- `policy_gradient_loss` - Should be small

## Expected Results

**After Training:**
- Lidar Agent: Mean reward 15-30 on mixed scenarios
- Grayscale Agent: Mean reward 10-25 on mixed scenarios
- Both agents generalize to Highway/Merge/Intersection

**Success Criteria:**
- ✅ No training crashes
- ✅ Reward trends upward
- ✅ Models save successfully
- ✅ Agents beat baseline performance

## Troubleshooting

**Import Errors:**
```bash
python -c "from highway_distillation import training"
```

**Slow Training:**
- Reduce parallel_envs in config.py
- Use only Lidar agent (skip Grayscale)

**Memory Issues:**
- Reduce batch_size in config.py
- Close other GPU applications

## Next Steps

1. ✅ Run validation test
2. ✅ Start production training
3. ✅ Monitor WandB dashboard
4. ✅ Generate evaluation plots
5. ✅ Submit results with plots

For detailed documentation, see `TRAINING_READY.md`.

