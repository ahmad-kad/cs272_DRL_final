# Training Ready

**Status**: All systems verified and ready for training.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup
python highway_distillation/tests/test_all_scenarios.py
python highway_distillation/tests/test_training_pipeline.py

## Training

```bash
# Quick test (10K timesteps each)
TEST_MODE=true python highway_distillation/training/phase1_training.py
TEST_MODE=true python highway_distillation/training/phase2_training.py
TEST_MODE=true python highway_distillation/training/phase3_training.py

# Full training (1M+ timesteps each)
python highway_distillation/training/phase1_training.py
python highway_distillation/training/phase2_training.py
python highway_distillation/training/phase3_training.py
python highway_distillation/training/phase4_validation.py
```

## Key Metrics

After training, expect:
- Phase 1: >90% success in basic scenarios
- Phase 2: Context-aware behavior differences
- Phase 3: >75% success with moderate traffic difficulty
- Phase 4: >70% success under extreme conditions

## Troubleshooting

**Memory issues**: Reduce `n_envs` or `batch_size`
**CUDA problems**: Use CPU-only PyTorch
**No learning**: Check reward normalization, adjust learning rate
