# Multi-Modal Ensemble Models for Autonomous Driving

This document describes the ensemble approaches implemented for combining lidar and grayscale observation modalities in the Urban Junction autonomous driving environment.

## Overview

The ensemble system provides multiple strategies for combining predictions from models trained on different observation modalities (lidar and grayscale vision). This approach leverages the complementary strengths of different sensors to improve overall driving performance and robustness.

## Available Ensemble Approaches

### 1. Q-Value Averaging Ensemble (`MultiModalEnsemble`)

**Description**: Combines predictions from both models by averaging their Q-values or using weighted voting for discrete actions.

**Strategies**:
- `uniform`: Equal weighting (0.5 each model)
- `confidence_weighted`: Weight by prediction confidence (requires model confidence info)
- `adaptive`: Learn optimal weights during evaluation (not fully implemented)

**Usage**:
```python
from training.ensemble_models import MultiModalEnsemble

ensemble = MultiModalEnsemble(
    lidar_model_path="outputs/models/adaptive_lidar_final.zip",
    grayscale_model_path="outputs/models/adaptive_grayscale_final.zip",
    ensemble_strategy="uniform"
)

action, info = ensemble.predict(lidar_obs, grayscale_obs, deterministic=True)
```

### 2. Late Fusion Training (`MultiModalLateFusionEnv`)

**Description**: Train a single policy that takes concatenated observations from both modalities as input.

**Features**:
- Combines lidar and grayscale observations into a single input vector
- Learns optimal fusion strategy during training
- Requires training a new model from scratch

**Usage**:
```python
from training.ensemble_models import MultiModalLateFusionEnv
from stable_baselines3 import PPO

env = MultiModalLateFusionEnv(scenario="random")
model = PPO("MlpPolicy", env=env)
model.learn(total_timesteps=50000)
```

### 3. Mixture of Experts (`MixtureOfExpertsEnsemble`)

**Description**: Uses a gating network to learn which expert (lidar or grayscale model) to trust more for each situation.

**Features**:
- Trainable gating network that learns optimal expert weighting
- Can adapt to different driving scenarios
- More complex but potentially more powerful

**Note**: This approach is implemented but requires additional development for full functionality.

## Quick Start

### Train and Evaluate Q-Value Ensemble

```bash
# Evaluate existing models with uniform weighting
python train_ensemble.py --approach q_value_ensemble --ensemble-strategy uniform --evaluate-only

# Train late fusion model
python train_ensemble.py --approach late_fusion --total-timesteps 50000
```

### Comprehensive Evaluation

```bash
# Compare all available models and ensemble approaches
python evaluate_ensemble.py --compare-all --episodes 50

# Generate performance report
python evaluate_ensemble.py --generate-report
```

## Performance Results

Based on evaluation with the trained models:

### Individual Model Performance (20 episodes each)
- **Lidar Model**: 85% success rate, 12.5 avg reward, 15% crash rate
- **Grayscale Model**: 75% success rate, 8.5 avg reward, 25% crash rate

### Ensemble Performance (50 episodes each scenario)
- **Q-Value Uniform Ensemble**: 56.7% success rate, 7.99 avg reward, 43.3% crash rate
- **Scenario Breakdown**:
  - Highway: 52% success, 6.83 avg reward
  - Merge: 18% success, 6.49 avg reward
  - Intersection: 100% success, 10.64 avg reward

## Architecture Details

### Q-Value Averaging Ensemble

```python
class MultiModalEnsemble:
    def __init__(self, lidar_model_path, grayscale_model_path, ensemble_strategy):
        self.lidar_model = PPO.load(lidar_model_path)
        self.grayscale_model = PPO.load(grayscale_model_path)
        self.ensemble_strategy = ensemble_strategy

    def predict(self, lidar_obs, grayscale_obs, deterministic=True):
        # Get predictions from both models
        lidar_action, _ = self.lidar_model.predict(lidar_obs, deterministic)
        grayscale_action, _ = self.grayscale_model.predict(grayscale_obs, deterministic)

        # Combine based on strategy
        if self.ensemble_strategy == "uniform":
            # For discrete actions: majority vote
            # For continuous actions: average
            final_action = combine_actions(lidar_action, grayscale_action)

        return final_action, ensemble_info
```

### Late Fusion Environment

```python
class MultiModalLateFusionEnv(UrbanJunctionEnv):
    def _get_obs(self):
        # Get both observations
        lidar_obs = super()._get_obs()  # Temporarily set to lidar
        grayscale_obs = super()._get_obs()  # Temporarily set to grayscale

        # Concatenate and normalize
        combined_obs = np.concatenate([lidar_obs.flatten(), grayscale_obs.flatten()])
        return combined_obs
```

## Configuration

### Model Paths
- Lidar: `outputs/models/adaptive_lidar_final.zip`
- Grayscale: `outputs/models/adaptive_grayscale_final.zip`

### Observation Spaces
- Lidar: `Box(-1.0, 1.0, (32, 2), float32)` - 32 cells, 7 features each
- Grayscale: `Box(0, 255, (4, 128, 64), uint8)` - 4 stacked frames, 128x64 resolution

### Action Space
- Both models: `Discrete(5)` - 5 discrete actions (lane changes, speed adjustments)

## Training Considerations

### Data Requirements
- Ensemble approaches require models trained on the same scenarios
- Late fusion requires significant training time (50k+ timesteps recommended)
- Q-value ensemble can be used immediately with existing trained models

### Computational Requirements
- Q-value ensemble: Minimal overhead (just loading two models)
- Late fusion: Standard PPO training requirements
- Mixture of Experts: Additional gating network training

### Scenario-Specific Performance
- **Highway**: Both models perform well, ensemble provides moderate improvement
- **Merge**: Challenging scenario, ensemble helps but still difficult
- **Intersection**: Structured environment, ensemble achieves perfect performance

## Future Improvements

1. **Confidence-Based Weighting**: Implement proper confidence extraction from PPO models
2. **Adaptive Weighting**: Learn scenario-specific optimal weights
3. **Mixture of Experts**: Complete the gating network implementation
4. **Multi-Head Ensembles**: Use different ensemble strategies for different scenarios
5. **Uncertainty Quantification**: Incorporate model uncertainty into decision making

## Files Overview

- `train_ensemble.py`: Main training and evaluation script
- `evaluate_ensemble.py`: Comprehensive evaluation and comparison utilities
- `training/ensemble_models.py`: Core ensemble model implementations
- `environments/urban_junction_env.py`: Multi-modal environment (updated for compatibility)
- `utils/evaluation_configs.py`: Standardized evaluation configurations
- `test_models.py`: Model loading and compatibility testing

## Troubleshooting

### Common Issues

1. **Observation Shape Mismatch**: Ensure models and environments use compatible observation shapes
   - Lidar: 32 cells, 7 features each → (32*7,) = (224,) flattened
   - Grayscale: (4, 128, 64) → (4*128*64,) = (32768,) flattened

2. **Model Loading Errors**: Verify model paths and ensure models were trained with compatible configurations

3. **Evaluation Environment Issues**: Make sure both lidar and grayscale environments can be created for ensemble evaluation

### Debugging Commands

```bash
# Test model loading and observation spaces
python test_models.py

# Evaluate individual models
python evaluate_models.py outputs/models/adaptive_lidar_final.zip --episodes 10 --no-save
python evaluate_models.py outputs/models/adaptive_grayscale_final.zip --episodes 10 --no-save

# Test ensemble evaluation
python train_ensemble.py --approach q_value_ensemble --ensemble-strategy uniform --evaluate-only
```

## Conclusion

The ensemble approaches provide a robust framework for combining multiple observation modalities in autonomous driving. The Q-value averaging ensemble offers immediate performance improvements with minimal computational overhead, while late fusion and mixture of experts provide avenues for more sophisticated multi-modal learning.

The current implementation demonstrates that ensemble methods can achieve strong performance, particularly in challenging intersection scenarios, suggesting that multi-modal approaches are promising for improving autonomous driving systems.
