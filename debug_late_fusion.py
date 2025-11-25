#!/usr/bin/env python3
from environments.urban_junction_env import UrbanJunctionEnv
from training.ensemble_models import MultiModalLateFusionEnv

# Test individual environments
env_lidar = UrbanJunctionEnv(scenario='highway', modality='lidar')
obs_lidar, info = env_lidar.reset()
print(f'Lidar obs shape: {obs_lidar.shape}, size: {obs_lidar.size}')

env_gray = UrbanJunctionEnv(scenario='highway', modality='grayscale')
obs_gray, info = env_gray.reset()
print(f'Grayscale obs shape: {obs_gray.shape}, size: {obs_gray.size}')

print(f'Expected total size: {obs_lidar.size + obs_gray.size}')

# Test late fusion environment
env2 = MultiModalLateFusionEnv(scenario='highway')
obs2, info = env2.reset()
print(f'Late fusion env obs shape: {obs2.shape}')
print(f'Late fusion obs space: {env2.observation_space}')
