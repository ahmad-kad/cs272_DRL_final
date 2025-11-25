#!/usr/bin/env python3
"""Test the new 'both' modality in UrbanJunctionEnv."""

from environments.urban_junction_env import UrbanJunctionEnv

print("Testing UrbanJunctionEnv with modality='both'...")

# Test with default (should be 'both' now)
env = UrbanJunctionEnv(scenario="highway")
print(f"Modality: {env.modality}")
print(f"Observation space: {env.observation_space}")

# Test reset
obs, info = env.reset()
print(f"Observation shape after reset: {obs.shape}")
print(f"Expected shape: {env.observation_space.shape}")

# Test step
action = env.action_space.sample()
step_result = env.step(action)
if len(step_result) == 5:
    next_obs, reward, terminated, truncated, info = step_result
else:
    next_obs, reward, done, info = step_result

print(f"Observation shape after step: {next_obs.shape}")

env.close()
print("Test completed successfully!")
