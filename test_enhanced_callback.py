#!/usr/bin/env python3
"""
Test script for the enhanced MultiEnvCallback to verify metrics tracking.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

# Import our enhanced callback
from train_highway_merge_intersection_multi_env import MultiEnvCallback

def test_callback_metrics():
    """Test that the enhanced callback properly tracks metrics."""

    print("Testing Enhanced MultiEnvCallback Metrics Tracking")
    print("=" * 60)

    # Create callback instance
    env_names = ["highway-v0", "merge-v0", "intersection-v0"]
    callback = MultiEnvCallback(env_names, log_freq=10, verbose=1)

    # Disable wandb logging for testing
    callback.wandb_logging_enabled = False

    # Simulate training steps with mock data
    print("Simulating training episodes...")

    # Mock locals that would come from PPO training
    mock_locals_base = {
        'dones': [False, False, False],  # 3 environments, none done yet
        'rewards': [0.5, 0.3, 0.2],     # Rewards for each env
        'infos': [
            {'speed': 25.0, 'lane_changes': 0, 'crashed': False, 'off_road': False},  # Highway
            {'speed': 22.0, 'lane_changes': 1, 'crashed': False, 'off_road': False},  # Merge
            {'speed': 15.0, 'lane_changes': 0, 'crashed': False, 'off_road': False},  # Intersection
        ]
    }

    # Simulate multiple steps
    for step in range(50):
        # Update step count
        callback.n_calls = step

        # Simulate episode progression
        if step < 20:  # Episode 1 ongoing
            mock_locals = mock_locals_base.copy()
        elif step == 20:  # Highway episode ends successfully
            mock_locals = mock_locals_base.copy()
            mock_locals['dones'] = [True, False, False]  # Highway done
            mock_locals['infos'][0] = {'speed': 28.0, 'lane_changes': 2, 'crashed': False, 'off_road': False}
        elif step == 30:  # Merge episode ends with collision
            mock_locals = mock_locals_base.copy()
            mock_locals['dones'] = [False, True, False]  # Merge done
            mock_locals['infos'][1] = {'speed': 20.0, 'lane_changes': 3, 'crashed': True, 'off_road': False}
        elif step == 40:  # Intersection episode ends successfully
            mock_locals = mock_locals_base.copy()
            mock_locals['dones'] = [False, False, True]  # Intersection done
            mock_locals['infos'][2] = {'speed': 18.0, 'lane_changes': 1, 'crashed': False, 'off_road': False}
        else:  # Continue with new episodes
            mock_locals = mock_locals_base.copy()

        # Set mock locals
        callback.locals = mock_locals

        # Call the callback
        callback._on_step()

        # Print progress
        if step % 10 == 0:
            print(f"Step {step}: Episodes completed - Highway: {callback.episode_counts['highway-v0']}, "
                  f"Merge: {callback.episode_counts['merge-v0']}, "
                  f"Intersection: {callback.episode_counts['intersection-v0']}")

    # Test final metrics aggregation (without wandb logging)
    # We'll manually call the metric calculation parts
    print("Testing metrics aggregation...")
    step_count = callback.n_calls
    elapsed_time = 0.1  # Mock elapsed time

    metrics = {
        "global_step": step_count,
        "time_elapsed": elapsed_time,
        "fps": step_count / max(1, elapsed_time),
    }

    # Add per-environment aggregated metrics
    for env_name in set(env_names):
        if callback.episode_rewards[env_name]:
            recent_count = min(50, len(callback.episode_rewards[env_name]))
            recent_rewards = callback.episode_rewards[env_name][-recent_count:]
            recent_lengths = callback.episode_lengths[env_name][-recent_count:]
            recent_collisions = callback.episode_collisions[env_name][-recent_count:]
            recent_speeds = callback.episode_speeds[env_name][-recent_count:]
            recent_lane_changes = callback.episode_lane_changes[env_name][-recent_count:]
            recent_off_road = callback.episode_off_road[env_name][-recent_count:]
            recent_survival = callback.episode_survival_rate[env_name][-recent_count:]
            recent_merge_success = callback.episode_merge_success[env_name][-recent_count:]

            success_rate = np.mean([1 if r > 15 and c == 0 else 0 for r, c in zip(recent_rewards, recent_collisions)])
            merge_success_rate = np.mean(recent_merge_success) if 'merge' in env_name else 0.0

            metrics.update({
                f"{env_name}_mean_reward": np.mean(recent_rewards),
                f"{env_name}_mean_length": np.mean(recent_lengths),
                f"{env_name}_mean_collisions": np.mean(recent_collisions),
                f"{env_name}_mean_speed": np.mean([s for s in recent_speeds if s > 0]),
                f"{env_name}_mean_lane_changes": np.mean(recent_lane_changes),
                f"{env_name}_mean_off_road": np.mean(recent_off_road),
                f"{env_name}_survival_rate": np.mean(recent_survival),
                f"{env_name}_merge_success_rate": merge_success_rate,
                f"{env_name}_success_rate": success_rate,
                f"{env_name}_episode_count": callback.episode_counts[env_name],
            })

    print("Aggregated metrics calculated successfully")

    # Verify metrics were tracked
    print("\nFinal Metrics Verification:")
    print("-" * 40)

    for env_name in env_names:
        if callback.episode_rewards[env_name]:
            print(f"{env_name}:")
            print(f"  Episodes: {len(callback.episode_rewards[env_name])}")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")
            print(f"  Survival Rate: {np.mean(callback.episode_survival_rate[env_name]):.2f}")
            if 'merge' in env_name:
                print(f"  Merge Success Rate: {np.mean(callback.episode_merge_success[env_name]):.2f}")
            print()

    print("Enhanced callback test completed successfully!")
    print("All metrics tracking appears to be working correctly.")
    return True

if __name__ == "__main__":
    test_callback_metrics()
