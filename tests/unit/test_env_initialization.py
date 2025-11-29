#!/usr/bin/env python3
"""
Test script to verify the urban junction environment still works correctly.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import numpy as np

def test_env_initialization():
    """Test that the environment can be initialized correctly."""
    print("Testing environment initialization...")

    try:
        from environments.urban_junction_env import UrbanJunctionEnv

        # Test different scenarios and modalities
        scenarios = ["highway", "merge", "intersection"]
        modalities = ["lidar", "grayscale", "both"]

        for scenario in scenarios:
            for modality in modalities:
                try:
                    env = UrbanJunctionEnv(scenario=scenario, modality=modality, render_mode=None)
                    print(f"✓ Environment initialized: {scenario} - {modality}")

                    # Test a simple step
                    obs, info = env.reset()
                    action = env.action_space.sample()
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    # Verify observation shapes
                    if modality == "lidar":
                        assert obs.shape == (32, 2), f"Lidar shape incorrect: {obs.shape}"
                    elif modality == "grayscale":
                        assert obs.shape == (4, 128, 64), f"Grayscale shape incorrect: {obs.shape}"
                    elif modality == "both":
                        assert obs.shape == (32 * 2 + 4 * 128 * 64,), f"Both shape incorrect: {obs.shape}"

                    env.close()
                    print(f"✓ Environment step test passed: {scenario} - {modality}")

                except Exception as e:
                    print(f"✗ Environment failed: {scenario} - {modality}: {e}")
                    return False

        return True

    except ImportError as e:
        print(f"Import error: {e}")
        return False

def test_reward_calculation():
    """Test that reward calculation works correctly."""
    print("Testing reward calculation...")

    try:
        from environments.urban_junction_env import UrbanJunctionEnv

        env = UrbanJunctionEnv(scenario="highway", modality="lidar", render_mode=None)
        obs, info = env.reset()

        # Test a few steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Verify reward is a float
            assert isinstance(reward, (int, float)), f"Reward should be numeric, got {type(reward)}"

            # Verify reward is reasonable
            assert -10 <= reward <= 10, f"Reward out of expected range: {reward}"

            if terminated or truncated:
                break

        env.close()
        print("✓ Reward calculation test passed")
        return True

    except Exception as e:
        print(f"✗ Reward calculation test failed: {e}")
        return False

def test_config_compatibility():
    """Test that the environment works with the refactored config system."""
    print("Testing config compatibility...")

    try:
        from environments.urban_junction_env import UrbanJunctionEnv
        from utils.config import get_curriculum_config

        # Test with curriculum config
        config = get_curriculum_config("highway-v0", "easy", "lidar")

        # Create environment with config
        env = UrbanJunctionEnv(config=config, scenario="highway", modality="lidar", render_mode=None)

        # Verify config was applied
        assert env.config["collision_reward"] == config["collision_reward"]
        assert env.config["high_speed_reward"] == config["high_speed_reward"]

        env.close()
        print("✓ Config compatibility test passed")
        return True

    except Exception as e:
        print(f"✗ Config compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running environment initialization tests...\n")

    try:
        success = True
        success &= test_env_initialization()
        success &= test_reward_calculation()
        success &= test_config_compatibility()

        if success:
            print("\n[CELEBRATE] All environment tests passed! Urban junction environment is working correctly.")
            sys.exit(0)
        else:
            print("\n[ERROR] Some environment tests failed.")
            sys.exit(1)

    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
