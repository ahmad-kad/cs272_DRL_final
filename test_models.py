#!/usr/bin/env python3
"""
Simple test script to check model observation spaces and basic functionality.
"""

import numpy as np
from stable_baselines3 import PPO
from environments.urban_junction_env import UrbanJunctionEnv

def test_model_loading():
    """Test loading models and check their observation spaces."""

    print("Testing model loading and observation spaces...")

    # Load models
    lidar_model_path = "outputs/models/adaptive_lidar_final.zip"
    grayscale_model_path = "outputs/models/adaptive_grayscale_final.zip"

    try:
        lidar_model = PPO.load(lidar_model_path)
        print(f"[OK] Lidar model loaded successfully")
        print(f"  Observation space: {lidar_model.observation_space}")
        print(f"  Action space: {lidar_model.action_space}")
    except Exception as e:
        print(f"[ERROR] Failed to load lidar model: {e}")
        return

    try:
        grayscale_model = PPO.load(grayscale_model_path)
        print(f"[OK] Grayscale model loaded successfully")
        print(f"  Observation space: {grayscale_model.observation_space}")
        print(f"  Action space: {grayscale_model.action_space}")
    except Exception as e:
        print(f"[ERROR] Failed to load grayscale model: {e}")
        return

    # Test environments
    print("\nTesting environments...")

    for modality in ["lidar", "grayscale"]:
        try:
            env = UrbanJunctionEnv(scenario="highway", modality=modality)
            obs, info = env.reset()
            print(f"[OK] {modality} environment created")
            print(f"  Observation shape: {obs.shape}")
            print(f"  Observation space: {env.observation_space}")

            # Test model prediction
            if modality == "lidar":
                action, _ = lidar_model.predict(obs, deterministic=True)
                print(f"  [OK] Lidar model prediction: {action}")
            else:
                action, _ = grayscale_model.predict(obs, deterministic=True)
                print(f"  [OK] Grayscale model prediction: {action}")

            env.close()

        except Exception as e:
            print(f"[ERROR] Failed {modality} environment test: {e}")

if __name__ == "__main__":
    test_model_loading()
