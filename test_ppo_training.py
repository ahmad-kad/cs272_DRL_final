#!/usr/bin/env python3
"""
Simple test to see if PPO training works at all.
"""

import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import time

def test_basic_ppo():
    """Test basic PPO training on highway environment."""
    print("Testing basic PPO training...")

    # Create single environment
    env = gym.make('highway-v0', render_mode=None)
    env = DummyVecEnv([lambda: env])

    # Create simple PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        verbose=1,
        device='cpu'
    )

    print("Starting training for 1000 timesteps...")
    start_time = time.time()

    # Train for just 1000 timesteps
    model.learn(total_timesteps=1000)

    elapsed = time.time() - start_time
    print(".2f")

    return elapsed

if __name__ == "__main__":
    elapsed = test_basic_ppo()
    print(f"Training took {elapsed:.2f} seconds - should be > 1 second if working properly")


