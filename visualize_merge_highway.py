import time
import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import PPO


def main(model_path="outputs/models/merge_v0_ppo_200k.zip", episodes=5, delay=0.05):
    # Use same config as training (if you changed to LidarObservation, copy it here)
    config = {
        "observation": {
            "type": "LidarObservation",
            "cells": 32,
            "maximum_range": 60,
            "normalize": True,
            # if you specified "features" during training, include them here too
            # "features": ["presence", "vx"],  # example
        },
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "offscreen_rendering": False,
    }

    env = gym.make(
        "merge-v0",
        render_mode="rgb_array",
        config=config,
    )


    model = PPO.load(model_path)
    print(f"[LOAD] Model loaded from {model_path}")

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        steps = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1

            env.render()
            time.sleep(delay)

        print(f"Episode {ep}: steps={steps}, reward={ep_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
