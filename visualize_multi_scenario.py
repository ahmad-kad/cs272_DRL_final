import time
from stable_baselines3 import PPO
from multi_scenario_env import MultiScenarioHighwayEnv


def main(model_path="outputs/models/multi_scenario_lidar_ppo_500k.zip",
         episodes: int = 5,
         delay: float = 0.05):

    env = MultiScenarioHighwayEnv(
        env_ids=["highway-v0", "merge-v0", "intersection-v0"],
        observation_config={
            "type": "LidarObservation",
            "cells": 32,
            "maximum_range": 60,
            "normalize": True,
        },
        render_mode="rgb_array",
        aggressiveness=1.0,  # <--- match “hard” end of curriculum
    )

    model = PPO.load(model_path)
    print(f"[LOAD] Model loaded from {model_path}")

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        scenario = info.get("scenario", "unknown")
        done = False
        truncated = False
        ep_reward = 0.0
        steps = 0

        print(f"\n[EPISODE {ep}] Scenario: {scenario}")

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1

            env.render()
            time.sleep(delay)

        print(f"Episode {ep} finished in scenario {scenario}: steps={steps}, reward={ep_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
