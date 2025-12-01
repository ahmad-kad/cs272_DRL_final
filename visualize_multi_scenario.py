# visualize_multi_scenario.py

import os
import time

import imageio.v2 as imageio      # <-- for saving GIFs
from stable_baselines3 import PPO
from multi_scenario_env import MultiScenarioHighwayEnv


def main(
    model_path: str = "outputs/models/multi_scenario_lidar_ppo_750k.zip",
    episodes: int = 5,
    delay: float = 0.05,
):
    # where to save GIFs
    out_dir = "outputs/visualization/multi_scenario"
    os.makedirs(out_dir, exist_ok=True)

    env = MultiScenarioHighwayEnv(
        env_ids=["highway-v0", "merge-v0", "intersection-v0"],
        observation_config={
            "type": "LidarObservation",
            "cells": 32,
            "maximum_range": 60,
            "normalize": True,
        },
        render_mode="rgb_array",   # important: returns RGB arrays
        aggressiveness=1.0,        # match “hard” end of curriculum
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

        frames = []  # store RGB frames for this episode

        print(f"\n[EPISODE {ep}] Scenario: {scenario}")

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1

            # grab frame as numpy array
            frame = env.render()   # because render_mode="rgb_array"
            frames.append(frame)

            time.sleep(delay)

        print(
            f"Episode {ep} finished in scenario {scenario}: "
            f"steps={steps}, reward={ep_reward:.2f}"
        )

        # -------- Save GIF for this episode --------
        # fps ~ 1/delay, but clamp to something reasonable
        fps = int(1.0 / delay) if delay > 0 else 20
        gif_name = f"multi_scenario_ep{ep}_{scenario}.gif"
        gif_path = os.path.join(out_dir, gif_name)

        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"[GIF] Saved {gif_path}")

    env.close()


if __name__ == "__main__":
    main()
