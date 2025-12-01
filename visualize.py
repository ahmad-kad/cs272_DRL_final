# visualize.py

import argparse
import os
import time

import gymnasium as gym
import highway_env  # noqa: F401  # register envs
import imageio.v2 as imageio
from stable_baselines3 import PPO

# Reuse the same mapping as train.py
ENV_ID_MAP = {
    "highway": "highway-v0",
    "merge": "merge-v0",
    "intersection": "intersection-v0",
}


def make_observation_config(obs_type: str) -> dict:
    """Must match what you used in train.py."""
    obs_type = obs_type.lower()
    if obs_type == "lidar":
        return {
            "type": "LidarObservation",
            "cells": 32,
            "maximum_range": 60,
            "normalize": True,
        }
    elif obs_type == "grayscale":
        return {
            "type": "GrayscaleObservation",
            "weights": [0.2989, 0.5870, 0.1140],
            "stack_size": 4,
            "observation_shape": (84, 84),
        }
    else:
        raise ValueError(f"Unknown obs_type: {obs_type} (use 'lidar' or 'grayscale')")


def make_env(env_id: str, obs_type: str, render_mode: str = "rgb_array"):
    """
    Build one env for visualization.

    render_mode="rgb_array" → env.render() returns RGB frames (good for GIFs).
    If you want a visible window instead, you can change to "human".
    """
    config = {
        "observation": make_observation_config(obs_type),
        "offscreen_rendering": render_mode == "rgb_array",
    }

    env = gym.make(
        env_id,
        render_mode=render_mode,
        config=config,
    )
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a trained PPO agent on a single highway-env scenario."
    )
    parser.add_argument(
        "scenario",
        type=str,
        choices=list(ENV_ID_MAP.keys()),
        help="Which environment: highway | merge | intersection",
    )
    parser.add_argument(
        "obs_type",
        type=str,
        choices=["lidar", "grayscale"],
        help="Observation type: lidar | grayscale",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model .zip file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to visualize (default: 5)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Delay between steps in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--save_gif",
        action="store_true",
        help="If set, save each episode as a GIF in outputs/visualization",
    )
    parser.add_argument(
        "--gif_dir",
        type=str,
        default="outputs/visualization",
        help="Directory to save GIFs (default: outputs/visualization)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Max steps per episode during visualization/GIF (default: 200)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    env_key = args.scenario.lower()
    env_id = ENV_ID_MAP[env_key]
    obs_type = args.obs_type.lower()

    # For GIFs we want rgb_array; for just a live viewer, you can switch to "human"
    env = make_env(env_id, obs_type, render_mode="rgb_array")
    model = PPO.load(args.model)
    print(f"[LOAD] Model loaded from {args.model}")
    print(f"[ENV]  {env_id} with {obs_type} observations")

    if args.save_gif:
        os.makedirs(args.gif_dir, exist_ok=True)

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        steps = 0
        frames = []

        print(f"\n[EPISODE {ep}] Starting...")

        while not (done or truncated) and steps < args.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1

            frame = env.render()  # numpy array (because render_mode="rgb_array")
            frames.append(frame)

            time.sleep(args.delay)

        print(f"Episode {ep}: steps={steps}, reward={ep_reward:.2f}")

        if args.save_gif:
            scenario_name = env_id.replace("-v0", "")
            gif_name = f"{scenario_name}_{obs_type}_ep{ep}.gif"
            gif_path = os.path.join(args.gif_dir, gif_name)
            fps = int(1.0 / args.delay) if args.delay > 0 else 20
            imageio.mimsave(gif_path, frames, fps=fps)
            print(f"[GIF] Saved {gif_path}")

    env.close()


if __name__ == "__main__":
    main()
