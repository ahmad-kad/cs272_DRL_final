# visualize_copchase.py

import argparse
import os
import time

import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import PPO
import numpy as np

# Import the custom environment
from team2_env.crazy_driver_enviornment import crazy_driver_env


def make_env(render_mode: str = "rgb_array"):
    """
    Create the CopChase environment for visualization.

    render_mode="rgb_array" → env.render() returns RGB frames (good for GIFs).
    If you want a visible window instead, you can change to "human".
    """
    config = crazy_driver_env.default_config()

    # Visualization-specific config overrides
    config.update({
        "offscreen_rendering": render_mode == "rgb_array",
        "render_agent": True,  # Show the agent
        "show_trajectories": False,  # Don't show trajectories for cleaner visualization
    })

    env = gym.make(
        "CopChase-v0",
        render_mode=render_mode,
        config=config,
    )
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the CopChase environment (crazy driver scenario)."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to a trained PPO model .zip file (optional - if not provided, uses random actions)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to visualize (default: 3)",
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
    parser.add_argument(
        "--render_mode",
        type=str,
        choices=["human", "rgb_array"],
        default="rgb_array",
        help="Render mode: 'human' for live window, 'rgb_array' for GIFs (default: rgb_array)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create environment
    env = make_env(render_mode=args.render_mode)

    # Load model if provided, otherwise use random actions
    if args.model:
        model = PPO.load(args.model)
        print(f"[LOAD] Model loaded from {args.model}")
        use_model = True
    else:
        model = None
        print("[INFO] No model provided - using random actions")
        use_model = False

    print(f"[ENV] CopChase-v0 environment")
    print(f"[ENV] Action space: {env.action_space}")
    print(f"[ENV] Observation space: {env.observation_space}")
    config = env.unwrapped.config
    print(f"[ENV] Duration: {config['duration']} seconds")
    print(f"[ENV] Vehicles: {config['vehicles_count']} NPCs + {config['cop_count']} cops")

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
            if use_model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Random action: sample from action space
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1

            if args.render_mode == "rgb_array":
                frame = env.render()  # numpy array
                frames.append(frame)

            if args.render_mode == "human":
                time.sleep(args.delay)

        print(f"Episode {ep}: steps={steps}, reward={ep_reward:.2f}, crashed={info.get('crashed', False)}")

        if args.save_gif and args.render_mode == "rgb_array":
            gif_name = f"copchase_ep{ep}.gif"
            if use_model:
                gif_name = f"copchase_trained_ep{ep}.gif"
            else:
                gif_name = f"copchase_random_ep{ep}.gif"

            gif_path = os.path.join(args.gif_dir, gif_name)
            duration = int(1000 * args.delay) if args.delay > 0 else 50  # duration in ms
            imageio.mimsave(gif_path, frames, duration=duration)
            print(f"[GIF] Saved {gif_path}")

    env.close()


if __name__ == "__main__":
    main()
