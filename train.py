# train.py

import argparse
import os

import gymnasium as gym
import highway_env  # noqa: F401  # register environments
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from wandb_single_helpers import make_wandb_single_callback

# Same mapping as in visualize.py
ENV_ID_MAP = {
    "highway": "highway-v0",
    "merge": "merge-v0",
    "intersection": "intersection-v0",
}


def make_observation_config(obs_type: str) -> dict:
    """
    MUST match what you use in visualize.py so models are compatible.
    """
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
        raise ValueError(f"Unknown obs_type: {obs_type} (use 'lidar' or 'grayscale').")


def make_env(env_id: str, obs_type: str, render_mode=None):
    """
    Build a single highway-env environment with the requested observation.
    Wrapped in Monitor when used for training (render_mode=None).
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

    if render_mode is None:
        # For training: wrap in Monitor so infos contain 'episode'
        env = Monitor(env)

    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on a single highway-env scenario "
                    "with Lidar or Grayscale observations."
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
        "--steps",
        type=int,
        default=100_000,
        help="Total training timesteps (default: 100k)",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=4,
        help="Number of parallel envs for training (default: 4)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="If set, do not use Weights & Biases logging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    scenario_key = args.scenario.lower()
    obs_type = args.obs_type.lower()
    total_timesteps = args.steps

    env_id = ENV_ID_MAP[scenario_key]

    os.makedirs("outputs/models", exist_ok=True)

    # Choose policy based on observation type
    policy = "MlpPolicy" if obs_type == "lidar" else "CnnPolicy"

    # Build VecEnv
    def make_thunk():
        return make_env(env_id, obs_type, render_mode=None)

    env = DummyVecEnv([make_thunk for _ in range(args.n_envs)])

    # Model save path
    model_path = f"outputs/models/{env_id}_{obs_type}_{total_timesteps//1000}k.zip"

    model = PPO(
        policy,
        env,
        verbose=1,
        learning_rate=5e-4,
        ent_coef=0.05,
        n_steps=2048,
        batch_size=128,
        gamma=0.95,
        tensorboard_log=f"./logs/tb_{env_id}_{obs_type}/",
    )

    # W&B callback (optional)
    if args.no_wandb:
        callbacks = None
        print("[INFO] W&B logging disabled (--no_wandb set).")
    else:
        callbacks = make_wandb_single_callback(
            total_timesteps=total_timesteps,
            env_id=env_id,
            obs_type=obs_type,
            project="single-scenario-training",
            run_name=f"{env_id}-{obs_type}-{total_timesteps//1000}k",
            verbose=1,
        )

    print(
        f"[TRAIN] env={env_id}, obs={obs_type}, policy={policy}, "
        f"steps={total_timesteps}, n_envs={args.n_envs}"
    )

    if callbacks is not None:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
    else:
        model.learn(total_timesteps=total_timesteps)

    model.save(model_path)
    env.close()
    print(f"[SAVE] Model saved to {model_path}")
    print("[DONE] Training finished.")


if __name__ == "__main__":
    main()
