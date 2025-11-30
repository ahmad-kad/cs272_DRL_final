import gymnasium as gym
import highway_env  # noqa: F401, needed to register envs
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

def make_merge_env():
    """
    Create the built-in highway-env merge-v0 environment.
    This uses the Lidar observation and built-in reward.
    """
    # Optional: tweak config here if you want (lanes_count, vehicles_count, etc.)
    config = {
        "observation": {"type": "LidarObservation", "cells": 32,
                        "maximum_range": 60, "normalize": True},
        "lanes_count": 3,          # default is 2
        "vehicles_count": 25,      # more traffic → must change lanes
        "duration": 60,            # seconds
        "collision_reward": -5,
        "high_speed_reward": 0.5,
        "lane_change_reward": 0.2, # encourage lateral moves
        "offscreen_rendering": True,
    }

    env = gym.make(
        "merge-v0",
        render_mode=None,                # or "rgb_array" if you want to render
        config=config,
    )
    env = Monitor(env)                  # record ep_len / ep_rew etc.
    return env

def main(total_timesteps=250_000, model_path="merge_v0_ppo_lidar_better.zip"):
    # 4 parallel environments for better exploration
    env = DummyVecEnv([make_merge_env for _ in range(4)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-4,   # was 3e-4
        ent_coef=0.05,        # was 0.0
        n_steps=2048,
        batch_size=128,
        gamma=0.95,
        tensorboard_log="./tb_merge_v0/",
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)

if __name__ == "__main__":
    # You can change this or later add argparse
    main(total_timesteps=100_000, model_path="outputs/models/merge_v0_ppo_100k.zip")
