# train_multi_scenario.py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from multi_scenario_env import MultiScenarioHighwayEnv
from multi_wandb_helpers import make_wandb_and_curriculum_callbacks

import torch as th
from stable_baselines3.common.vec_env import SubprocVecEnv


def make_multi_env():
    """
    Factory that creates ONE instance of the multi-scenario env,
    then wraps it in a Monitor so we log episode rewards/lengths.
    """
    env = MultiScenarioHighwayEnv(
        env_ids=["highway-v0", "merge-v0", "intersection-v0"],
        observation_config={
            "type": "LidarObservation",
            "cells": 32,
            "maximum_range": 60,
            "normalize": True,
        },
        render_mode=None,
        aggressiveness=0.0,  # start easy; curriculum will ramp this up
    )
    return Monitor(env)

def main(
    total_timesteps: int = 500_000,
    model_path: str = "outputs/models/multi_scenario_lidar_500k.zip",
    n_envs: int = 4,
):
    # ----- Device selection (GPU if available) -----
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"[DEVICE] Using {device}")

    # ----- Parallel envs (true parallel with subprocesses) -----
    # We pass a list of callables; each creates its own env instance in a child process
    env_fns = [make_multi_env for _ in range(n_envs)]
    env = SubprocVecEnv(env_fns)

    # ----- PPO agent -----
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-4,
        ent_coef=0.05,
        n_steps=2048,
        batch_size=128,
        gamma=0.95,
        tensorboard_log="./logs/tb_multi_scenario/",
        device=device,  # <-- send networks to GPU if available
    )

    # ----- W&B + curriculum callbacks -----
    callbacks = make_wandb_and_curriculum_callbacks(
        total_timesteps=total_timesteps,
        project="multi-scenario-training",
        run_name=f"multi-lidar-{total_timesteps//1000}k",
        verbose=1,
    )

    print(
        f"[TRAIN] multi-scenario, obs=lidar, policy=MlpPolicy, "
        f"steps={total_timesteps}, n_envs={n_envs}"
    )

    # progress_bar=True gives you the same nice ETA bar as single-scenario training
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(model_path)
    env.close()

    print(f"[DONE] Trained PPO on multi-scenario env for {total_timesteps} steps.")
    print(f"[SAVE] Model saved to {model_path}")
