# train_multi_scenario.py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from multi_scenario_env import MultiScenarioHighwayEnv

from multi_wandb_helpers import make_wandb_and_curriculum_callbacks

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
    total_timesteps: int = 25_000,
    model_path: str = "outputs/models/multi_scenario_lidar_ppo_25k.zip",
):
    # 4 parallel copies of the multi-scenario env
    env = DummyVecEnv([make_multi_env for _ in range(4)])

    # PPO agent (tuned a bit for highway-style tasks)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-4,
        ent_coef=0.05,
        n_steps=2048,
        batch_size=128,
        gamma=0.95,
        tensorboard_log="./tb_multi_scenario/",
    )

    # Combined curriculum + W&B logging callbacks from helper file
    callbacks = make_wandb_and_curriculum_callbacks(
        total_timesteps=total_timesteps,
        project="multi-scenario-training",
        run_name=f"multi-lidar-{total_timesteps//1000}k",
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(model_path)
    env.close()

    print(f"[DONE] Trained PPO on multi-scenario env for {total_timesteps} steps.")
    print(f"[SAVE] Model saved to {model_path}")


if __name__ == "__main__":
    main()
