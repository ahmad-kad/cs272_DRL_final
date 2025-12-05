# evaluate_multi.py

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from stable_baselines3 import PPO

from multi_scenario_env import MultiScenarioHighwayEnv  # custom env


def make_multi_eval_env(
    aggressiveness: float = 1.0,
):
    """
    Build the multi-scenario env for evaluation.
    Use same obs config as training; fix aggressiveness (e.g. hard = 1.0).
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
        aggressiveness=aggressiveness,
    )
    return env


def evaluate_multi_and_violin(
    model_path: str,
    n_episodes: int = 500,
    out_dir: str = "outputs/plots",
    plot_id: str = "ID14",
    aggressiveness: float = 1.0,
):
    env = make_multi_eval_env(aggressiveness=aggressiveness)
    model = PPO.load(model_path)

    returns = []
    lengths = []

    for ep in tqdm(range(n_episodes), desc="Evaluating multi-scenario custom env"):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_ret = 0.0
        ep_len = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_ret += reward
            ep_len += 1

        returns.append(ep_ret)
        lengths.append(ep_len)

    env.close()

    returns = np.array(returns, dtype=float)
    lengths = np.array(lengths, dtype=float)

    print(
        f"[EVAL] multi-scenario custom env, "
        f"episodes={n_episodes}, "
        f"mean_return={returns.mean():.2f} ± {returns.std():.2f}"
    )

    os.makedirs(out_dir, exist_ok=True)

    # Violin plot of overall performance on the custom env
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.violinplot(returns, showmeans=True, showmedians=True)
    ax.set_title(f"Custom multi-scenario – Lidar (n={n_episodes})")
    ax.set_ylabel("Episode return")
    ax.set_xticks([1])
    ax.set_xticklabels(["return"])

    filename = f"{plot_id}_multi_custom_lidar_violin.png"
    path = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"[SAVE] Violin plot saved to {path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate multi-scenario custom env (500 episodes, violin plot)."
    )
    p.add_argument("--model", required=True, help="Path to trained multi-scenario model .zip")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--plot_id", type=str, default="ID14")  # matches your table
    p.add_argument("--aggressiveness", type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_multi_and_violin(
        model_path=args.model,
        n_episodes=args.episodes,
        out_dir="outputs/plots",
        plot_id=args.plot_id,
        aggressiveness=args.aggressiveness,
    )
