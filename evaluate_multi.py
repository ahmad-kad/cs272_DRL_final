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

    # overall stats
    returns = []
    lengths = []

    # per-scenario stats
    scenario_returns = {}
    scenario_lengths = {}

    for ep in tqdm(range(n_episodes), desc="Evaluating multi-scenario custom env"):
        obs, info = env.reset()
        scenario = info.get("scenario", "unknown")  # <- which env this episode uses

        done = False
        truncated = False
        ep_ret = 0.0
        ep_len = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_ret += reward
            ep_len += 1

        # overall
        returns.append(ep_ret)
        lengths.append(ep_len)

        # per-scenario
        if scenario not in scenario_returns:
            scenario_returns[scenario] = []
            scenario_lengths[scenario] = []
        scenario_returns[scenario].append(ep_ret)
        scenario_lengths[scenario].append(ep_len)

    env.close()

    returns = np.array(returns, dtype=float)
    lengths = np.array(lengths, dtype=float)

    print(
        f"[EVAL] multi-scenario custom env, "
        f"episodes={n_episodes}, "
        f"mean_return={returns.mean():.2f} ± {returns.std():.2f}"
    )

    os.makedirs(out_dir, exist_ok=True)

    # ---- plotting: overall + per-scenario violins ----
    scenarios = sorted(scenario_returns.keys())
    n_plots = 1 + len(scenarios)  # overall + each env

    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), sharey=True)

    # 0) overall violin (what you already had)
    axes[0].violinplot(returns, showmeans=True, showmedians=True)
    axes[0].set_title(f"All scenarios (n={n_episodes})")
    axes[0].set_ylabel("Episode return")
    axes[0].set_xticks([1])
    axes[0].set_xticklabels(["return"])

    # 1..N) one violin per scenario
    for i, sc in enumerate(scenarios, start=1):
        sc_rets = np.array(scenario_returns[sc], dtype=float)
        axes[i].violinplot(sc_rets, showmeans=True, showmedians=True)
        axes[i].set_title(f"{sc} (n={len(sc_rets)})")
        axes[i].set_xticks([1])
        axes[i].set_xticklabels(["return"])

    fig.suptitle(f"Custom multi-scenario – Lidar (n={n_episodes})", y=1.05)
    fig.tight_layout()

    filename = f"{plot_id}_multi_custom_lidar_violin.png"
    path = os.path.join(out_dir, filename)
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
