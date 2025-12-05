# evaluate_single.py
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import PPO

from train import ENV_ID_MAP, make_observation_config  # reuse your helpers

from tqdm import tqdm


def make_eval_env(env_id: str, obs_type: str):
    """
    Build a single env for evaluation. Must match training config
    (same observation type, duration, etc.).
    """
    config = {
        "observation": make_observation_config(obs_type),
        "duration": 30,              # whatever you used for training
        "offscreen_rendering": True, # no interactive window needed
    }

    env = gym.make(env_id, render_mode=None, config=config)
    return env


def evaluate_and_violin(
    scenario: str,
    obs_type: str,
    model_path: str,
    n_episodes: int = 500,
    out_dir: str = "outputs/plots",
    plot_id: str = "2",
):
    env_id = ENV_ID_MAP[scenario]
    env = make_eval_env(env_id, obs_type)
    model = PPO.load(model_path)

    returns = []
    lengths = []

    # tqdm progress bar over episodes
    for ep in tqdm(range(n_episodes), desc=f"Evaluating {env_id} ({obs_type})"):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_ret = 0.0
        ep_len = 0

        # deterministic=True -> NO exploration noise
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
        f"[EVAL] {env_id} ({obs_type}), "
        f"episodes={n_episodes}, "
        f"mean_return={returns.mean():.2f} ± {returns.std():.2f}"
    )

    os.makedirs(out_dir, exist_ok=True)

    # ---- violin plot ----
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.violinplot(returns, showmeans=True, showmedians=True)
    ax.set_title(f"{env_id} – {obs_type} (n={n_episodes})")
    ax.set_ylabel("Episode return")
    ax.set_xticks([1])
    ax.set_xticklabels(["return"])

    filename = f"{plot_id}_{env_id}_{obs_type}_violin.png"
    path = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"[SAVE] Violin plot saved to {path}")



def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a trained model for 500 episodes and make a violin plot."
    )
    p.add_argument("scenario", choices=list(ENV_ID_MAP.keys()))
    p.add_argument("obs_type", choices=["lidar", "grayscale"])
    p.add_argument("--model", required=True, help="Path to trained model .zip")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--plot_id", type=str, default="2")  # e.g. 1..14
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_and_violin(
        scenario=args.scenario,
        obs_type=args.obs_type,
        model_path=args.model,
        n_episodes=args.episodes,
        plot_id=args.plot_id,
    )
