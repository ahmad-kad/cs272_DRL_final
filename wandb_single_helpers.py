# wandb_single_helpers.py

from typing import Dict

import wandb
from stable_baselines3.common.callbacks import BaseCallback, CallbackList


class WandbSingleEnvCallback(BaseCallback):
    """
    Generic W&B logger for a single highway-env environment.

    Assumes:
      - Env is wrapped in Monitor so infos contain "episode" when an ep ends.
      - info.get("crashed", False) is True if episode ended with a crash
        (highway-env does that for most envs).
    """

    def __init__(
        self,
        project: str,
        run_name: str,
        env_id: str,
        obs_type: str,
        total_timesteps: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.project = project
        self.run_name = run_name
        self.env_id = env_id
        self.obs_type = obs_type
        self.total_timesteps = total_timesteps

        self.ep_returns = 0.0
        self.ep_lengths = 0
        self.n_episodes = 0
        self.n_success = 0

    def _on_training_start(self) -> None:
        wandb.init(
            project=self.project,
            name=self.run_name,
            config={
                "algo": "PPO",
                "env_id": self.env_id,
                "obs_type": self.obs_type,
                "total_timesteps": self.total_timesteps,
            },
        )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if info is None:
                continue
            ep_info = info.get("episode")
            if ep_info is None:
                continue

            r = float(ep_info.get("r", 0.0))
            l = int(ep_info.get("l", 0))
            crashed = bool(info.get("crashed", False))
            success = 0 if crashed else 1

            self.ep_returns += r
            self.ep_lengths += l
            self.n_episodes += 1
            self.n_success += success

            wandb.log(
                {
                    "episode_return": r,
                    "episode_length": l,
                    "success": success,
                    "global/num_timesteps": self.num_timesteps,
                    "global/learning_rate": float(
                        self.model.lr_schedule(self.model._current_progress_remaining)
                    ),
                },
                step=self.num_timesteps,
            )

        return True

    def _on_training_end(self) -> None:
        if self.n_episodes > 0:
            mean_ret = self.ep_returns / self.n_episodes
            mean_len = self.ep_lengths / self.n_episodes
            success_rate = self.n_success / self.n_episodes
        else:
            mean_ret = mean_len = success_rate = 0.0

        wandb.log(
            {
                "summary/mean_return": mean_ret,
                "summary/mean_length": mean_len,
                "summary/success_rate": success_rate,
            },
            step=self.num_timesteps,
        )
        wandb.finish()


def make_wandb_single_callback(
    total_timesteps: int,
    env_id: str,
    obs_type: str,
    project: str = "autonomous-driving-single-env",
    run_name: str | None = None,
    verbose: int = 0,
) -> CallbackList:
    """
    Small helper so train.py can just call this and get a callback list.
    """
    if run_name is None:
        run_name = f"{env_id}-{obs_type}-{total_timesteps//1000}k"

    cb = WandbSingleEnvCallback(
        project=project,
        run_name=run_name,
        env_id=env_id,
        obs_type=obs_type,
        total_timesteps=total_timesteps,
        verbose=verbose,
    )
    return CallbackList([cb])
