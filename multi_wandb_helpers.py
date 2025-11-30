# multi_wandb_helpers.py

from typing import Dict

import wandb
from stable_baselines3.common.callbacks import BaseCallback, CallbackList


class AggressionCurriculumCallback(BaseCallback):
    """
    Ramps MultiScenarioHighwayEnv aggressiveness over training.

    Assumes each underlying env in the VecEnv has:
        env.set_curriculum_progress(progress: float in [0, 1])
    """

    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # Fraction of training completed
        progress = self.num_timesteps / float(self.total_timesteps)

        # Example schedule:
        #  0–10%: 0.0 → 0.2
        # 10–50%: 0.2 → 0.7
        # 50–100%: 0.7 → 1.0
        if progress < 0.1:
            target_aggr = 0.2 * (progress / 0.1)
        elif progress < 0.5:
            target_aggr = 0.2 + 0.5 * ((progress - 0.1) / 0.4)
        else:
            target_aggr = 0.7 + 0.3 * ((progress - 0.5) / 0.5)

        # Push the new aggressiveness into each env in the VecEnv
        try:
            for venv_env in self.training_env.envs:
                # venv_env is usually Monitor(MultiScenarioHighwayEnv)
                base_env = getattr(venv_env, "env", venv_env)
                if hasattr(base_env, "set_curriculum_progress"):
                    base_env.set_curriculum_progress(target_aggr)
        except Exception:
            # Don't crash training on weird edge cases
            pass

        if self.verbose > 0 and self.num_timesteps % 5000 == 0:
            print(
                f"[Curriculum] steps={self.num_timesteps} "
                f"progress={progress:.2f} aggressiveness={target_aggr:.2f}"
            )

        return True


class WandbScenarioCallback(BaseCallback):
    """
    Logs per-scenario metrics (reward, length, success) to Weights & Biases.

    Requirements:
      - You wrapped the env in Monitor(env) so 'episode' info is added to infos.
      - MultiScenarioHighwayEnv adds 'scenario' (string) into info.
      - Optional: info['crashed'] (bool) to log success vs failure.
    """

    def __init__(
        self,
        project: str,
        run_name: str,
        total_timesteps: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.project = project
        self.run_name = run_name
        self.total_timesteps = total_timesteps

        # episode-level accumulators per scenario
        self.scenario_returns: Dict[str, float] = {}
        self.scenario_lengths: Dict[str, int] = {}
        self.scenario_episodes: Dict[str, int] = {}
        self.scenario_successes: Dict[str, int] = {}

    def _on_training_start(self) -> None:
        wandb.init(
            project=self.project,
            name=self.run_name,
            config={
                "algo": "PPO",
                "total_timesteps": self.total_timesteps,
                "env": "MultiScenarioHighwayEnv",
                "obs_type": "Lidar",
            },
        )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        # Each info corresponds to one env in the VecEnv
        for info in infos:
            if info is None:
                continue

            # Monitor adds 'episode' when an episode terminates in that env
            ep_info = info.get("episode")
            if ep_info is None:
                continue

            scenario = info.get("scenario", "unknown")
            reward = float(ep_info.get("r", 0.0))
            length = int(ep_info.get("l", 0))
            crashed = bool(info.get("crashed", False))
            success = 0 if crashed else 1

            # Init counters if new scenario
            if scenario not in self.scenario_returns:
                self.scenario_returns[scenario] = 0.0
                self.scenario_lengths[scenario] = 0
                self.scenario_episodes[scenario] = 0
                self.scenario_successes[scenario] = 0

            self.scenario_returns[scenario] += reward
            self.scenario_lengths[scenario] += length
            self.scenario_episodes[scenario] += 1
            self.scenario_successes[scenario] += success

            # Log the raw episode metrics
            wandb.log(
                {
                    f"{scenario}/episode_return": reward,
                    f"{scenario}/episode_length": length,
                    f"{scenario}/success": success,
                    "global/num_timesteps": self.num_timesteps,
                    "global/learning_rate": float(
                        self.model.lr_schedule(self.model._current_progress_remaining)
                    ),
                },
                step=self.num_timesteps,
            )

        return True

    def _on_training_end(self) -> None:
        # Final summary per scenario
        summary = {}
        for scenario, n_ep in self.scenario_episodes.items():
            if n_ep == 0:
                continue
            mean_ret = self.scenario_returns[scenario] / n_ep
            mean_len = self.scenario_lengths[scenario] / n_ep
            success_rate = self.scenario_successes[scenario] / n_ep

            summary[f"{scenario}/mean_return"] = mean_ret
            summary[f"{scenario}/mean_length"] = mean_len
            summary[f"{scenario}/success_rate"] = success_rate

        wandb.log(summary, step=self.num_timesteps)
        wandb.finish()


def make_wandb_and_curriculum_callbacks(
    total_timesteps: int,
    project: str,
    run_name: str,
    verbose: int = 0,
) -> CallbackList:
    """
    Convenience helper: returns a CallbackList with:
      - AggressionCurriculumCallback
      - WandbScenarioCallback
    """
    curriculum_cb = AggressionCurriculumCallback(
        total_timesteps=total_timesteps,
        verbose=verbose,
    )

    wandb_cb = WandbScenarioCallback(
        project=project,
        run_name=run_name,
        total_timesteps=total_timesteps,
        verbose=verbose,
    )

    return CallbackList([curriculum_cb, wandb_cb])
