# train_multi_scenario.py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from multi_scenario_env import MultiScenarioHighwayEnv


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


class AggressionCurriculumCallback(BaseCallback):
    """
    Callback that gradually increases env aggressiveness over training.
    We call env.set_curriculum_progress(progress) where progress ∈ [0, 1].
    """

    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # Fraction of training completed in [0, 1]
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

        # Update all underlying MultiScenarioHighwayEnv instances
        # self.training_env is a VecEnv (DummyVecEnv)
        try:
            for venv_env in self.training_env.envs:
                # venv_env is Monitor(MultiScenarioHighwayEnv)
                base_env = getattr(venv_env, "env", venv_env)
                if hasattr(base_env, "set_curriculum_progress"):
                    base_env.set_curriculum_progress(target_aggr)
        except Exception:
            # If something goes weird, don't crash training
            pass

        if self.verbose > 0 and self.num_timesteps % 5000 == 0:
            print(
                f"[Curriculum] steps={self.num_timesteps} "
                f"progress={progress:.2f} aggressiveness={target_aggr:.2f}"
            )

        return True


def main(
    total_timesteps: int = 500_000,
    model_path: str = "outputs/models/multi_scenario_lidar_ppo_500k.zip",
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

    # Curriculum callback to ramp traffic aggressiveness
    callback = AggressionCurriculumCallback(
        total_timesteps=total_timesteps,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(model_path)
    env.close()

    print(f"[DONE] Trained PPO on multi-scenario env for {total_timesteps} steps.")
    print(f"[SAVE] Model saved to {model_path}")


if __name__ == "__main__":
    main()
