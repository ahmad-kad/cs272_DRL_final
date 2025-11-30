# multi_scenario_env.py

import random
from typing import List, Optional, Tuple, Dict, Any

import gymnasium as gym
import highway_env  # noqa: F401  # needed to register highway-env environments
from gymnasium import spaces


class MultiScenarioHighwayEnv(gym.Env):
    """
    Custom environment that randomly switches between:
      - highway-v0
      - merge-v0
      - intersection-v0

    Each episode:
      * picks one scenario at random,
      * configures traffic based on a shared 'aggressiveness' parameter,
      * uses a common observation config (e.g. Lidar),
      * exposes a single observation_space/action_space to the agent.

    The 'aggressiveness' parameter (in [0, 1]) can be used for curriculum:
      - 0.0 → easy traffic
      - 1.0 → dense / antagonistic traffic
    """

    metadata = {"render_modes": ["rgb_array", "human", None]}

    def __init__(
        self,
        env_ids: Optional[List[str]] = None,
        observation_config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        aggressiveness: float = 0.0,
    ):
        super().__init__()

        # Which base scenarios to include
        if env_ids is None:
            env_ids = ["highway-v0", "merge-v0", "intersection-v0"]
        self.env_ids = env_ids
        self.render_mode = render_mode
        self._seed = seed

        # Curriculum / difficulty knob: 0.0 = easy, 1.0 = hardest
        self.aggressiveness: float = float(max(0.0, min(1.0, aggressiveness)))

        # Default observation: LidarObservation shared across all scenarios
        if observation_config is None:
            observation_config = {
                "type": "LidarObservation",
                "cells": 32,
                "maximum_range": 60,
                "normalize": True,
            }
        self.observation_config = observation_config

        # Create one instance of each underlying env upfront
        self._envs: Dict[str, gym.Env] = {}
        for eid in self.env_ids:
            base_config = {
                "observation": self.observation_config,
                "simulation_frequency": 15,
                "policy_frequency": 5,
                # We'll override vehicles_count etc. *per reset* based on aggressiveness
            }
            env = gym.make(eid, config=base_config, render_mode=self.render_mode)
            if seed is not None:
                env.reset(seed=seed)
            self._envs[eid] = env

        # Assume all scenarios share the same obs/action spaces (true for default configs)
        first_env = self._envs[self.env_ids[0]]
        self.observation_space: spaces.Space = first_env.observation_space
        self.action_space: spaces.Space = first_env.action_space

        self.current_env_id: Optional[str] = None
        self.current_env: Optional[gym.Env] = None

    # ---------- Curriculum / aggressiveness control ----------

    def set_curriculum_progress(self, progress: float) -> None:
        """
        Set aggressiveness based on curriculum progress in [0, 1].
        You can call this from a PPO callback during training.

        Example schedule:
            progress = num_timesteps / total_timesteps
        """
        self.aggressiveness = float(max(0.0, min(1.0, progress)))

    # ---------- Core env API ----------

    def _select_scenario(self) -> None:
        """Randomly choose one of the configured scenarios for this episode."""
        self.current_env_id = random.choice(self.env_ids)
        self.current_env = self._envs[self.current_env_id]

    def _build_scenario_config(self, env_id: str) -> Dict[str, Any]:
        """
        Build a per-scenario config that depends on self.aggressiveness.
        You can tweak these formulas to make traffic more 'antagonistic'.

        IMPORTANT:
        We also force the same 5-action DiscreteMetaAction for ALL scenarios,
        so PPO always sees a consistent action space.
        """
        # Base vehicles per scenario
        base_vehicles = {
            "highway-v0": 20,
            "merge-v0": 18,
            "intersection-v0": 10,
        }
        base = base_vehicles.get(env_id, 20)

        # Increase traffic with aggressiveness (0.0 → base, 1.0 → ~2.5x base)
        vehicles_count = int(base * (1.0 + 1.5 * self.aggressiveness))

        # Shared config: same observation + SAME ACTION SPACE for all envs
        cfg: Dict[str, Any] = {
            "observation": self.observation_config,
            "vehicles_count": vehicles_count,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "action": {
                # This gives us the 5-action ACTIONS_ALL:
                # 0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True,
            },
        }

        # Example: tweak duration per scenario
        if env_id == "highway-v0":
            cfg["duration"] = 40  # seconds
        elif env_id == "merge-v0":
            cfg["duration"] = 35
        elif env_id == "intersection-v0":
            cfg["duration"] = 35

        return cfg


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment:
          * pick a random scenario
          * configure its difficulty based on aggressiveness
          * reset the underlying highway-env
        """
        # Select scenario for this episode
        self._select_scenario()
        assert self.current_env is not None
        env_id = self.current_env_id

        # Configure based on current aggressiveness *before* reset
        cfg = self._build_scenario_config(env_id)
        try:
            self.current_env.unwrapped.configure(cfg)
        except AttributeError:
            # If for some reason configure doesn't exist, just skip
            pass

        # Now reset (optionally reseed)
        if seed is not None:
            obs, info = self.current_env.reset(seed=seed, options=options)
        else:
            obs, info = self.current_env.reset(options=options)

        # Tag scenario and aggressiveness in info for logging
        info = info or {}
        info["scenario"] = env_id
        info["aggressiveness"] = self.aggressiveness
        return obs, info


    def step(self, action):
        """Step through the current selected scenario."""
        assert self.current_env is not None, "Call reset() before step()."
        obs, reward, terminated, truncated, info = self.current_env.step(action)
        info = info or {}
        info["scenario"] = self.current_env_id
        info["aggressiveness"] = self.aggressiveness
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.current_env is None:
            return None
        return self.current_env.render()

    def close(self):
        for env in self._envs.values():
            env.close()
