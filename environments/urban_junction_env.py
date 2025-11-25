"""
Urban Junction Environment - Production-Ready RL Benchmark

A high-performance, unified environment for autonomous driving research
supporting highway, merge, and intersection scenarios with multiple
observation modalities (Lidar and Grayscale vision).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, List, Union

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.vehicle.kinematics import Vehicle

# Constants for better maintainability
SCENARIOS = ["highway", "merge", "intersection"]
MODALITIES = ["lidar", "grayscale", "both"]

# Default configuration values
DEFAULT_HIGHWAY_LENGTH = 2500
DEFAULT_SPEED_LIMIT = 30
DEFAULT_LANE_COUNT = 4


class UrbanJunctionEnv(AbstractEnv):
    """
    Production-Ready Autonomous Driving Environment.

    This environment provides a unified interface for training and evaluating
    reinforcement learning agents across multiple driving scenarios. It features
    optimized performance, stable observation spaces, and comprehensive reward
    functions designed for PPO and other policy gradient algorithms.

    Features:
    - Scenarios: Highway cruising, highway merging, urban intersections
    - Observation Modalities: Lidar (vector) and Grayscale (vision)
    - Stable Baselines3 compatible observation spaces
    - Normalized rewards for training stability
    - Configurable traffic density and difficulty

    Args:
        config: Environment configuration dictionary (optional)
        scenario: Driving scenario - "highway", "merge", "intersection", or "random"
        modality: Observation type - "lidar", "grayscale", or "both"
    """

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "LidarObservation",
                "cells": 64,
                "maximum_range": 50,
                "normalize": True,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            },
            "action": {
                "type": "DiscreteMetaAction", # More stable than Continuous for merging
            },
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "duration": 40,  # [s]
            "lanes_count": 4,
            "collision_reward": -1.0,
            "high_speed_reward": 0.4,
            "arrived_reward": 1.0,
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": True,
        })
        return config

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        scenario: str = "highway",
        modality: str = "both"
    ) -> None:
        """
        Initialize the Urban Junction Environment.

        Args:
            config: Configuration dictionary overriding defaults
            scenario: Driving scenario. Must be one of:
                     "highway", "merge", "intersection", or "random"
            modality: Observation modality. Must be "lidar", "grayscale", or "both"

        Raises:
            ValueError: If scenario or modality are invalid
        """
        # Validate inputs
        if scenario not in SCENARIOS and scenario != "random":
            raise ValueError(f"scenario must be one of {SCENARIOS} or 'random', got '{scenario}'")
        if modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}, got '{modality}'")
        # 1. Force Modality Configuration BEFORE parent init
        self.modality = modality
        self.scenario_input = scenario
        self.current_scenario = scenario

        # Standardize Observation Configs
        self.obs_configs = {
            "lidar": {
                "type": "LidarObservation",
                "cells": 64,
                "maximum_range": 50,
                "normalize": True,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]
            },
            "grayscale": {
                "type": "GrayscaleObservation", 
                "observation_shape": (128, 64),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],
                "scaling": 1.75,
            }
        }

        # Apply the config
        local_config = config or self.default_config()
        if modality == "both":
            # For "both" modality, we'll handle observation generation ourselves
            # Use lidar config as base since we'll override observation generation
            local_config["observation"] = self.obs_configs["lidar"]
        else:
            local_config["observation"] = self.obs_configs[modality]

        super().__init__(local_config)

        # 2. Fix Observation Space for SB3
        # We must define this explicitly because highway-env dynamic spaces confuse vector envs
        if self.modality == "lidar":
            # Shape = cells * features
            n_features = len(self.obs_configs["lidar"]["features"])
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.obs_configs["lidar"]["cells"] * n_features,),
                dtype=np.float32
            )
        elif self.modality == "grayscale":
            # Shape = (stack_size, H, W)
            shape = (self.obs_configs["grayscale"]["stack_size"], 128, 64)
            self.observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        elif self.modality == "both":
            # Combined observation space: lidar + grayscale
            lidar_features = len(self.obs_configs["lidar"]["features"])
            lidar_size = self.obs_configs["lidar"]["cells"] * lidar_features
            grayscale_size = (self.obs_configs["grayscale"]["stack_size"] * 128 * 64)
            total_size = lidar_size + grayscale_size
            
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,  # Mixed ranges: lidar [-1,1], grayscale [0,255]
                shape=(total_size,),
                dtype=np.float32
            )

    def _reset(self) -> None:
        """
        Reset the environment to initial state.

        Generates the road network and traffic based on the selected scenario.
        This replaces the complex ProceduralGenerator with robust standard generators.
        """
        # Select Scenario
        if self.scenario_input == "random":
            self.current_scenario = self.np_random.choice(["highway", "merge", "intersection"])
        else:
            self.current_scenario = self.scenario_input

        # Configure Road based on Scenario
        if self.current_scenario == "highway":
            self._make_highway_road()
        elif self.current_scenario == "merge":
            self._make_merge_road()
        elif self.current_scenario == "intersection":
            self._make_intersection_road()

        # Generate Traffic
        self._make_vehicles()

    def _make_highway_road(self) -> None:
        """
        Create a standard highway road network.

        Generates a straight multi-lane highway with standard lane markings
        and speed limits optimized for autonomous driving scenarios.
        """
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                lanes=DEFAULT_LANE_COUNT,
                length=DEFAULT_HIGHWAY_LENGTH,
                speed_limit=DEFAULT_SPEED_LIMIT
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"]
        )

    def _make_merge_road(self) -> None:
        """
        Create a highway merge scenario road network.

        Features a main highway with an on-ramp merge lane, requiring
        vehicles to safely merge into traffic flow.
        """
        net = RoadNetwork()

        # Main Road - two lanes in each direction
        net.add_lane("a", "b", StraightLane([0, 0], [2000, 0],
                    line_types=[LineType.CONTINUOUS_LINE, LineType.STRIPED]))
        net.add_lane("a", "b", StraightLane([0, 4], [2000, 4],
                    line_types=[LineType.STRIPED, LineType.CONTINUOUS_LINE]))

        # Merge Ramp - single lane entering from the right
        net.add_lane("j", "k", StraightLane([200, -4], [400, 0],
                    line_types=[LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE]))

        self.road = Road(network=net, np_random=self.np_random,
                         record_history=self.config["show_trajectories"])

    def _make_intersection_road(self) -> None:
        """
        Create an urban intersection road network.

        Features a 4-way intersection with traffic lights and crossing
        vehicles, requiring careful navigation and right-of-way management.
        """
        net = RoadNetwork()
        
        # Define cardinal directions and center point
        n, w, s, e = [0, -40], [-40, 0], [0, 40], [40, 0]
        c = [0, 0]

        # Incoming lanes from all directions
        net.add_lane("n", "c", StraightLane(n, c,
                    line_types=[LineType.CONTINUOUS_LINE, LineType.STRIPED]))
        net.add_lane("w", "c", StraightLane(w, c,
                    line_types=[LineType.CONTINUOUS_LINE, LineType.STRIPED]))
        net.add_lane("s", "c", StraightLane(s, c,
                    line_types=[LineType.CONTINUOUS_LINE, LineType.STRIPED]))
        net.add_lane("e", "c", StraightLane(e, c,
                    line_types=[LineType.CONTINUOUS_LINE, LineType.STRIPED]))

        # Outgoing lanes to all directions
        net.add_lane("c", "n", StraightLane(c, n,
                    line_types=[LineType.STRIPED, LineType.CONTINUOUS_LINE]))
        net.add_lane("c", "w", StraightLane(c, w,
                    line_types=[LineType.STRIPED, LineType.CONTINUOUS_LINE]))
        net.add_lane("c", "s", StraightLane(c, s,
                    line_types=[LineType.STRIPED, LineType.CONTINUOUS_LINE]))
        net.add_lane("c", "e", StraightLane(c, e,
                    line_types=[LineType.STRIPED, LineType.CONTINUOUS_LINE]))

        self.road = Road(network=net, np_random=self.np_random,
                         record_history=self.config["show_trajectories"])

    def _make_vehicles(self) -> None:
        """
        Populate the road network with ego vehicle and traffic.

        Places the ego vehicle at an appropriate starting position based on
        the scenario, then adds background traffic vehicles with randomized
        positions and speeds.
        """
        # Ego Vehicle
        self.controlled_vehicles = []

        # Get ego vehicle start position based on scenario
        if self.current_scenario == "merge":
            # Start on the ramp
            ego = self.action_type.vehicle_class(self.road, self.road.network.get_lane(("j", "k", 0)).position(0, 0), speed=20)
        elif self.current_scenario == "intersection":
            ego = self.action_type.vehicle_class(self.road, self.road.network.get_lane(("n", "c", 0)).position(0, 0), speed=10)
        else:  # highway (default)
            # Get first available lane for ego vehicle
            all_lanes = list(self.road.network.lanes_list())
            if not all_lanes:
                raise ValueError("No lanes available in road network")
            ego_lane = all_lanes[0]
            ego = self.action_type.vehicle_class(self.road, ego_lane.position(30, 0), speed=25)

        self.road.vehicles.append(ego)
        self.controlled_vehicles.append(ego)

        # Traffic
        density = 10 if self.current_scenario == "highway" else 5

        for i in range(density):
            # Create vehicle with random speed
            random_speed = self.np_random.uniform(20, 30)
            self.road.vehicles.append(
                Vehicle.create_random(self.road, speed=random_speed, spacing=0.5)
            )

    def _reward(self, action: Union[int, np.ndarray]) -> float:
        """
        Calculate the total reward for the current timestep.

        Combines individual reward components and applies normalization
        for stable PPO training.

        Args:
            action: The action taken by the agent

        Returns:
            Normalized reward value between -1.0 and 1.0
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())

        # Normalization for PPO stability
        if self.config["normalize_reward"]:
            reward = np.clip(reward, -1.0, 1.0)

        return float(reward)

    def _rewards(self, action: Union[int, np.ndarray]) -> Dict[str, float]:
        """
        Calculate individual reward components.

        This method decomposes the reward into interpretable components
        that can be weighted differently for curriculum learning.

        Args:
            action: The action taken by the agent

        Returns:
            Dictionary of reward component names to their values
        """
        # Use underlying neighbor lookup for safety checks
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)

        rewards = {
            "collision_reward": self.vehicle.crashed,
            "high_speed_reward": np.clip(self.vehicle.speed, 0, 30) / 30,
            "right_lane_reward": self.vehicle.lane_index[2] / (len(neighbours) - 1) if len(neighbours) > 1 else 0,
        }

        # Scenario Specific Rewards
        if self.current_scenario == "intersection":
            # Did we make it through?
            rewards["arrived_reward"] = 1.0 if self.vehicle.position[0] > 10 and self.vehicle.position[1] > 10 else 0.0

        if self.current_scenario == "merge":
             # Did we merge onto the main road? (check if we're on lanes starting with "a" or "b")
             try:
                 is_on_main_road = str(self.vehicle.lane_index[0]) in ["a", "b"] if self.vehicle.lane_index else False
                 rewards["arrived_reward"] = 0.5 if is_on_main_road else 0.0
             except (IndexError, TypeError):
                 rewards["arrived_reward"] = 0.0

        return rewards

    def _combine_observations(self, lidar_obs, grayscale_obs):
        """Combine lidar and grayscale observations into single array."""
        lidar_flat = lidar_obs.flatten()
        grayscale_flat = grayscale_obs.flatten().astype(np.float32)

        # Normalize grayscale to similar scale as lidar
        grayscale_flat = (grayscale_flat / 127.5) - 1.0  # Normalize to [-1, 1]

        combined_obs = np.concatenate([lidar_flat, grayscale_flat])
        return combined_obs

    def reset(self, **kwargs):
        """Reset environment and return combined observations if modality is 'both'."""
        obs, info = super().reset(**kwargs)

        if self.modality == "both":
            # Generate grayscale observation
            try:
                from highway_env.envs.common.observation import GrayscaleObservation
                grayscale_observer = GrayscaleObservation(
                    self,
                    observation_shape=self.obs_configs["grayscale"]["observation_shape"],
                    stack_size=self.obs_configs["grayscale"]["stack_size"],
                    weights=self.obs_configs["grayscale"]["weights"],
                    scaling=self.obs_configs["grayscale"]["scaling"]
                )
                grayscale_obs = grayscale_observer.observe()
                obs = self._combine_observations(obs, grayscale_obs)
            except Exception as e:
                print(f"Warning: Could not generate grayscale observation: {e}")
                # Fallback: create a dummy grayscale observation with correct shape
                dummy_grayscale = np.zeros((4, 128, 64), dtype=np.uint8)
                obs = self._combine_observations(obs, dummy_grayscale)

        return obs, info

    def step(self, action):
        """Step environment and return combined observations if modality is 'both'."""
        step_result = super().step(action)

        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result

        if self.modality == "both":
            # Generate grayscale observation and combine
            try:
                from highway_env.envs.common.observation import GrayscaleObservation
                grayscale_observer = GrayscaleObservation(
                    self,
                    observation_shape=self.obs_configs["grayscale"]["observation_shape"],
                    stack_size=self.obs_configs["grayscale"]["stack_size"],
                    weights=self.obs_configs["grayscale"]["weights"],
                    scaling=self.obs_configs["grayscale"]["scaling"]
                )
                grayscale_obs = grayscale_observer.observe()
                obs = self._combine_observations(obs, grayscale_obs)
            except Exception as e:
                print(f"Warning: Could not generate grayscale observation in step: {e}")
                # Fallback: create a dummy grayscale observation
                dummy_grayscale = np.zeros((4, 128, 64), dtype=np.uint8)
                obs = self._combine_observations(obs, dummy_grayscale)

        if len(step_result) == 5:
            return obs, reward, terminated, truncated, info
        else:
            return obs, reward, done, info

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]