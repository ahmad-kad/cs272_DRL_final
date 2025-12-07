"""
Reward wrapper for crazy_driver_env to prevent reward gaming through crashes.

This wrapper implements proper reward shaping to achieve:
- High survival rates through survival bonuses
- Low crash rates through severe crash penalties
- Long episode lengths through completion bonuses
- Conservative driving behavior through balanced incentives

KEY FIX: Eliminates crash exploitation where crash penalties equal avoidance penalties.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional
import copy

class CrazyDriverRewardWrapper(gym.Wrapper):
    """
    Wrapper that fixes reward exploitation in crazy_driver_env.

    Key fixes:
    - Crash penalties much harsher than avoidance penalties (prevents exploitation)
    - Survival bonuses encourage long episodes
    - Speed rewards for optimal driving
    - Balanced reward scales prevent optimization instability
    """

    def __init__(self, env, reward_config: Dict[str, Any] = None):
        """
        Args:
            env: The crazy_driver_env to wrap
            reward_config: Configuration for reward shaping
                - "survival_bonus": Bonus per step for surviving (default: 0.1)
                - "completion_bonus": Bonus for completing episodes (default: 1.0)
                - "speed_reward": Base speed reward (default: 0.2)
                - "near_miss_reward": Reward for close dodges (default: 0.5)
                - "cop_avoidance_penalty": Penalty for being near cops (default: 0.5)
                - "cop_behind_penalty": Penalty for getting behind cops (default: 2.0)
                - "cop_crash_penalty": Severe penalty for crashing into cops (default: 50.0)
                - "npc_crash_penalty": Penalty for crashing into NPCs (default: 25.0)
                - "offroad_penalty": Base off-road penalty (default: 0.25)
        """
        super().__init__(env)

        # Default reward configuration
        default_config = {
            "survival_bonus": 0.1,      # Small bonus per step for surviving
            "completion_bonus": 1.0,    # Bonus for completing episodes
            "speed_reward": 0.2,        # Base speed reward
            "near_miss_reward": 0.5,    # Close dodge reward
            "cop_avoidance_penalty": 0.5,  # Small penalty for being near cops
            "cop_behind_penalty": 2.0,     # Moderate penalty for getting behind cops
            "cop_crash_penalty": 50.0,     # SEVERE penalty for crashing into cops
            "npc_crash_penalty": 25.0,     # Heavy penalty for crashing into NPCs
            "offroad_penalty": 5.0,        # HARSH penalty for going off-road (increased from 0.25)
        }

        self.reward_config = {**default_config, **(reward_config or {})}

        # Track episode state for completion bonuses
        self.episode_start_time = 0
        self.episode_length = 0

        # Track off-road behavior for progressive termination
        self.offroad_timer = 0  # Consecutive steps off-road
        self.max_offroad_steps = 10  # Terminate after 10 consecutive off-road steps

        # Find the base environment with state information
        def find_base_env(obj, depth=0, max_depth=5):
            """Recursively find the base environment with road/vehicle attributes."""
            if hasattr(obj, 'road') and hasattr(obj, 'vehicle'):
                return obj, depth
            elif hasattr(obj, 'env') and depth < max_depth:
                return find_base_env(obj.env, depth + 1, max_depth)
            else:
                return None, -1

        self.base_env, _ = find_base_env(env)
        if self.base_env is None:
            raise AttributeError("Cannot find base environment with road/vehicle attributes")

        # Store original reward method for reference
        self._original_reward = self.base_env._reward


    def step(self, action):
        """Step with corrected reward shaping to prevent exploitation."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Apply corrected reward calculation
        corrected_reward = self._calculate_corrected_reward(action)

        # Track episode progress for completion bonuses
        self.episode_length += 1

        return obs, corrected_reward, terminated, truncated, info

    def _calculate_corrected_reward(self, action):
        """Reward safe navigation through dense traffic."""
        reward = 0
        env = self.base_env

        # ===== 1. BASE SURVIVAL (primary objective) =====
        if env.vehicle.on_road:
            reward += 1.0  # Strong constant reward for being alive and on-road

        # ===== 2. SAFE DISTANCE (avoid "walls of cars") =====
        min_distance = float('inf')
        vehicle_count_nearby = 0

        for v in env.road.vehicles:
            if v == env.vehicle:
                continue

            distance = np.linalg.norm(np.array(env.vehicle.position) - np.array(v.position))
            min_distance = min(min_distance, distance)

            # Count vehicles within 10m (situational awareness)
            if distance < 10.0:
                vehicle_count_nearby += 1

        # Reward maintaining safe buffer distance
        if 3.0 < min_distance < 8.0:
            reward += 0.5  # Sweet spot: close enough to progress, far enough to react
        elif min_distance < 2.0:
            reward -= 1.0  # Too close - danger zone

        # Bonus for navigating dense traffic successfully
        if vehicle_count_nearby > 5 and min_distance > 3.0:
            reward += 0.3  # Reward threading through "walls of cars"

        # ===== 3. FORWARD PROGRESS (don't camp) =====
        speed = env.vehicle.speed
        if 15 < speed < 22:  # Optimal speed range
            reward += 0.2
        elif speed < 10:
            reward -= 0.2  # Penalty for being too cautious

        # ===== 4. SMOOTH DRIVING (discourage erratic behavior) =====
        steering_magnitude = abs(action[1]) if len(action) > 1 else 0
        if steering_magnitude > 0.5:  # Large steering changes
            reward -= 0.1 * steering_magnitude

        # ===== 4.5 DIRECTION INCENTIVE (encourage driving at 22.5 degrees) =====
        heading = env.vehicle.heading
        # Normalize heading to [-π, π] range
        heading = ((heading + np.pi) % (2 * np.pi)) - np.pi

        # Target direction: 22.5 degrees (π/8 radians) - slight rightward angle
        target_heading = np.pi / 8  # 22.5 degrees in radians

        # Calculate angular distance from target direction
        heading_error = abs(heading - target_heading)
        # Handle circular nature: minimum angular distance
        heading_error = min(heading_error, 2 * np.pi - heading_error)

        # Reward being close to 22.5 degrees target (±22.5° tolerance = 0°-45° range), penalize deviation
        if heading_error < np.pi / 8:  # Within 22.5 degrees of target (±22.5° tolerance)
            reward += 0.3 * (1.0 - heading_error / (np.pi / 8))  # Up to 0.3 reward
        elif heading_error > np.pi / 2:  # More than 90 degrees off
            reward -= 0.2 * min(1.0, (heading_error - np.pi / 2) / (np.pi / 2))  # Penalty up to -0.2

        # ===== 5. COLLISION PENALTIES (catastrophic) =====
        if env.vehicle.crashed:
            reward -= 100.0  # Any collision is unacceptable
            return reward  # Immediate termination of reward calculation

        # ===== 6. OFF-ROAD HANDLING (progressive escalation) =====
        if not env.vehicle.on_road:
            self.offroad_timer += 1
            env.vehicle.MAX_SPEED = env.config["MAX_SPEED"] * 0.7

            # Escalating penalty: worse the longer you're off-road
            reward -= (3.0 * self.offroad_timer)  # 3, 6, 9, 12, ...

            if self.offroad_timer >= 10:
                env.vehicle.crashed = True  # Force termination
                reward -= 50.0  # Additional penalty
        else:
            self.offroad_timer = 0
            env.vehicle.MAX_SPEED = env.config["MAX_SPEED"]

        # ===== 7. TIME BONUS (encourage longer episodes) =====
        reward += 0.05 * env.time  # Gradually increasing value for survival

        return reward

    def get_reward_config(self) -> Dict[str, float]:
        """Get current reward configuration."""
        return copy.deepcopy(self.reward_config)

    def update_reward_config(self, new_config: Dict[str, float]):
        """Update reward configuration during training."""
        self.reward_config.update(new_config)

    def reset(self, **kwargs):
        """Reset environment and episode tracking."""
        self.episode_length = 0
        self.episode_start_time = 0
        self.offroad_timer = 0  # Reset off-road timer
        return self.env.reset(**kwargs)

    @classmethod
    def create_for_sac_td3(cls, env):
        """Create wrapper optimized for SAC/TD3 with survival-focused rewards."""
        config = {
            "survival_bonus": 1.0,        # Strong survival incentive (primary)
            "completion_bonus": 5.0,      # Reward for full episodes
            "speed_reward": 0.5,          # Secondary - only after survival proven
            "near_miss_reward": 0.5,      # Secondary - only after sustained survival
            "cop_crash_penalty": -20.0,    # Only penalty is for actual crashes
            "npc_crash_penalty": -25.0,    # Much harsher - all crashes should be very bad
            # Remove independent avoidance penalties to prioritize survival
            "cop_avoidance_penalty": 0.0,  # No penalty for being near cops
            "cop_behind_penalty": 0.0,     # No penalty for being behind cops
            "offroad_penalty": -40.0,      # Harsh penalty for off-road - worse than crashes
        }
        return cls(env, config)

    @classmethod
    def create_for_ppo_attention(cls, env):
        """Create wrapper optimized for PPO with attention."""
        config = {
            "survival_bonus": 0.05,  # Smaller survival bonus for PPO
            "completion_bonus": 0.5,
            "speed_reward": 0.2,
            "near_miss_reward": 0.5,  # Standard reward for PPO
            "cop_avoidance_penalty": 0.3,
            "cop_behind_penalty": 1.5,
            "cop_crash_penalty": 75.0,   # Less harsh than SAC/TD3
            "npc_crash_penalty": 35.0,   # Less harsh than SAC/TD3
            "offroad_penalty": 0.25,
        }
        return cls(env, config)


def create_wrapped_crazy_driver_env(algorithm: str = "tqc", episode_duration: int = 120):
    """
    Factory function to create a properly configured crazy_driver_env with reward wrapper.

    Args:
        algorithm: Which algorithm to optimize for ("tqc", "ppo")
        episode_duration: Episode length in seconds (default 120 for longer training)

    Returns:
        Wrapped environment ready for training
    """
    import gymnasium as gym
    from team2_env.crazy_driver_enviornment import crazy_driver_env

    # Configure environment for longer episodes and off-road termination
    config = crazy_driver_env.default_config()
    config.update({
        "duration": episode_duration,  # Longer episodes for better learning
        "offroad_terminal": True,      # Terminate episodes when going off-road
        "simulation_frequency": 15,
        "policy_frequency": 3,
    })

    env = gym.make("CopChase-v0", config=config)

    # Apply algorithm-specific reward wrapper
    if algorithm.lower() == "tqc":
        wrapped_env = CrazyDriverRewardWrapper.create_for_sac_td3(env)  # TQC uses same config as SAC/TD3
    elif algorithm.lower() == "ppo":
        wrapped_env = CrazyDriverRewardWrapper.create_for_ppo_attention(env)
    else:
        # Default to TQC configuration
        wrapped_env = CrazyDriverRewardWrapper.create_for_sac_td3(env)

    return wrapped_env
