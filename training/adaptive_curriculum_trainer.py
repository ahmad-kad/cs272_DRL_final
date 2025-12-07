"""
Adaptive Curriculum Trainer for Crazy Driver Environment

Gradually increases episode length as the agent improves, enabling learning of
long-term navigation strategies in dense traffic scenarios.

Key Features:
- Monitors survival rates and reward trends
- Dynamically adjusts episode duration (30s to 300s)
- Implements staged curriculum progression
- Tracks learning milestones for curriculum advancement
"""

import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from typing import Dict, Any, Optional, List
import time
import logging

logger = logging.getLogger(__name__)


class AdaptiveCurriculumTrainer(BaseCallback):
    """
    Adaptive curriculum that increases episode length based on agent performance.

    Curriculum Stages:
    1. Short episodes (30s): Learn basic survival and collision avoidance
    2. Medium episodes (60s): Learn traffic navigation patterns
    3. Long episodes (120s): Learn strategic positioning and long-term planning
    4. Extended episodes (180s-300s): Master complex traffic scenarios
    """

    def __init__(
        self,
        curriculum_stages: Optional[List[Dict[str, Any]]] = None,
        performance_window: int = 100,  # episodes to evaluate performance
        advancement_threshold: float = 0.7,  # survival rate needed to advance
        min_episodes_per_stage: int = 500,  # minimum episodes before advancement check
        verbose: int = 1,
    ):
        """
        Args:
            curriculum_stages: List of curriculum stages with episode durations and requirements
            performance_window: Number of recent episodes to evaluate performance
            advancement_threshold: Survival rate threshold for stage advancement
            min_episodes_per_stage: Minimum episodes before checking advancement
            verbose: Verbosity level
        """
        super().__init__(verbose)

        # Default curriculum stages
        if curriculum_stages is None:
            self.curriculum_stages = [
                {
                    "name": "survival_basics",
                    "duration": 30,  # seconds
                    "description": "Learn basic survival and collision avoidance",
                    "min_performance": 0.5,  # 50% survival rate
                },
                {
                    "name": "traffic_navigation",
                    "duration": 60,
                    "description": "Learn traffic flow and basic navigation",
                    "min_performance": 0.6,  # 60% survival rate
                },
                {
                    "name": "strategic_positioning",
                    "duration": 120,
                    "description": "Learn strategic positioning and gap finding",
                    "min_performance": 0.7,  # 70% survival rate
                },
                {
                    "name": "complex_traffic",
                    "duration": 180,
                    "description": "Master complex multi-vehicle interactions",
                    "min_performance": 0.75,  # 75% survival rate
                },
                {
                    "name": "expert_navigation",
                    "duration": 300,
                    "description": "Expert-level traffic navigation and long-term planning",
                    "min_performance": 0.8,  # 80% survival rate
                }
            ]
        else:
            self.curriculum_stages = curriculum_stages

        self.performance_window = performance_window
        self.advancement_threshold = advancement_threshold
        self.min_episodes_per_stage = min_episodes_per_stage

        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_survivals = []  # 1 if survived full episode, 0 if crashed
        self.current_stage_idx = 0
        self.episodes_in_current_stage = 0
        self.stage_start_time = time.time()

        # Curriculum state
        self.current_duration = self.curriculum_stages[0]["duration"]
        self.best_survival_rate = 0.0

        # Environment reference (will be set during training)
        self.env = None

        if verbose > 0:
            print(f"[CURRICULUM] Initialized with {len(self.curriculum_stages)} stages")
            print(f"[CURRICULUM] Starting with {self.current_duration}s episodes")

    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        self.stage_start_time = time.time()
        if self.verbose > 0:
            print(f"[CURRICULUM] Training started at stage {self.current_stage_idx + 1}: "
                  f"{self.curriculum_stages[self.current_stage_idx]['name']}")

    def _on_step(self) -> bool:
        """Called at each training step."""
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout (when episode ends)."""
        # Get episode info from the monitor
        if hasattr(self.model, 'env') and hasattr(self.model.env, 'envs'):
            # Multi-environment case
            for env_idx, env in enumerate(self.model.env.envs):
                if hasattr(env, 'episode_returns') and len(env.episode_returns) > 0:
                    self._process_episode_info(env)
        elif hasattr(self.model, 'env') and hasattr(self.model.env, 'episode_returns'):
            # Single environment case
            self._process_episode_info(self.model.env)

        # Check for curriculum advancement
        self._check_curriculum_advancement()

    def _process_episode_info(self, env):
        """Process episode information from monitor."""
        if hasattr(env, 'episode_returns') and len(env.episode_returns) > 0:
            reward = env.episode_returns[-1]

            # Check if episode was completed (survived) or terminated early (crashed)
            # This is a simplified heuristic - in practice you'd want more sophisticated detection
            episode_length = getattr(env, 'episode_lengths', [-1])[-1]
            max_possible_length = self.current_duration * env.unwrapped.config["policy_frequency"]

            # Survival is determined by whether the episode reached near-maximum length
            survived = episode_length >= (max_possible_length * 0.8)  # 80% of max length

            self.episode_rewards.append(reward)
            self.episode_lengths.append(episode_length)
            self.episode_survivals.append(1 if survived else 0)

            # Keep only recent episodes for performance evaluation
            if len(self.episode_rewards) > self.performance_window:
                self.episode_rewards.pop(0)
                self.episode_lengths.pop(0)
                self.episode_survivals.pop(0)

    def _check_curriculum_advancement(self):
        """Check if agent is ready to advance to next curriculum stage."""
        if len(self.episode_survivals) < self.performance_window:
            return  # Not enough data yet

        # Calculate current performance metrics
        survival_rate = np.mean(self.episode_survivals)
        avg_reward = np.mean(self.episode_rewards[-50:])  # Last 50 episodes
        avg_length = np.mean(self.episode_lengths[-50:])

        # Update best survival rate
        self.best_survival_rate = max(self.best_survival_rate, survival_rate)

        current_stage = self.curriculum_stages[self.current_stage_idx]
        episodes_in_stage = len([s for s in self.episode_survivals[-self.min_episodes_per_stage:]])

        # Check advancement conditions
        ready_to_advance = (
            survival_rate >= current_stage["min_performance"] and
            episodes_in_stage >= self.min_episodes_per_stage and
            survival_rate >= self.advancement_threshold
        )

        if ready_to_advance and self.current_stage_idx < len(self.curriculum_stages) - 1:
            self._advance_curriculum(survival_rate, avg_reward, avg_length)

    def _advance_curriculum(self, survival_rate, avg_reward, avg_length):
        """Advance to the next curriculum stage."""
        old_stage = self.curriculum_stages[self.current_stage_idx]
        self.current_stage_idx += 1
        new_stage = self.curriculum_stages[self.current_stage_idx]

        old_duration = self.current_duration
        self.current_duration = new_stage["duration"]
        self.episodes_in_current_stage = 0

        # Update environment duration
        self._update_environment_duration()

        stage_time = time.time() - self.stage_start_time
        self.stage_start_time = time.time()

        if self.verbose > 0:
            print(f"\n[CURRICULUM] 🎯 ADVANCING TO STAGE {self.current_stage_idx + 1}")
            print(f"[CURRICULUM] Previous: {old_stage['name']} ({old_duration}s)")
            print(f"[CURRICULUM] New: {new_stage['name']} ({self.current_duration}s)")
            print(f"[CURRICULUM] Survival Rate: {survival_rate:.1%}")
            print(f"[CURRICULUM] Average Reward: {avg_reward:.2f}")
            print(f"[CURRICULUM] Stage completed in: {stage_time:.1f}s")
            print(f"[CURRICULUM] {new_stage['description']}\n")

    def _update_environment_duration(self):
        """Update the episode duration in the environment configuration."""
        if self.model and hasattr(self.model, 'env'):
            # Update the underlying environment config
            if hasattr(self.model.env, 'envs'):
                # VecEnv case
                for env in self.model.env.envs:
                    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'config'):
                        env.unwrapped.config["duration"] = self.current_duration
            else:
                # Single env case
                if hasattr(self.model.env, 'unwrapped') and hasattr(self.model.env.unwrapped, 'config'):
                    self.model.env.unwrapped.config["duration"] = self.current_duration

    def get_curriculum_status(self) -> Dict[str, Any]:
        """Get current curriculum status for logging."""
        if len(self.episode_survivals) == 0:
            survival_rate = 0.0
            avg_reward = 0.0
        else:
            survival_rate = np.mean(self.episode_survivals)
            avg_reward = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards)

        current_stage = self.curriculum_stages[self.current_stage_idx]

        return {
            "stage": self.current_stage_idx + 1,
            "stage_name": current_stage["name"],
            "episode_duration": self.current_duration,
            "survival_rate": survival_rate,
            "avg_reward": avg_reward,
            "episodes_in_stage": self.episodes_in_current_stage,
            "best_survival_rate": self.best_survival_rate,
            "total_stages": len(self.curriculum_stages)
        }


def create_adaptive_curriculum_callback(
    curriculum_stages=None,
    performance_window=100,
    advancement_threshold=0.7,
    min_episodes_per_stage=500,
    verbose=1
):
    """
    Create an adaptive curriculum callback for training.

    Args:
        curriculum_stages: Custom curriculum stages (optional)
        performance_window: Episodes to evaluate performance
        advancement_threshold: Survival rate needed to advance
        min_episodes_per_stage: Minimum episodes before advancement check
        verbose: Verbosity level

    Returns:
        AdaptiveCurriculumTrainer callback
    """
    return AdaptiveCurriculumTrainer(
        curriculum_stages=curriculum_stages,
        performance_window=performance_window,
        advancement_threshold=advancement_threshold,
        min_episodes_per_stage=min_episodes_per_stage,
        verbose=verbose
    )





