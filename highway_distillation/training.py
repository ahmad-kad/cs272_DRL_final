#!/usr/bin/env python3
"""
Highway Merge Training - Optimized Generalist Agents

Streamlined training script for creating Generalist Agents (Lidar & Grayscale)
that can beat multiple highway-env scenarios.
"""

import os
import time
import numpy as np
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import wandb

try:
    from environments.urban_junction_env import UrbanJunctionEnv
    from config import PHASES, TRAINING, PATHS, ENVIRONMENT
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from environments.urban_junction_env import UrbanJunctionEnv
    from config import PHASES, TRAINING, PATHS, ENVIRONMENT


class WandBCallback(BaseCallback):
    """Enhanced WandB callback with comprehensive logging for vectorized environments."""

    def __init__(self, agent_type: str, log_freq: int = 250, verbose: int = 0):
        super().__init__(verbose)
        self.agent_type = agent_type
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.num_envs = None  # Will be set on first step
        self.episodes_completed_this_interval = 0

    def _on_step(self) -> bool:
        # Get environment data
        dones = self.locals.get('dones', np.array([]))
        rewards = self.locals.get('rewards', np.array([]))

        # Initialize per-environment tracking on first step
        if self.num_envs is None:
            self.num_envs = len(dones) if len(dones) > 0 else 1
            # Initialize tracking for each environment
            for env_idx in range(self.num_envs):
                setattr(self, f'current_reward_{env_idx}', 0.0)
                setattr(self, f'current_length_{env_idx}', 0)
                setattr(self, f'env_episode_rewards_{env_idx}', [])

        # Track episodes for each environment
        for env_idx in range(self.num_envs):
            if env_idx >= len(rewards):
                continue

            # Accumulate reward and length for this environment
            current_reward = getattr(self, f'current_reward_{env_idx}')
            current_length = getattr(self, f'current_length_{env_idx}')

            current_reward += rewards[env_idx]
            current_length += 1

            setattr(self, f'current_reward_{env_idx}', current_reward)
            setattr(self, f'current_length_{env_idx}', current_length)

            # Check if episode ended for this environment
            if env_idx < len(dones) and dones[env_idx]:
                # Episode completed
                env_episode_rewards = getattr(self, f'env_episode_rewards_{env_idx}')
                env_episode_rewards.append(current_reward)
                self.episode_rewards.append(current_reward)
                self.episode_lengths.append(current_length)
                self.episodes_completed_this_interval += 1

                # Keep only recent episodes (global and per-environment)
                if len(self.episode_rewards) > 100:
                    self.episode_rewards = self.episode_rewards[-100:]
                    self.episode_lengths = self.episode_lengths[-100:]

                # Keep per-environment episodes for rolling stats
                if len(env_episode_rewards) > 50:
                    env_episode_rewards[:] = env_episode_rewards[-50:]

                # Reset for next episode in this environment
                setattr(self, f'current_reward_{env_idx}', 0.0)
                setattr(self, f'current_length_{env_idx}', 0)

        # Log at regular intervals
        if self.n_calls % self.log_freq == 0:
            metrics = {
                "train/timesteps": self.num_timesteps,
                "train/updates": self.n_calls,
                "train/agent_type": self.agent_type,
                "train/parallel_envs": self.num_envs,
                "train/episodes_this_interval": self.episodes_completed_this_interval
            }

            # PPO-specific metrics
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                logs = self.model.logger.name_to_value
                for key in ['train/value_loss', 'train/policy_gradient_loss',
                           'train/entropy_loss', 'train/approx_kl', 'train/clip_fraction',
                           'train/learning_rate', 'train/n_updates']:
                    if key in logs:
                        clean_key = key.replace('train/', 'train/ppo_')
                        metrics[clean_key] = logs[key]

            # Episode statistics (rolling averages from all environments)
            if self.episode_rewards:
                recent_rewards = self.episode_rewards[-20:]  # Last 20 episodes across all envs
                recent_lengths = self.episode_lengths[-20:]
                all_rewards = self.episode_rewards

                metrics.update({
                    "rewards/episode_mean": np.mean(recent_rewards),
                    "rewards/episode_best": np.max(recent_rewards),
                    "rewards/episode_worst": np.min(recent_rewards),
                    "rewards/episode_std": np.std(recent_rewards) if len(recent_rewards) > 1 else 0,
                    "rewards/all_time_best": np.max(all_rewards),
                    "episode/length_mean": np.mean(recent_lengths),
                    "episode/total_count": len(self.episode_rewards),
                })

                # Learning progress indicators
                if len(self.episode_rewards) > 10:
                    early_avg = np.mean(self.episode_rewards[:10])
                    recent_avg = np.mean(recent_rewards)
                    metrics["rewards/learning_improvement"] = recent_avg - early_avg
                    
                    # Calculate success rate (episodes with positive reward)
                    positive_episodes = sum(1 for r in recent_rewards if r > 0)
                    metrics["episode/success_rate"] = positive_episodes / len(recent_rewards)

            # Reset interval counter
            self.episodes_completed_this_interval = 0

            # Log to wandb
            wandb.log(metrics, step=self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        """Log final training summary."""
        if self.episode_rewards:
            final_metrics = {
                "final/avg_reward": np.mean(self.episode_rewards[-50:]),
                "final/best_reward": np.max(self.episode_rewards),
                "final/total_episodes": len(self.episode_rewards),
                "final/reward_std": np.std(self.episode_rewards[-50:]) if len(self.episode_rewards) > 50 else 0,
                "final/training_completed": True
            }
            wandb.log(final_metrics)


def create_environment(use_grayscale_only: bool = False, use_lidar_only: bool = False, num_envs: int = 1, use_subprocess: bool = False):
    """Create environment with specific sensor configuration."""
    env_config = UrbanJunctionEnv.default_config()
    
    # Set optimized sensor flags
    env_config["use_grayscale_only"] = use_grayscale_only
    env_config["use_lidar_only"] = use_lidar_only

    def make_env():
        def _init():
            # Create env with explicit flags
            env = UrbanJunctionEnv(
                config=env_config, 
                use_lidar_only=use_lidar_only,
                use_grayscale_only=use_grayscale_only
            )
            
            # Monitor wrapper
            monitor_dir = os.path.join(PATHS["logs"], "monitor")
            os.makedirs(monitor_dir, exist_ok=True)
            return Monitor(env, filename=os.path.join(monitor_dir, f"env_{os.getpid()}"))
        return _init

    # Create test env to check shapes
    temp_env = make_env()()
    obs_shape = temp_env.observation_space.shape
    temp_env.close()
    
    print(f"Creating {num_envs} environments. Observation shape: {obs_shape}")

    if use_subprocess and num_envs > 1:
        env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    else:
        env = DummyVecEnv([make_env() for _ in range(num_envs)])

    # Normalize observations and rewards (crucial for PPO performance)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    return env


def create_model(env, use_grayscale_only: bool = False, use_lidar_only: bool = False, algorithm: str = 'ppo'):
    """Create optimized model based on sensor type."""
    
    # 1. LIDAR AGENT (MLP)
    if use_lidar_only:
        print("[LIDAR] Creating LIDAR Agent (MlpPolicy)")
        return PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[256, 256]),  # Simple but deep MLP
            learning_rate=3e-4,
            batch_size=256,
            n_steps=2048,
            verbose=1,
            device="cuda",
            tensorboard_log=PATHS["logs"]
        )

    # 2. GRAYSCALE AGENT (CNN)
    elif use_grayscale_only:
        print("[GRAYSCALE] Creating GRAYSCALE Agent (CnnPolicy)")
        return PPO(
            "CnnPolicy",
            env,
            learning_rate=2e-4,  # Slightly lower for CNNs
            batch_size=128,
            n_steps=1024,
            verbose=1,
            device="cuda",
            tensorboard_log=PATHS["logs"]
        )

    # 3. FUSION AGENT (Legacy/Advanced)
    else:
        print("[FUSION] Creating FUSION Agent (Custom Policy)")
        from custom_policies import SensorBasedActorCriticPolicy
        return PPO(
            SensorBasedActorCriticPolicy,
            env,
            verbose=1,
            device="cuda",
            tensorboard_log=PATHS["logs"]
        )


def train_generalist_agent(agent_type: str):
    """
    Train a single generalist agent type.
    agent_type: 'lidar' or 'grayscale'
    """
    if agent_type not in ['lidar', 'grayscale']:
        raise ValueError("agent_type must be 'lidar' or 'grayscale'")

    print(f"\n{'='*60}")
    print(f"[TRAIN] TRAINING {agent_type.upper()} GENERALIST AGENT")
    print(f"{'='*60}")

    # Initialize wandb for this agent
    wandb_config = {
        "agent_type": agent_type,
        "algorithm": "PPO",
        "sensor_type": "lidar_only" if agent_type == "lidar" else "grayscale_only",
        "environment": "UrbanJunctionEnv",
        "scenarios": ["highway", "merge", "intersection"],
        "goal": "Single agent to beat all highway-env environments"
    }

    # Initialize wandb project
    wandb.init(
        project="highway-distillation-generalist",
        name=f"{agent_type}_generalist_{int(time.time())}",
        config=wandb_config,
        notes=f"Training {agent_type} generalist agent to master highway, merge, and intersection scenarios"
    )

    print("[WANDB] WandB initialized - monitoring training progress...")

    # Configuration
    phase_name = f"{agent_type}_generalist"
    use_lidar = (agent_type == 'lidar')
    use_gray = (agent_type == 'grayscale')

    # More parallel envs for Lidar (fast), fewer for Grayscale (slow rendering)
    num_envs = 8 if use_lidar else 4
    total_timesteps = 300000 if use_lidar else 500000  # Images need more samples

    # Create Environment
    env = create_environment(
        use_grayscale_only=use_gray,
        use_lidar_only=use_lidar,
        num_envs=num_envs,
        use_subprocess=True
    )

    # Create Model
    model = create_model(env, use_grayscale_only=use_gray, use_lidar_only=use_lidar)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=25000 // num_envs,
        save_path=os.path.join(PATHS['models'], phase_name),
        name_prefix=phase_name
    )

    # Enhanced WandB logging
    wandb_callback = WandBCallback(
        agent_type=agent_type,
        log_freq=500,  # Log every 500 steps
        verbose=1
    )

    # Track training start time
    training_start_time = time.time()

    print(f"Training for {total_timesteps} steps...")
    print(f"[INFO] WandB logging enabled - check your dashboard at: https://wandb.ai")
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, wandb_callback])

    training_duration = time.time() - training_start_time

    # Save Final Model
    final_path = os.path.join(PATHS['models'], phase_name, "final_model")
    model.save(final_path)
    env.save(os.path.join(PATHS['models'], phase_name, "vec_normalize.pkl"))

    print(f"[SAVED] {agent_type.upper()} Agent Saved to: {final_path}")

    # Final wandb logging
    if wandb.run is not None:
        final_summary = {
            "training_completed": True,
            "total_timesteps": total_timesteps,
            "parallel_envs": num_envs,
            "final_model_path": final_path,
            "sensor_type": "lidar_only" if use_lidar else "grayscale_only",
            "actual_training_time_minutes": training_duration / 60,
            "timesteps_per_second": total_timesteps / training_duration if training_duration > 0 else 0,
        }
        wandb.log(final_summary)
        print(f"[WANDB] Final metrics logged to WandB (training took {training_duration/60:.1f} minutes)")

    env.close()
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, choices=['lidar', 'grayscale', 'all'], default='all')
    args = parser.parse_args()
    
    if args.agent in ['lidar', 'all']:
        train_generalist_agent('lidar')
        
    if args.agent in ['grayscale', 'all']:
        train_generalist_agent('grayscale')
