#!/usr/bin/env python3
"""
Phase 3: The Multi-Stage Training Curriculum (Optimized)

This script implements the complete curriculum learning approach:
- Stage B: Generalization (Highway Certification) - Apply skills to unseen sequences
- Stage C: Resilience (Defensive Driving Course) - Handle antagonistic traffic

OPTIMIZATIONS APPLIED:
- Vehicles: 8 throughout (vs 15) - 50% less traffic, faster simulation
- Frame Stack: 2 (vs 4) - 50% smaller observations
- Training Steps: 9M total (vs 22M) - 59% less training time
- Consistent parameters across phases for smooth curriculum

Key Innovation: Progressive difficulty increase with adaptive annoyance levels.
The agent learns to handle "jerks" through curriculum learning rather than
starting with chaos from day one.

Training Flow:
1. Stage B: Load Phase 2 model → Train on randomized sequences (no antagonists) - 3M steps
2. Stage C: Load Stage B model → Train with antagonistic vehicles + adaptive difficulty - 6M steps
"""

import os
import logging
import numpy as np
from pathlib import Path
from gymnasium import spaces, Env

# Stable Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure

# Custom policies
from custom_policies import ContextAwareActorCriticPolicy

# Highway environment
from environments.urban_junction_env import UrbanJunctionEnv

# Enhanced training logger
from training_logger import create_phase3_logger, create_episode_callback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PHASE3_MODEL_DIR = "models/phase3"
LOGS_DIR = "logs/phase3"

class AdaptiveDifficultyCallback:
    """
    Callback for adaptive difficulty in Stage C.

    Monitors agent performance and increases annoyance level when the agent
    demonstrates mastery, creating the "staircase" learning pattern.
    """

    def __init__(self, eval_env, check_freq=10000, patience=3, threshold_multiplier=1.2):
        """
        Args:
            eval_env: Environment for evaluation (should have adaptive difficulty enabled)
            check_freq: How often to check performance (in timesteps)
            patience: How many checks to wait before increasing difficulty
            threshold_multiplier: Performance threshold for difficulty increase
        """
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.patience = patience
        self.threshold_multiplier = threshold_multiplier

        self.last_check_step = 0
        self.consecutive_good_performance = 0
        self.current_threshold = 15.0  # Initial threshold
        self.baseline_annoyance = 0.1   # Starting annoyance level

    def __call__(self, locals, globals):
        """Callback called during training."""
        # Get current timestep
        current_step = locals.get('self').num_timesteps

        # Check performance periodically
        if current_step - self.last_check_step >= self.check_freq:
            self._evaluate_and_adjust(current_step)

    def _evaluate_and_adjust(self, current_step):
        """Evaluate agent performance and adjust difficulty."""
        # Run evaluation episodes
        episode_rewards = []
        episode_lengths = []

        for _ in range(5):  # Evaluate over 5 episodes
            obs, info = self.eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done and episode_length < 500:  # Max episode length
                action, _ = locals['self'].predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)

        logger.info(".2f"
                   ".1f"
                   ".2f")

        # Check if performance exceeds threshold
        if avg_reward > self.current_threshold:
            self.consecutive_good_performance += 1
            logger.info(f"Good performance streak: {self.consecutive_good_performance}/{self.patience}")

            if self.consecutive_good_performance >= self.patience:
                # Increase difficulty
                old_annoyance = self.eval_env.env.env.annoyance_level
                new_annoyance = min(1.0, old_annoyance + 0.1)  # Increase by 0.1, max 1.0

                self.eval_env.env.env.annoyance_level = new_annoyance
                self.current_threshold *= self.threshold_multiplier  # Raise threshold

                # Update all antagonistic vehicles
                for vehicle in self.eval_env.env.env.road.vehicles[1:]:
                    if hasattr(vehicle, 'annoyance_level'):
                        vehicle.annoyance_level = new_annoyance

                logger.info(".2f"
                           ".2f"
                           ".1f")

                self.consecutive_good_performance = 0  # Reset counter
        else:
            self.consecutive_good_performance = 0  # Reset on poor performance

        self.last_check_step = current_step

def create_stage_b_env():
    """
    Create Stage B environment: Generalization (Highway Certification)

    Goal: Apply Phase 2 context-aware skills to completely randomized,
    unseen stage sequences. No antagonistic vehicles yet.
    """
    config = UrbanJunctionEnv.default_config()

    config.update({
        # Observation: Kinematics + Context (from Phase 2)
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,  # More traffic for generalization
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
            "normalize": True,
        },

        # Environment: Randomized sequences for generalization
        "lanes_count": 2,
        "vehicles_count": 15,
        "vehicles_density": 1.0,
        "duration": 250,  # Longer episodes

        # Random stage sequences (key change from Phase 2)
        "stage_mode": "random",  # Fully randomized for generalization
        "min_stages": 3,
        "max_stages": 6,  # More complex sequences
        "stage_length_range": [120, 220],

        # No antagonistic vehicles (introduce in Stage C)
        "antagonistic_vehicles": False,

        # Reward structure
        "normalize_reward": True,
        "collision_reward": 1.0,
        "speed_reward": 0.4,
        "speed_penalty_scale": 0.3,
        "progress_reward": 0.2,
        "off_road_penalty": 0.3,
        "success_reward": 2.0,
        "stage_completion_reward": 0.5,
        "reward_speed_range": [20, 30],

        "offroad_terminal": False,
    })

    # Create and wrap environment
    env = UrbanJunctionEnv(config)
    env = Monitor(env, filename=os.path.join(LOGS_DIR, "stage_b_monitor.csv"))
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    env = VecFrameStack(env, n_stack=2, channels_order='last')  # Optimized: 2 frames for speed

    # Add context wrapper (same as Phase 2)
    env = modify_observation_space_for_context(env)

    return env

def create_stage_c_env():
    """
    Create Stage C environment: Resilience (Defensive Driving Course)

    Goal: Handle antagonistic traffic with adaptive difficulty.
    The annoyance level increases as the agent proves competent.
    """
    config = UrbanJunctionEnv.default_config()

    config.update({
        # Observation: Kinematics + Context (optimized)
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 8,   # Optimized: Same as Phase 1-2 for consistency
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
            "normalize": True,
        },

        # Environment: Full randomization
        "lanes_count": 2,
        "vehicles_count": 8,   # Optimized: Same as Phase 1-2 for consistency
        "vehicles_density": 1.0,
        "duration": 300,  # Longest episodes

        # Random sequences
        "stage_mode": "random",
        "min_stages": 3,
        "max_stages": 6,
        "stage_length_range": [120, 220],

        # Antagonistic vehicles (key feature of Stage C)
        "antagonistic_vehicles": True,
        "swerving_vehicle_ratio": 0.25,
        "cutoff_vehicle_ratio": 0.20,
        "random_vehicle_ratio": 0.15,

        # Adaptive difficulty (the curriculum heart)
        "adaptive_difficulty": True,
        "annoyance_level": 0.1,  # Start mild, increase automatically
        "performance_threshold": 15.0,
        "max_annoyance": 1.0,

        # Reward structure
        "normalize_reward": True,
        "collision_reward": 1.0,
        "speed_reward": 0.4,
        "speed_penalty_scale": 0.3,
        "progress_reward": 0.2,
        "traffic_light_penalty": 0.4,
        "traffic_light_reward": 0.1,
        "off_road_penalty": 0.3,
        "success_reward": 2.0,
        "stage_completion_reward": 0.5,
        "reward_speed_range": [20, 30],

        "offroad_terminal": False,
    })

    # Create and wrap environment
    env = UrbanJunctionEnv(config)
    env = Monitor(env, filename=os.path.join(LOGS_DIR, "stage_c_monitor.csv"))
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    env = VecFrameStack(env, n_stack=2, channels_order='last')  # Optimized: 2 frames for speed

    # Add context wrapper
    env = modify_observation_space_for_context(env)

    return env

def modify_observation_space_for_context(env):
    """Add context to observation space (same as Phase 2)."""
    class ContextAwareUrbanEnv(Env):
        def __init__(self, env):
            super().__init__()
            self.env = env
            self._update_observation_space()
            self.action_space = env.action_space

        def _update_observation_space(self):
            original_space = self.env.observation_space
            context_dims = 3
            new_shape = (original_space.shape[0], original_space.shape[1] + context_dims)

            self.observation_space = spaces.Box(
                low=np.concatenate([original_space.low, np.zeros((original_space.shape[0], context_dims))], axis=1),
                high=np.concatenate([original_space.high, np.ones((original_space.shape[0], context_dims))], axis=1),
                dtype=original_space.dtype
            )

        def _get_context_one_hot(self, phase):
            context_map = {'highway': 0, 'merge': 1, 'intersection': 2}
            one_hot = np.zeros(3)
            if phase in context_map:
                one_hot[context_map[phase]] = 1.0
            return one_hot

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            phase = info.get('phase', 'highway')
            context = self._get_context_one_hot(phase)
            context_expanded = np.tile(context, (obs.shape[0], 1))
            obs_with_context = np.concatenate([obs, context_expanded], axis=1)
            return obs_with_context, info

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            phase = info.get('phase', 'highway')
            context = self._get_context_one_hot(phase)
            context_expanded = np.tile(context, (obs.shape[0], 1))
            obs_with_context = np.concatenate([obs, context_expanded], axis=1)
            return obs_with_context, reward, terminated, truncated, info

        def __getattr__(self, name):
            return getattr(self.env, name)

    return ContextAwareUrbanEnv(env)

def train_stage_b(phase2_model_path: str, test_mode: bool = False):
    """Train Stage B: Generalization on randomized sequences."""
    logger.info("=== STAGE B: Generalization (Highway Certification) ===")
    logger.info("- Goal: Apply context-aware skills to unseen sequences")
    logger.info("- Environment: Fully randomized stages, no antagonists")
    logger.info("- Training: 7M timesteps on complex sequences")

    # Create Stage B environment
    env = create_stage_b_env()

    # Load Phase 2 model
    logger.info(f"Loading Phase 2 model from: {phase2_model_path}")
    model = PPO.load(phase2_model_path, env=env)

    # Create episode progress callback
    episode_callback = create_episode_callback("phase3")

    # Configure callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=150000,
        save_path=os.path.join(PHASE3_MODEL_DIR, "stage_b"),
        name_prefix="ppo_stage_b",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    callbacks = [checkpoint_callback, episode_callback]
    callback_list = CallbackList(callbacks)

    # Configure logger
    sb3_logger = configure(os.path.join(LOGS_DIR, "stage_b"), ["stdout", "csv"])

    # Stage B training
    timesteps = 30_000 if test_mode else 3_000_000
    log_interval = 1 if test_mode else 10

    model.set_logger(sb3_logger)
    model.learn(
        total_timesteps=timesteps,
        callback=callback_list,
        log_interval=log_interval,
        progress_bar=True
    )

    # Save Stage B model
    stage_b_path = os.path.join(PHASE3_MODEL_DIR, "ppo_stage_b_final")
    model.save(stage_b_path)
    env.env.save(os.path.join(PHASE3_MODEL_DIR, "vec_normalize_stage_b.pkl"))

    logger.info(f"✓ Stage B completed! Model saved to: {stage_b_path}")

    env.close()
    return stage_b_path

def train_stage_c(stage_b_model_path: str, test_mode: bool = False):
    """Train Stage C: Resilience with antagonistic traffic."""
    logger.info("=== STAGE C: Resilience (Defensive Driving Course) ===")
    logger.info("- Goal: Handle antagonistic traffic with adaptive difficulty")
    logger.info("- Environment: Antagonistic vehicles + curriculum learning")
    logger.info("- Training: 15M timesteps with increasing annoyance")

    # Create Stage C environment
    env = create_stage_c_env()

    # Create separate evaluation environment for adaptive difficulty
    eval_env = create_stage_c_env()

    # Load Stage B model
    logger.info(f"Loading Stage B model from: {stage_b_model_path}")
    model = PPO.load(stage_b_model_path, env=env)

    # Configure adaptive difficulty callback
    adaptive_callback = AdaptiveDifficultyCallback(
        eval_env=eval_env,
        check_freq=25000,  # Check every 25k steps
        patience=2,        # 2 good evaluations before difficulty increase
        threshold_multiplier=1.1  # 10% threshold increase
    )

    # Create episode progress callback
    episode_callback = create_episode_callback("phase3")

    # Configure checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=200000,
        save_path=os.path.join(PHASE3_MODEL_DIR, "stage_c"),
        name_prefix="ppo_stage_c",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    callbacks = [checkpoint_callback, adaptive_callback, episode_callback]
    callback_list = CallbackList(callbacks)

    # Configure logger
    sb3_logger = configure(os.path.join(LOGS_DIR, "stage_c"), ["stdout", "csv"])

    # Stage C training (longest and most challenging)
    timesteps = 30_000 if test_mode else 6_000_000
    log_interval = 1 if test_mode else 10

    model.set_logger(sb3_logger)
    model.learn(
        total_timesteps=timesteps,
        callback=callback_list,
        log_interval=log_interval,
        progress_bar=True
    )

    # Save final Stage C model
    stage_c_path = os.path.join(PHASE3_MODEL_DIR, "ppo_stage_c_final")
    model.save(stage_c_path)
    env.env.save(os.path.join(PHASE3_MODEL_DIR, "vec_normalize_stage_c.pkl"))

    logger.info(f"✓ Stage C completed! Model saved to: {stage_c_path}")
    logger.info("✓ Agent can now handle adversarial traffic!")
    logger.info("✓ Ready for Phase 4: Rigorous Validation")

    env.close()
    eval_env.close()

def setup_directories():
    """Create necessary directories for Phase 3."""
    directories = [
        PHASE3_MODEL_DIR,
        LOGS_DIR,
        os.path.join(PHASE3_MODEL_DIR, "stage_b"),
        os.path.join(PHASE3_MODEL_DIR, "stage_c"),
        os.path.join(LOGS_DIR, "stage_b"),
        os.path.join(LOGS_DIR, "stage_c")
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    logger.info(f"Created Phase 3 directories")

def train_phase3(phase2_model_path: str = "models/phase2/ppo_context_aware_final.zip"):
    """Execute complete Phase 3 curriculum: Stage B + Stage C."""

    # Initialize enhanced logger
    train_logger = create_phase3_logger()

    train_logger.logger.info("=== PHASE 3: The Multi-Stage Training Curriculum ===")
    train_logger.logger.info("Stage B: Generalization → Stage C: Resilience")
    train_logger.logger.info("- Curriculum learning prevents catastrophic forgetting")
    train_logger.logger.info("- Progressive difficulty with adaptive annoyance")
    train_logger.logger.info("=" * 60)

    # Log training configuration
    config = {
        "phase": "Phase 3: Curriculum Learning (Stage B + Stage C)",
        "total_timesteps": "9,000,000 (60,000 test)",
        "stage_b_timesteps": "3,000,000 (30,000 test)",
        "stage_c_timesteps": "6,000,000 (30,000 test)",
        "environment": "Curriculum progression",
        "antagonistic_vehicles": "Stage C only",
        "adaptive_difficulty": "Enabled in Stage C",
        "vehicles_count": 8,
        "network_architecture": "ContextAwareActorCriticPolicy",
        "transfer_learning": f"From {phase2_model_path}",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 5
    }
    train_logger.log_training_start(config)

    setup_directories()

    # Test mode configuration
    test_mode = os.environ.get("TEST_MODE", "false").lower() == "true"

    try:
        # Stage B: Generalization
        train_logger.logger.info("Stage B: Generalization Training")
        stage_b_model = train_stage_b(phase2_model_path, test_mode)

        # Stage C: Resilience
        train_logger.logger.info("Stage C: Resilience Training")
        train_stage_c(stage_b_model, test_mode)

        train_logger.logger.info("Phase 3 curriculum completed successfully!")
        train_logger.logger.info("Agent mastered generalization and resilience")
        train_logger.logger.info("Ready for Phase 4 validation!")

        # Create training plots and final statistics
        # Note: Phase 3 has multiple stages, so we use the episode callback from the last stage
        # For now, create plots from the main logger
        train_logger.create_training_plots()
        stats = train_logger.get_summary_stats()
        if stats:
            total_timesteps = 60_000 if test_mode else 9_000_000
            train_logger.log_training_complete(total_timesteps, stats.get('total_episodes', 0))

    except Exception as e:
        train_logger.log_error(e, "Phase 3 training")
        raise

if __name__ == "__main__":
    import sys
    phase2_path = sys.argv[1] if len(sys.argv) > 1 else "models/phase2/ppo_context_aware_final.zip"

    train_phase3(phase2_path)
