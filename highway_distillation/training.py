#!/usr/bin/env python3
"""
Unified Training Framework - Less is More

Simple, unified training for all phases. Eliminates 80% code duplication.
"""

import os
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from environments.urban_junction_env import UrbanJunctionEnv
from custom_policies import MultiModalActorCriticPolicy, ContextAwareActorCriticPolicy
from training.training_logger import SimpleLogger


class EpisodeLoggerCallback(BaseCallback):
    """Callback for logging episode results."""

    def __init__(self, episode_logger, verbose=0):
        super().__init__(verbose)
        self.episode_logger = episode_logger
        self.episode_reward = 0
        self.episode_length = 0

    def _on_step(self) -> bool:
        # Accumulate reward and length
        self.episode_reward += self.locals.get('rewards', [0])[0]
        self.episode_length += 1

        # Check if episode is done
        dones = self.locals.get('dones', [False])
        if dones[0]:  # Episode ended
            # Determine success (episode completed without collision)
            success = self.episode_reward > 0  # Simple success criteria

            # Log the episode
            self.episode_logger.log_episode(self.episode_reward, success)

            # Reset for next episode
            self.episode_reward = 0
            self.episode_length = 0

        return True


def create_environment(config):
    """Create environment with given configuration."""
    env_config = UrbanJunctionEnv.default_config()
    env_config.update(config)

    def make_env():
        env = UrbanJunctionEnv(env_config)
        # Skip Monitor for Phase 1 multi-modal due to compatibility issues
        if not (hasattr(env, 'config') and env.config.get("observation", {}).get("multi_modal", False)):
            env = Monitor(env, filename=os.path.join("outputs", "logs", f"{config.get('phase', 'unknown')}", "monitor.csv"))
        return env

    # Create vectorized environment for all phases
    env = DummyVecEnv([make_env])

    # Use VecNormalize for all phases to ensure consistent buffer sizing
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    return env


def create_model(model_config, env, phase_name='unknown'):
    """Create model with given configuration."""
    policy_class = {
        'MultiModal': MultiModalActorCriticPolicy,
        'ContextAware': ContextAwareActorCriticPolicy,
    }.get(model_config['policy'], MultiModalActorCriticPolicy)

    # ✅ NOW SAFE: Use PPO consistently for ALL phases (vectorized)
    model = PPO(
        policy_class,
        env,
        learning_rate=model_config.get('learning_rate', 3e-4),
        n_steps=model_config.get('n_steps', 2048),
        batch_size=model_config.get('batch_size', 64),
        n_epochs=model_config.get('n_epochs', 10),
        gamma=model_config.get('gamma', 0.99),
        gae_lambda=model_config.get('gae_lambda', 0.95),
        clip_range=model_config.get('clip_range', 0.2),
        ent_coef=model_config.get('ent_coef', 0.0),
        vf_coef=model_config.get('vf_coef', 0.5),
        max_grad_norm=model_config.get('max_grad_norm', 0.5),
        policy_kwargs=model_config.get('policy_kwargs', {}),
        verbose=0,
    )

    return model


def train_phase(phase_config):
    """Unified training function for any phase."""

    print(f"Starting {phase_config['name']} training...")

    # Create components
    env = create_environment(phase_config['env_config'])
    model = create_model(phase_config['model_config'], env, phase_config['name'])
    use_wandb = os.getenv('USE_WANDB', 'true').lower() == 'true'
    logger = SimpleLogger(phase_config['name'], use_wandb=use_wandb)

    # Create callbacks
    episode_callback = EpisodeLoggerCallback(episode_logger=logger)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000, phase_config['timesteps'] // 10),
        save_path=f"outputs/models/{phase_config['name']}",
        name_prefix=phase_config['name']
    )

    # Training loop with logging
    total_timesteps = phase_config['timesteps']
    model.learn(
        total_timesteps=total_timesteps,
        callback=[episode_callback, checkpoint_callback]
    )

    # Save final model and results
    os.makedirs(f"outputs/models/{phase_config['name']}", exist_ok=True)
    model.save(f"outputs/models/{phase_config['name']}/final")
    # Save VecNormalize if it exists (Phase 2 only)
    if hasattr(env, 'save'):
        env.save(f"outputs/models/{phase_config['name']}/vec_normalize.pkl")

    logger.save_results()

    print(f"{phase_config['name']} training completed!")
    return model


# Phase configurations - single source of truth
PHASE_CONFIGS = {
    'phase1': {
        'name': 'phase1',
        'timesteps': 10000 if os.getenv('TEST_MODE') else 1000000,  # 10K for testing, 1M for real
        'env_config': {
            'observation': {
                'type': 'Kinematics',
                'multi_modal': True,
                'lidar_rays': 64,
                'lidar_range': 50.0,
                'visual_width': 84,
                'visual_height': 84,
                'vehicles_count': 8,
                'features': ['presence', 'x', 'y', 'vx', 'vy'],
                'normalize': True,
            },
            'vehicles_count': 8,
            'stage_mode': 'deterministic',
            'antagonistic_vehicles': False,
            'duration': 200,
            'modality_dropout': 0.2,  # 20% chance to drop modalities for robustness
        },
        'model_config': {
            'policy': 'MultiModal',
            'learning_rate': 3e-4,
            'policy_kwargs': {'features_extractor_kwargs': {'kinematics_dim': 40, 'lidar_dim': 64, 'visual_dim': (84, 84, 1), 'fusion_dim': 512}},
        }
    },

    'phase2': {
        'name': 'phase2',
        'timesteps': 20000 if os.getenv('TEST_MODE') else 2000000,  # 20K for testing, 2M for real
        'env_config': {
            'vehicles_count': 8,
            'stage_mode': 'random',
            'antagonistic_vehicles': False,
            'duration': 200,
        },
        'model_config': {
            'policy': 'ContextAware',
            'learning_rate': 3e-4,
            'policy_kwargs': {'features_extractor_kwargs': {'kinematics_features': 40, 'fusion_dim': 256}},
        }
    },

    'phase3': {
        'name': 'phase3',
        'timesteps': 60000 if os.getenv('TEST_MODE') else 9000000,  # 60K for testing, 9M for real
        'env_config': {
            'vehicles_count': 8,
            'stage_mode': 'curriculum',
            'antagonistic_vehicles': True,
            'annoyance_level': 0.7,
            'duration': 200,
        },
        'model_config': {
            'policy': 'ContextAware',
            'learning_rate': 3e-4,
            'policy_kwargs': {'features_extractor_kwargs': {'kinematics_features': 40, 'fusion_dim': 256}},
        }
    }
}


def train_phase1():
    """Train Phase 1: Multi-modal foundation."""
    return train_phase(PHASE_CONFIGS['phase1'])


def train_phase2():
    """Train Phase 2: Context-aware policies."""
    return train_phase(PHASE_CONFIGS['phase2'])


def train_phase3():
    """Train Phase 3: Curriculum learning."""
    return train_phase(PHASE_CONFIGS['phase3'])


if __name__ == "__main__":
    # Allow direct execution for testing
    import sys
    if len(sys.argv) > 1:
        phase = sys.argv[1]
        if phase in PHASE_CONFIGS:
            train_phase(PHASE_CONFIGS[phase])
        else:
            print(f"Unknown phase: {phase}")
    else:
        print("Usage: python training.py phase1|phase2|phase3")
