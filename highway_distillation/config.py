#!/usr/bin/env python3
"""
Centralized Configuration

All training parameters, environment settings, and model configurations.
"""

import os

# Training settings
TRAINING = {
    'learning_rate': 3e-4,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.0,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'n_steps': 2048,
}

# Environment settings
ENVIRONMENT = {
    'vehicles_count': 8,
    'duration': 200,
    'lidar_rays': 64,
    'lidar_range': 50.0,
    'visual_size': (84, 84),
    'frame_stack': 2,
}

# Model architectures
POLICIES = {
    # Legacy policies
    'MultiModal': {
        'features_dim': 512,
        'kinematics_dim': 75,  # 15 vehicles × 5 features
        'lidar_dim': 64,
        'visual_dim': (84, 84, 1),
    },
    'ContextAware': {
        'features_dim': 256,
        'kinematics_dim': 75,
        'context_dim': 3,  # One-hot encoded scenario type
    },

    # Generalist Agent Policies (Optimized)
    'LidarMlp': {
        'net_arch': [256, 256],  # Simple but deep MLP
        'activation_fn': 'tanh',
    },
    'GrayCnn': {
        'net_arch': [],  # Use default CNN architecture
        'activation_fn': 'relu',
    }
}

# Phase-specific overrides
PHASES = {
    # Legacy phases (for compatibility)
    'phase1': {
        'timesteps': 10000 if os.getenv('TEST_MODE') else 1000000,
        'env_overrides': {
            'multi_modal': True,
            'stage_mode': 'curriculum',         # Non-deterministic curriculum learning
            'antagonistic_vehicles': True,      # Enable early for robustness
            'annoyance_level': 0.3,             # Mild difficulty to start
            'num_antagonistic': 2,              # Add parameter for vehicle count
            'modality_dropout': 0.15,           # Start with mild dropout
            'kinematics_dropout_allowed': False, # Don't drop kinematics yet
        },
        'policy': 'MultiModal',
    },

    'phase2': {
        'timesteps': 20000 if os.getenv('TEST_MODE') else 2000000,
        'env_overrides': {
            'stage_mode': 'random',
            'antagonistic_vehicles': True,
            'annoyance_level': 0.5,             # Medium difficulty
            'num_antagonistic': 3,
            'modality_dropout': 0.25,           # Increase dropout
            'kinematics_dropout_allowed': True, # Allow kinematics dropout
        },
        'policy': 'ContextAware',
    },

    'phase3': {
        'timesteps': 60000 if os.getenv('TEST_MODE') else 9000000,
        'env_overrides': {
            'stage_mode': 'curriculum',
            'antagonistic_vehicles': True,
            'annoyance_level': 0.9,             # Maximum difficulty
            'num_antagonistic': 4,
            'modality_dropout': 0.3,            # High dropout for robustness
            'kinematics_dropout_allowed': True, # Full sensor independence
        },
        'policy': 'ContextAware',
    },

    # Generalist Agent Phases (Optimized for project goals)
    'lidar_generalist': {
        'timesteps': 300000,  # 300K steps for fast training
        'num_envs': 8,         # Parallel environments for speed
        'use_lidar_only': True,
        'env_overrides': {
            'stage': 13,         # Mixed scenarios (highway + merge + intersection)
            'lidar_rays': 32,    # Optimized: reduced from 64
            'vehicles_count': 10,
            'duration': 200,
        },
        'policy': 'LidarMlp'
    },

    'gray_generalist': {
        'timesteps': 500000,     # 500K steps - vision needs more data
        'num_envs': 4,           # Fewer parallel envs (heavier rendering)
        'use_grayscale_only': True,
        'env_overrides': {
            'stage': 13,         # Mixed scenarios
            'visual_width': 64,  # Optimized: reduced from 84
            'visual_height': 64,
            'stack_size': 4,
            'vehicles_count': 8,
            'duration': 200,
        },
        'policy': 'GrayCnn'
    }
}

# File paths
PATHS = {
    'outputs': 'outputs',
    'models': 'outputs/models',
    'logs': 'outputs/logs',
    'plots': 'outputs/plots',
    'data': 'outputs/data',
}
