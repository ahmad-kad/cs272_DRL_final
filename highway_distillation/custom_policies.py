#!/usr/bin/env python3
"""
Custom Policies for Context-Aware Reinforcement Learning

This module implements custom policy architectures that can condition behavior
on contextual information, enabling robust generalization across different
driving scenarios (highway, merge, intersection).

Key Features:
- Context-Aware Actor-Critic Policy: Separates kinematics processing from context
- Modular architecture for easy extension
- PyTorch-based implementation compatible with Stable-Baselines3
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, List, Tuple, Type, Any, Optional
import numpy as np


class MultiModalFeaturesExtractor(BaseFeaturesExtractor):
    """
    Multi-modal feature extractor for kinematics + lidar + visual observations.

    Architecture:
    - Kinematics branch: Processes vehicle position/velocity data
    - Lidar branch: Processes distance measurements from lidar sensor
    - Visual branch: Processes grayscale image data
    - Fusion: Combines all modalities for joint representation learning
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        kinematics_dim: int = 75,        # 15 vehicles × 5 features
        lidar_dim: int = 64,             # 64 lidar rays
        visual_dim: tuple = (84, 84, 1), # Visual dimensions
        hidden_dim: int = 256,
        fusion_dim: int = 512,
    ):
        """
        Args:
            observation_space: Dict space with 'kinematics', 'lidar', 'visual' keys
            kinematics_dim: Flattened kinematics features
            lidar_dim: Number of lidar rays
            visual_dim: Visual observation dimensions (H, W, C)
            hidden_dim: Hidden layer size for modality processing
            fusion_dim: Hidden layer size for fused representation
        """
        super().__init__(observation_space, features_dim=fusion_dim)

        # Kinematics processing branch
        self.kinematics_net = nn.Sequential(
            nn.Linear(kinematics_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Lidar processing branch (1D convolution for distance patterns)
        self.lidar_net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
        )

        # Visual processing branch (CNN for image features)
        self.visual_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
        )

        # Fusion network (combines all modalities)
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
        )

        # Store dimensions for reference
        self.kinematics_dim = kinematics_dim
        self.lidar_dim = lidar_dim
        self.visual_dim = visual_dim

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process multi-modal observations through the fusion architecture.

        Args:
            observations: Dict with 'kinematics', 'lidar', 'visual' tensors

        Returns:
            Fused feature representation
        """
        # Process kinematics branch
        kinematics_obs = observations["kinematics"]
        if kinematics_obs.dim() > 2:
            kinematics_obs = kinematics_obs.view(kinematics_obs.size(0), -1)
        kinematics_features = self.kinematics_net(kinematics_obs)

        # Process lidar branch
        lidar_obs = observations["lidar"]
        if lidar_obs.dim() == 2:  # Add channel dimension for Conv1d
            lidar_obs = lidar_obs.unsqueeze(1)
        lidar_features = self.lidar_net(lidar_obs)

        # Process visual branch
        visual_obs = observations["visual"]
        visual_features = self.visual_net(visual_obs)

        # Fuse all modalities
        combined = torch.cat([kinematics_features, lidar_features, visual_features], dim=1)
        fused_features = self.fusion_net(combined)

        return fused_features


class ContextAwareFeaturesExtractor(BaseFeaturesExtractor):
    """
    Context-aware feature extractor for kinematics + context observations.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        kinematics_features: int = 75,  # Flattened kinematics features
        context_features: int = 3,      # Context categories
        hidden_dim: int = 128,
        fusion_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim=fusion_dim)

        # Kinematics processing
        self.kinematics_net = nn.Sequential(
            nn.Linear(kinematics_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Context processing
        self.context_net = nn.Sequential(
            nn.Linear(context_features, hidden_dim // 2),
            nn.ReLU(),
        )

        # Fusion
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        kinematics = observations["kinematics"]
        if kinematics.dim() > 2:
            kinematics = kinematics.view(kinematics.size(0), -1)
        k_features = self.kinematics_net(kinematics)

        context = observations["context"]
        c_features = self.context_net(context)

        combined = torch.cat([k_features, c_features], dim=1)
        return self.fusion_net(combined)


class MultiModalActorCriticPolicy(ActorCriticPolicy):
    """
    Multi-Modal Actor-Critic Policy for kinematics + lidar + visual observations.

    This policy extends Stable-Baselines3's ActorCriticPolicy to handle
    multi-modal observations combining ground truth, sensor, and visual data.

    Key innovation: Fuses information from multiple sensor modalities for
    robust autonomous driving behavior.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.Tanh,
        ortho_init=True,
        use_sde=False,
        log_std_init=0.0,
        full_std=True,
        use_expln=False,
        squash_output=False,
        features_extractor_class=MultiModalFeaturesExtractor,
        features_extractor_kwargs=None,
        share_features_extractor=True,
        normalize_images=True,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=None,
    ):
        # Set default network architecture if not provided
        if net_arch is None:
            net_arch = dict(pi=[256, 256], vf=[256, 256])

        # Set default features extractor kwargs
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )


class ContextAwareActorCriticPolicy(ActorCriticPolicy):
    """
    Context-Aware Actor-Critic Policy for multi-stage driving scenarios.

    This policy extends Stable-Baselines3's ActorCriticPolicy to handle
    structured observations with kinematics and context components.

    Key innovation: The agent learns both general driving skills and
    context-specific adaptations simultaneously.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.Tanh,
        ortho_init=True,
        use_sde=False,
        log_std_init=0.0,
        full_std=True,
        use_expln=False,
        squash_output=False,
        features_extractor_class=ContextAwareFeaturesExtractor,
        features_extractor_kwargs=None,
        share_features_extractor=True,
        normalize_images=True,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=None,
    ):
        # Set default network architecture if not provided
        if net_arch is None:
            net_arch = dict(pi=[256, 256], vf=[256, 256])

        # Set default features extractor kwargs
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )


class ObservationWrapper:
    """
    Wrapper to convert flat observations to structured dict format.

    This wrapper transforms the environment's flat observation space into
    a structured Dict space that separates kinematics from context.
    """

    def __init__(self, env, context_dims: int = 3):
        """
        Args:
            env: Original environment with flat observation space
            context_dims: Number of context categories
        """
        self.env = env
        self.context_dims = context_dims

        # Create structured observation space
        original_obs_space = env.observation_space

        # Assume first N dimensions are context (one-hot), rest are kinematics
        self.observation_space = spaces.Dict({
            "kinematics": spaces.Box(
                low=original_obs_space.low[:, :-context_dims],
                high=original_obs_space.high[:, :-context_dims],
                dtype=original_obs_space.dtype
            ),
            "context": spaces.Box(
                low=np.zeros(context_dims),
                high=np.ones(context_dims),
                dtype=original_obs_space.dtype
            )
        })

    def __getattr__(self, name):
        """Delegate all other attributes to wrapped environment."""
        return getattr(self.env, name)

    def reset(self, **kwargs):
        """Reset environment and structure observation."""
        obs, info = self.env.reset(**kwargs)
        return self._structure_observation(obs), info

    def step(self, action):
        """Step environment and structure observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._structure_observation(obs), reward, terminated, truncated, info

    def _structure_observation(self, flat_obs):
        """
        Convert flat observation to structured dict.

        Assumes: [kinematics_features..., context_one_hot...]
        """
        # Split observation into kinematics and context
        kinematics = flat_obs[:, :-self.context_dims]
        context = flat_obs[:, -self.context_dims:]

        return {
            "kinematics": kinematics,
            "context": context
        }


def create_context_aware_policy(
    observation_space: spaces.Space,
    action_space: spaces.Space,
    lr_schedule,
    **kwargs
) -> ContextAwareActorCriticPolicy:
    """
    Factory function to create context-aware policy.

    This is the recommended way to instantiate the custom policy
    for use with Stable-Baselines3 algorithms.
    """
    return ContextAwareActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        **kwargs
    )


def create_multimodal_policy(
    observation_space: spaces.Space,
    action_space: spaces.Space,
    lr_schedule,
    **kwargs
) -> MultiModalActorCriticPolicy:
    """
    Factory function to create multi-modal policy.

    This creates a policy that processes kinematics + lidar + visual observations
    for robust autonomous driving behavior.
    """
    return MultiModalActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        **kwargs
    )

