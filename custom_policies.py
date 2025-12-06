# custom_policies.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import math


class SimpleSpatialExtractor(BaseFeaturesExtractor):
    """
    CNN-based spatial features extractor for vehicle kinematic data.

    Processes vehicles as a spatial grid where each "pixel" represents a vehicle
    with 7 feature channels. Much simpler than transformer while capturing
    spatial relationships effectively.

    Input: (batch_size, n_vehicles, features_per_vehicle) - e.g., (batch, 15, 7)
    Output: (batch_size, features_dim)
    """

    def __init__(self, observation_space, features_dim=128):
        """
        Args:
            observation_space: Gym observation space
            features_dim: Dimension of output features
        """
        super().__init__(observation_space, features_dim)

        # Observation space: (n_vehicles, features_per_vehicle)
        # e.g., (15 vehicles, 7 features: [presence, x, y, vx, vy, cos_h, sin_h])
        self.n_vehicles = observation_space.shape[0]
        self.features_per_vehicle = observation_space.shape[1]

        # CNN architecture for spatial vehicle processing
        self.conv_net = nn.Sequential(
            # Input: (batch, 1, n_vehicles, features_per_vehicle)
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global spatial pooling
            nn.Flatten(),  # (batch, 64) -> flatten to vector
            nn.Linear(64, features_dim),
            nn.ReLU()
        )

        # Optional: Add a small MLP for final feature processing
        self.feature_net = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim)
        )

    def forward(self, observations):
        """
        Args:
            observations: (batch_size, n_vehicles, features_per_vehicle)
                         e.g., (batch, 15, 7)

        Returns:
            features: (batch_size, features_dim)
        """
        batch_size = observations.shape[0]

        # Reshape for CNN: (batch, n_vehicles, features) -> (batch, 1, n_vehicles, features)
        # Treat vehicles as a "spatial grid" with features as channels
        x = observations.unsqueeze(1)  # Add channel dimension: (batch, 1, 15, 7)

        # Apply convolutional processing
        conv_features = self.conv_net(x)  # (batch, features_dim)

        # Optional final feature processing
        final_features = self.feature_net(conv_features)

        return final_features


class EgoAttentionExtractor(BaseFeaturesExtractor):
    """
    Ego-Attention feature extractor for highway environments.

    Processes vehicles as a permutation-invariant set using attention mechanism.
    Focuses on ego-centric relationships and identifies immediate threats.

    Based on "Social Attention" and "Ego Attention" architectures from highway-env literature.

    Input: (batch_size, n_vehicles, features_per_vehicle)
           e.g., (batch, 15, 7) for [presence, x, y, vx, vy, cos_h, sin_h]
    Output: (batch_size, features_dim)
    """

    def __init__(self, observation_space, features_dim=256, n_heads=4, n_layers=2):
        """
        Args:
            observation_space: Gym observation space
            features_dim: Dimension of output features
            n_heads: Number of attention heads
            n_layers: Number of attention layers
        """
        super().__init__(observation_space, features_dim)

        # Observation space: (n_vehicles, features_per_vehicle)
        self.n_vehicles = observation_space.shape[0]
        self.features_per_vehicle = observation_space.shape[1]

        # Feature preprocessing
        self.vehicle_encoder = nn.Sequential(
            nn.Linear(self.features_per_vehicle, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64)
        )

        # Ego vehicle is always at index 0 (first vehicle in list)
        self.ego_encoder = nn.Sequential(
            nn.Linear(self.features_per_vehicle, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64)
        )

        # Multi-head attention for vehicle-vehicle interactions
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=64, num_heads=n_heads, batch_first=True)
            for _ in range(n_layers)
        ])

        # Layer norms for attention residual connections
        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(64) for _ in range(n_layers)
        ])

        # Ego-attention: focus on how other vehicles relate to ego
        self.ego_attention = nn.MultiheadAttention(
            embed_dim=64, num_heads=n_heads, batch_first=True
        )

        # Final aggregation and feature extraction
        # attended_features is (batch, n_vehicles, 64), so we flatten to (batch, n_vehicles * 64)
        self.aggregator = nn.Sequential(
            nn.Linear(64 * self.n_vehicles, 256),  # Concatenate all attended vehicle features
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim)
        )

        # Threat assessment module (identifies immediate dangers)
        self.threat_detector = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # Output threat features
        )

        # Final projection to ensure correct output dimension
        # aggregated_features (features_dim) + threat_features (32) = features_dim + 32
        self.final_projection = nn.Linear(features_dim + 32, features_dim)

    def forward(self, observations):
        """
        Args:
            observations: (batch_size, n_vehicles, features_per_vehicle)

        Returns:
            features: (batch_size, features_dim)
        """
        batch_size = observations.shape[0]

        # Encode all vehicles
        vehicle_features = self.vehicle_encoder(observations)  # (batch, n_vehicles, 64)

        # Special encoding for ego vehicle (first in list)
        ego_features = self.ego_encoder(observations[:, 0:1, :])  # (batch, 1, 64)

        # Vehicle-to-vehicle attention (social interactions)
        attended_features = vehicle_features
        for attention, norm in zip(self.attention_layers, self.attention_norms):
            # Self-attention among vehicles
            attn_out, _ = attention(attended_features, attended_features, attended_features)
            attended_features = norm(attended_features + attn_out)  # Residual connection

        # Ego-centric attention: how does ego relate to all other vehicles?
        # Use ego features as query, all vehicles as keys/values
        ego_attn_out, ego_attn_weights = self.ego_attention(
            ego_features, attended_features, attended_features
        )

        # Flatten attended features for aggregation
        # attended_features shape: (batch, n_vehicles, 64) -> (batch, n_vehicles * 64)
        combined_features = attended_features.view(batch_size, -1)

        # Final feature aggregation
        aggregated_features = self.aggregator(combined_features)

        # Threat assessment (optional, can be used for additional processing)
        threat_features = self.threat_detector(aggregated_features)

        # Combine main features with threat assessment
        # aggregated_features is (batch, features_dim), threat_features is (batch, 32)
        combined_features = torch.cat([aggregated_features, threat_features], dim=-1)

        # Project to final feature dimension
        final_features = self.final_projection(combined_features)

        return final_features