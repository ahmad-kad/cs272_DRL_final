# custom_policies.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Transformer-based features extractor for vehicle kinematic data.

    Processes kinematic observations (presence, position, velocity, heading) using
    attention mechanisms to capture relationships between vehicles.

    Input: (batch_size, n_vehicles, features_per_vehicle)
    Output: (batch_size, features_dim)
    """

    def __init__(self, observation_space, features_dim=128, n_heads=4, n_layers=2):
        """
        Args:
            observation_space: Gym observation space
            features_dim: Dimension of output features
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
        """
        super().__init__(observation_space, features_dim)

        # Assume kinematic observation space: (n_vehicles, features_per_vehicle)
        # From crazy_driver_env: features = ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]
        self.n_vehicles = observation_space.shape[0]
        self.features_per_vehicle = observation_space.shape[1]

        # Vehicle feature embedding
        self.vehicle_embedding = nn.Linear(self.features_per_vehicle, features_dim // 2)

        # Positional encoding for vehicle positions
        self.position_embedding = nn.Embedding(self.n_vehicles, features_dim // 2)

        # Multi-head attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=features_dim,
            nhead=n_heads,
            dim_feedforward=features_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(features_dim, features_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(features_dim)

    def forward(self, observations):
        """
        Args:
            observations: (batch_size, n_vehicles, features_per_vehicle)

        Returns:
            features: (batch_size, features_dim)
        """
        batch_size = observations.shape[0]

        # Create vehicle indices for positional encoding
        vehicle_indices = torch.arange(self.n_vehicles, device=observations.device)
        vehicle_indices = vehicle_indices.unsqueeze(0).expand(batch_size, -1)  # (batch_size, n_vehicles)

        # Embed vehicle features
        vehicle_features = self.vehicle_embedding(observations)  # (batch_size, n_vehicles, features_dim//2)

        # Add positional encoding
        pos_encoding = self.position_embedding(vehicle_indices)  # (batch_size, n_vehicles, features_dim//2)
        combined_features = torch.cat([vehicle_features, pos_encoding], dim=-1)  # (batch_size, n_vehicles, features_dim)

        # Apply transformer attention
        # Create attention mask for vehicles that are not present (presence=0)
        presence_mask = observations[:, :, 0] == 0  # (batch_size, n_vehicles) - True for absent vehicles
        attention_mask = presence_mask  # Vehicles with presence=0 should be masked

        # Apply transformer
        transformer_out = self.transformer_encoder(combined_features)  # (batch_size, n_vehicles, features_dim)

        # Pool across vehicles (mean pooling, ignoring absent vehicles)
        presence_weights = observations[:, :, 0].unsqueeze(-1)  # (batch_size, n_vehicles, 1)
        weighted_features = transformer_out * presence_weights  # Zero out absent vehicles
        pooled_features = weighted_features.sum(dim=1) / (presence_weights.sum(dim=1) + 1e-8)  # (batch_size, features_dim)

        # Final projection and normalization
        output = self.output_proj(pooled_features)
        output = self.layer_norm(output)

        return output