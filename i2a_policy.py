# i2a_policy.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np


class WorldModel(nn.Module):
    """
    Learns to predict: (obs, action) → (next_obs, reward, done)
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Separate prediction heads
        self.next_obs_head = nn.Linear(hidden_dim, obs_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action):
        """
        Predict next state, reward, and termination probability.

        Args:
            obs: (batch, obs_dim)
            action: (batch, action_dim)

        Returns:
            next_obs_pred: (batch, obs_dim)
            reward_pred: (batch, 1)
            done_pred: (batch, 1) - probability of termination
        """
        x = torch.cat([obs, action], dim=-1)
        features = self.encoder(x)

        next_obs_pred = self.next_obs_head(features)
        reward_pred = self.reward_head(features)
        done_pred = torch.sigmoid(self.done_head(features))

        return next_obs_pred, reward_pred, done_pred

    def imagine_trajectory(self, start_obs, actions):
        """
        Imagine a trajectory given an action sequence.

        Args:
            start_obs: (obs_dim,) initial observation
            actions: (horizon, action_dim) action sequence

        Returns:
            obs_sequence: (horizon+1, obs_dim) - includes start observation
            reward_sequence: (horizon,) predicted rewards
            done_sequence: (horizon,) predicted termination probabilities
        """
        horizon = actions.shape[0]
        current_obs = start_obs.unsqueeze(0)  # Add batch dimension

        obs_sequence = [start_obs]
        reward_sequence = []
        done_sequence = []

        for t in range(horizon):
            action = actions[t:t+1]  # (1, action_dim)

            next_obs_pred, reward_pred, done_pred = self(current_obs, action)

            obs_sequence.append(next_obs_pred.squeeze(0))
            reward_sequence.append(reward_pred.squeeze(0))
            done_sequence.append(done_pred.squeeze(0))

            current_obs = next_obs_pred

        return torch.stack(obs_sequence), torch.stack(reward_sequence), torch.stack(done_sequence)


class ImaginationModule(nn.Module):
    """
    Uses world model to imagine and evaluate multiple action sequences.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=128, imagination_horizon=5, num_imaginations=8):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.imagination_horizon = imagination_horizon
        self.num_imaginations = num_imaginations

        # World model for imagination
        self.world_model = WorldModel(obs_dim, action_dim, hidden_dim)

        # Trajectory encoder - compresses imagined trajectories
        trajectory_dim = imagination_horizon * (obs_dim + 1 + 1)  # obs + reward + done
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(trajectory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, obs):
        """
        Imagine multiple trajectories and return compressed features.

        Args:
            obs: (batch, obs_dim) current observation

        Returns:
            imagination_features: (batch, hidden_dim) averaged imagined features
        """
        batch_size = obs.shape[0]
        device = obs.device

        imagination_features = []

        for b in range(batch_size):
            current_obs = obs[b]  # (obs_dim,)

            # Sample multiple action sequences and imagine trajectories
            trajectory_features = []

            for _ in range(self.num_imaginations):
                # Sample random action sequence
                actions = torch.randn(self.imagination_horizon, self.action_dim, device=device)

                # Imagine trajectory
                obs_seq, reward_seq, done_seq = self.world_model.imagine_trajectory(current_obs, actions)

                # Encode trajectory (skip initial observation, focus on future)
                trajectory = torch.cat([
                    obs_seq[1:],      # next observations (horizon, obs_dim)
                    reward_seq.unsqueeze(-1),    # rewards (horizon, 1)
                    done_seq.unsqueeze(-1)       # done flags (horizon, 1)
                ], dim=-1).view(-1)  # flatten to (horizon * (obs_dim + 1 + 1),)

                encoded_trajectory = self.trajectory_encoder(trajectory)
                trajectory_features.append(encoded_trajectory)

            # Average over imaginations for this batch element
            batch_imagination = torch.stack(trajectory_features).mean(dim=0)
            imagination_features.append(batch_imagination)

        return torch.stack(imagination_features)


class I2AFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that combines observation features with imagination.
    """

    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        obs_dim = observation_space.shape[0]
        action_dim = 2  # continuous actions for crazy_driver

        # Standard observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

        # Imagination module
        self.imagination_module = ImaginationModule(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=features_dim,
            imagination_horizon=5,
            num_imaginations=8
        )

        # Combine observation + imagination features
        self.combiner = nn.Sequential(
            nn.Linear(features_dim * 2, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim)
        )

    def forward(self, observations):
        # Encode current observation
        obs_features = self.obs_encoder(observations)

        # Get imagination features (this uses the world model)
        imagination_features = self.imagination_module(observations)

        # Combine features
        combined = torch.cat([obs_features, imagination_features], dim=-1)
        output_features = self.combiner(combined)

        return output_features


class I2APolicy(ActorCriticPolicy):
    """
    I2A policy for PPO - combines model-free learning with imagination.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=I2AFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=dict(pi=[64, 64], vf=[64, 64])  # Smaller networks since we have good features
        )
