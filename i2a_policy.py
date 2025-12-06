# i2a_policy.py - Imagination-Augmented Agents for Traffic Navigation
# The ultimate overkill solution for curriculum learning challenges

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import os
from typing import Dict, Any, Optional, Tuple
from custom_policies import SimpleSpatialExtractor


class WorldModel(nn.Module):
    """
    Learns traffic dynamics: (state, action) -> (next_state, reward, done)

    This is the core of I2A - understanding how traffic evolves allows
    the agent to imagine and plan ahead, crucial for curriculum progression.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Encoder for state-action pairs
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Separate prediction heads for different outputs
        self.next_state_head = nn.Linear(hidden_dim, obs_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)  # Termination probability

        # Prediction uncertainty (for better imagination)
        self.state_uncertainty_head = nn.Linear(hidden_dim, obs_dim)  # Log variance
        self.reward_uncertainty_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next state, reward, and termination probability.

        Args:
            state: (batch, obs_dim)
            action: (batch, action_dim)

        Returns:
            next_state_pred: (batch, obs_dim)
            reward_pred: (batch, 1)
            done_pred: (batch, 1) - probability of episode termination
        """
        x = torch.cat([state, action], dim=-1)
        features = self.encoder(x)

        next_state_pred = self.next_state_head(features)
        reward_pred = self.reward_head(features)
        done_logits = self.done_head(features)
        done_pred = torch.sigmoid(done_logits)

        return next_state_pred, reward_pred, done_pred

    def predict_with_uncertainty(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimates for more robust imagination.
        """
        x = torch.cat([state, action], dim=-1)
        features = self.encoder(x)

        # Mean predictions
        next_state_mean = self.next_state_head(features)
        reward_mean = self.reward_head(features)

        # Uncertainty (log variance)
        next_state_logvar = self.state_uncertainty_head(features)
        reward_logvar = self.reward_uncertainty_head(features)

        # Sample from distributions for stochastic imagination
        next_state_std = torch.exp(0.5 * next_state_logvar)
        reward_std = torch.exp(0.5 * reward_logvar)

        next_state_sample = next_state_mean + next_state_std * torch.randn_like(next_state_std)
        reward_sample = reward_mean + reward_std * torch.randn_like(reward_std)

        done_logits = self.done_head(features)
        done_pred = torch.sigmoid(done_logits)

        return next_state_sample, reward_sample, done_pred, next_state_std, reward_std


class ImaginationCore(nn.Module):
    """
    Uses world model to imagine multiple trajectories into the future.
    This is what gives I2A its planning superpower.
    """

    def __init__(self, world_model: WorldModel, imagination_horizon: int = 5, num_imaginations: int = 8):
        super().__init__()
        self.world_model = world_model
        self.imagination_horizon = imagination_horizon
        self.num_imaginations = num_imaginations

    def imagine_trajectories(self, start_state: torch.Tensor, action_sequences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Imagine multiple trajectories by rolling out action sequences through the world model.

        Args:
            start_state: (batch, obs_dim) - starting state for all trajectories
            action_sequences: (batch, num_imaginations, horizon, action_dim)

        Returns:
            imagined_states: (batch, num_imaginations, horizon+1, obs_dim)
            imagined_rewards: (batch, num_imaginations, horizon)
            imagined_dones: (batch, num_imaginations, horizon)
        """
        batch_size = start_state.shape[0]
        obs_dim = start_state.shape[1]

        # Initialize trajectories
        current_states = start_state.unsqueeze(1).expand(-1, self.num_imaginations, -1)  # (batch, num_imaginations, obs_dim)

        imagined_states = [current_states]  # Include starting state
        imagined_rewards = []
        imagined_dones = []

        # Roll out each trajectory
        for t in range(self.imagination_horizon):
            # Get actions for this timestep: (batch, num_imaginations, action_dim)
            actions_t = action_sequences[:, :, t, :]

            # Flatten for world model: (batch * num_imaginations, obs_dim) and (batch * num_imaginations, action_dim)
            states_flat = current_states.reshape(-1, obs_dim)
            actions_flat = actions_t.reshape(-1, actions_t.shape[-1])

            # Predict next states, rewards, dones
            next_states_flat, rewards_flat, dones_flat = self.world_model(states_flat, actions_flat)

            # Reshape back: (batch, num_imaginations, ...)
            next_states = next_states_flat.view(batch_size, self.num_imaginations, obs_dim)
            rewards = rewards_flat.view(batch_size, self.num_imaginations, 1)
            dones = dones_flat.view(batch_size, self.num_imaginations, 1)

            # Store results
            imagined_states.append(next_states)
            imagined_rewards.append(rewards)
            imagined_dones.append(dones)

            # Update current states (stop imagining if done)
            current_states = next_states * (1 - dones)  # Zero out finished trajectories

        # Stack results
        imagined_states = torch.stack(imagined_states, dim=2)  # (batch, num_imaginations, horizon+1, obs_dim)
        imagined_rewards = torch.stack(imagined_rewards, dim=2)  # (batch, num_imaginations, horizon, 1)
        imagined_dones = torch.stack(imagined_dones, dim=2)    # (batch, num_imaginations, horizon, 1)

        return imagined_states, imagined_rewards, imagined_dones


class RolloutEncoder(nn.Module):
    """
    Encodes imagined trajectories into fixed-size features for policy decisions.
    This allows the policy to consider multiple imagined futures.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Encode individual trajectory steps
        self.step_encoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim + 1 + 1, hidden_dim),  # state + action + reward + done
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Aggregate trajectory into single vector
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Attention over multiple imagined trajectories
        self.multi_trajectory_attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)

        # Final projection
        self.final_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, trajectories: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """
        Encode imagined trajectories into policy features.

        Args:
            trajectories: (batch, num_imaginations, horizon+1, obs_dim)
            actions: (batch, num_imaginations, horizon, action_dim)
            rewards: (batch, num_imaginations, horizon, 1)
            dones: (batch, num_imaginations, horizon, 1)

        Returns:
            encoded_features: (batch, hidden_dim)
        """
        batch_size, num_imaginations, horizon_plus_1, obs_dim = trajectories.shape
        horizon = actions.shape[2]

        # Encode each step of each trajectory
        step_features = []

        for t in range(horizon):
            # Current state (exclude final state)
            states_t = trajectories[:, :, t, :]      # (batch, num_imaginations, obs_dim)
            actions_t = actions[:, :, t, :]          # (batch, num_imaginations, action_dim)
            rewards_t = rewards[:, :, t, :]          # (batch, num_imaginations, 1)
            dones_t = dones[:, :, t, :]              # (batch, num_imaginations, 1)

            # Concatenate: state + action + reward + done
            step_input = torch.cat([states_t, actions_t, rewards_t, dones_t], dim=-1)
            step_input_flat = step_input.view(-1, step_input.shape[-1])  # (batch * num_imaginations, features)

            step_encoded = self.step_encoder(step_input_flat)  # (batch * num_imaginations, hidden_dim)
            step_features.append(step_encoded)

        # Stack step features: (batch * num_imaginations, horizon, hidden_dim)
        trajectory_features = torch.stack(step_features, dim=1)
        trajectory_features = trajectory_features.view(batch_size * num_imaginations, horizon, -1)

        # Aggregate each trajectory into single vector
        trajectory_summary = trajectory_features.mean(dim=1)  # (batch * num_imaginations, hidden_dim)
        trajectory_encoded = self.trajectory_encoder(trajectory_summary)  # (batch * num_imaginations, hidden_dim)

        # Reshape for multi-trajectory attention: (batch, num_imaginations, hidden_dim)
        trajectory_encoded = trajectory_encoded.view(batch_size, num_imaginations, -1)

        # Apply attention across imagined trajectories
        attended_features, _ = self.multi_trajectory_attention(
            trajectory_encoded, trajectory_encoded, trajectory_encoded
        )  # (batch, num_imaginations, hidden_dim)

        # Pool across trajectories: (batch, hidden_dim)
        pooled_features = attended_features.mean(dim=1)

        # Final projection
        final_features = self.final_proj(pooled_features)

        return final_features


class I2APolicy(ActorCriticPolicy):
    """
    Imagination-Augmented Agent Policy.

    Combines traditional model-free learning with model-based imagination
    for superior long-term planning and curriculum progression.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        features_extractor_class=None,
        features_extractor_kwargs=None,
        imagination_horizon: int = 5,
        num_imaginations: int = 8,
        world_model_update_freq: int = 1,
        **kwargs
    ):
        # Store imagination parameters BEFORE calling super().__init__
        self.imagination_horizon = imagination_horizon
        self.num_imaginations = num_imaginations
        self.world_model_update_freq = world_model_update_freq

        # World model caching
        self.world_model_cache_path = None
        self.world_model_cache_loaded = False

        # Initialize features extractor (CNN for spatial vehicle processing)
        if features_extractor_class is None:
            features_extractor_class = SimpleSpatialExtractor

        # I2A combines features with imagination features, so we need larger networks
        # Default net_arch expects features_dim input, but we feed features_dim * 2 (512D)
        # We need to override features_dim to tell MLP extractor about the combined input size
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {"features_dim": 256}

        # Calculate combined features dimension for MLP input
        base_features_dim = features_extractor_kwargs.get("features_dim", 256)
        combined_features_dim = base_features_dim * 2  # 256 * 2 = 512

        if 'net_arch' not in kwargs:
            # Adjust network architecture for combined features (512D input instead of 256D)
            # First layer handles the doubled input dimension (512 -> 256 -> 128 -> 64)
            kwargs['net_arch'] = dict(pi=[256, 128, 64], vf=[256, 128, 64])

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs
        )

        # Override features_dim for combined features AFTER parent init
        # This tells the MLP extractor about the correct input dimension
        self.features_dim = combined_features_dim

        # Rebuild the MLP extractor with the correct input dimension
        from stable_baselines3.common.torch_layers import MlpExtractor
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def forward(self, obs, deterministic=False):
        """
        Forward pass with imagination augmentation.
        """
        # Extract features from current observation
        features = self.extract_features(obs)  # (batch, feature_dim)

        # Generate imagined action sequences for planning
        imagined_actions = self._sample_imagination_actions(features, obs)

        # Imagine trajectories using world model
        start_state = features  # Use extracted features as state representation
        imagined_states, imagined_rewards, imagined_dones = self.imagination_core.imagine_trajectories(
            start_state, imagined_actions
        )

        # Encode imagined trajectories into policy features
        imagination_features = self.rollout_encoder(
            imagined_states, imagined_actions, imagined_rewards, imagined_dones
        )  # (batch, encoded_dim)

        # Combine current features with imagination features
        combined_features = torch.cat([features, imagination_features], dim=-1)

        # Feed through policy and value networks
        latent_pi, latent_vf = self.mlp_extractor(combined_features)

        # Value head
        values = self.value_net(latent_vf)

        # Get action distribution and sample actions
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # Reshape to action space shape

        return actions, values, log_prob

    def _sample_imagination_actions(self, features, obs):
        """
        Sample diverse action sequences for imagination.
        Uses current policy with added exploration noise.
        """
        batch_size = features.shape[0]
        action_dim = self.action_space.shape[0]

        # Create a simple policy network that works with original features (256D)
        # This avoids the circular dependency with the combined features MLP extractor
        if not hasattr(self, '_imagination_action_net'):
            # Simple 2-layer network for imagination action sampling
            self._imagination_action_net = nn.Sequential(
                nn.Linear(self.features_extractor.features_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            ).to(self.device)

        # Get current policy distribution using dedicated imagination network
        mean_actions = self._imagination_action_net(features)

        # Sample multiple action sequences for each batch element
        action_sequences = []

        for _ in range(self.num_imaginations):
            # Add exploration noise to actions
            noise = torch.randn_like(mean_actions) * 0.2  # Increased noise for diversity
            actions = torch.tanh(mean_actions + noise)  # Use tanh for action bounds

            # Expand to horizon: (batch, horizon, action_dim)
            actions_expanded = actions.unsqueeze(1).expand(-1, self.imagination_horizon, -1)

            action_sequences.append(actions_expanded)

        # Stack: (batch, num_imaginations, horizon, action_dim)
        action_sequences = torch.stack(action_sequences, dim=1)

        return action_sequences

    def update_world_model(self, replay_buffer):
        """
        Update world model using experiences from replay buffer.
        """
        if len(replay_buffer) < 1000:  # Not enough data
            return

        # Sample transitions for world model training
        obs, actions, rewards, next_obs, dones = replay_buffer.sample(256)

        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_obs_tensor = torch.FloatTensor(next_obs).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)

        # World model predictions
        pred_next_obs, pred_rewards, pred_dones = self.world_model(obs_tensor, actions_tensor)

        # Compute losses
        obs_loss = F.mse_loss(pred_next_obs, next_obs_tensor)
        reward_loss = F.mse_loss(pred_rewards, rewards_tensor)
        done_loss = F.binary_cross_entropy(pred_dones, dones_tensor)

        world_model_loss = obs_loss + reward_loss + done_loss

        # Update world model
        self.world_model_optimizer.zero_grad()
        world_model_loss.backward()
        self.world_model_optimizer.step()

        return {
            'world_model_obs_loss': obs_loss.item(),
            'world_model_reward_loss': reward_loss.item(),
            'world_model_done_loss': done_loss.item(),
            'world_model_total_loss': world_model_loss.item()
        }

    def _build(self, lr_schedule):
        """Build additional components after standard policy construction."""
        super()._build(lr_schedule)

        # Get observation and action dimensions (now that features_extractor is built)
        obs_dim = self.features_extractor.features_dim
        action_dim = self.action_space.shape[0]

        # Create world model components
        self.world_model = WorldModel(obs_dim, action_dim, hidden_dim=512)

        # Create imagination module
        self.imagination_core = ImaginationCore(
            self.world_model,
            imagination_horizon=self.imagination_horizon,
            num_imaginations=self.num_imaginations
        )

        # Create rollout encoder
        self.rollout_encoder = RolloutEncoder(obs_dim, action_dim, hidden_dim=256)

        # Action distribution sampler for imagination (will be set after parent init)
        self.action_sampler = None

        # Optimizer for world model (separate from policy optimizer)
        world_model_params = list(self.world_model.parameters()) + \
                           list(self.imagination_core.parameters()) + \
                           list(self.rollout_encoder.parameters())

        self.world_model_optimizer = torch.optim.Adam(world_model_params, lr=1e-4)

    def predict_values(self, obs):
        """
        Get the estimated values with imagination augmentation.

        :param obs: Observation
        :return: the estimated values.
        """
        # Extract features from current observation
        features = self.extract_features(obs)  # (batch, feature_dim)

        # Generate imagined action sequences for planning
        imagined_actions = self._sample_imagination_actions(features, obs)

        # Imagine trajectories using world model
        start_state = features  # Use extracted features as state representation
        imagined_states, imagined_rewards, imagined_dones = self.imagination_core.imagine_trajectories(
            start_state, imagined_actions
        )

        # Encode imagined trajectories into policy features
        imagination_features = self.rollout_encoder(
            imagined_states, imagined_actions, imagined_rewards, imagined_dones
        )  # (batch, encoded_dim)

        # Combine current features with imagination features
        combined_features = torch.cat([features, imagination_features], dim=-1)

        # Feed through value network (only critic part)
        latent_vf = self.mlp_extractor.forward_critic(combined_features)

        return self.value_net(latent_vf)

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions according to the current policy with imagination augmentation,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Extract features from current observation
        features = self.extract_features(obs)  # (batch, feature_dim)

        # For evaluation, we need to compute imagination features
        # Note: This is an approximation - we use the same imagination features for all actions
        # In practice, we'd want action-conditioned imagination, but that's complex
        imagined_actions = self._sample_imagination_actions(features, obs)

        # Imagine trajectories using world model
        start_state = features  # Use extracted features as state representation
        imagined_states, imagined_rewards, imagined_dones = self.imagination_core.imagine_trajectories(
            start_state, imagined_actions
        )

        # Encode imagined trajectories into policy features
        imagination_features = self.rollout_encoder(
            imagined_states, imagined_actions, imagined_rewards, imagined_dones
        )  # (batch, encoded_dim)

        # Combine current features with imagination features
        combined_features = torch.cat([features, imagination_features], dim=-1)

        # Feed through policy and value networks
        latent_pi, latent_vf = self.mlp_extractor(combined_features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def save_world_model_cache(self, cache_path: str):
        """
        Save the trained world model for future reuse.

        Args:
            cache_path: Path to save the world model cache
        """
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        world_model_state = {
            'world_model': self.world_model.state_dict(),
            'imagination_core': self.imagination_core.state_dict(),
            'rollout_encoder': self.rollout_encoder.state_dict(),
            'world_model_optimizer': self.world_model_optimizer.state_dict(),
            'imagination_horizon': self.imagination_horizon,
            'num_imaginations': self.num_imaginations,
        }

        torch.save(world_model_state, cache_path)
        print(f"[I2A] World model cached to {cache_path}")

    def load_world_model_cache(self, cache_path: str, freeze_world_model: bool = True):
        """
        Load a cached world model to speed up training.

        Args:
            cache_path: Path to the cached world model
            freeze_world_model: Whether to freeze the world model parameters
        """
        if not os.path.exists(cache_path):
            print(f"[I2A] World model cache not found at {cache_path}, training from scratch")
            return False

        try:
            world_model_state = torch.load(cache_path, map_location=self.device)

            # Load model states
            self.world_model.load_state_dict(world_model_state['world_model'])
            self.imagination_core.load_state_dict(world_model_state['imagination_core'])
            self.rollout_encoder.load_state_dict(world_model_state['rollout_encoder'])
            self.world_model_optimizer.load_state_dict(world_model_state['world_model_optimizer'])

            # Load parameters
            self.imagination_horizon = world_model_state.get('imagination_horizon', self.imagination_horizon)
            self.num_imaginations = world_model_state.get('num_imaginations', self.num_imaginations)

            # Optionally freeze world model
            if freeze_world_model:
                for param in self.world_model.parameters():
                    param.requires_grad = False
                for param in self.imagination_core.parameters():
                    param.requires_grad = False
                for param in self.rollout_encoder.parameters():
                    param.requires_grad = False
                print("[I2A] World model frozen - only policy training active")

            self.world_model_cache_loaded = True
            self.world_model_cache_path = cache_path
            print(f"[I2A] World model cache loaded from {cache_path}")
            return True

        except Exception as e:
            print(f"[I2A] Failed to load world model cache: {e}")
            return False

    def set_world_model_cache_path(self, cache_path: str):
        """
        Set the path for automatic world model caching.

        Args:
            cache_path: Path for caching the world model
        """
        self.world_model_cache_path = cache_path

    def auto_save_world_model(self, step_count: int, save_interval: int = 10000):
        """
        Automatically save world model cache at regular intervals.

        Args:
            step_count: Current training step count
            save_interval: Steps between saves
        """
        if self.world_model_cache_path and step_count % save_interval == 0:
            cache_path = f"{self.world_model_cache_path}_step_{step_count}.pt"
            self.save_world_model_cache(cache_path)
