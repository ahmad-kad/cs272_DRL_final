"""
Ensemble Model Architectures for Multi-Modal Autonomous Driving

This module provides several ensemble approaches for combining lidar and grayscale
models to create more robust autonomous driving agents.

Approaches implemented:
1. Q-Value Averaging Ensemble: Average Q-values from both models
2. Mixture of Experts: Train a gating network to weight expert contributions
3. Late Fusion: Train a new policy that takes both observation types as input

Author: AI Assistant
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Dict, Any, List, Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F

from environments.urban_junction_env import UrbanJunctionEnv


class TrainableEnsemble:
    """
    Trainable Ensemble that learns optimal weights between lidar and grayscale models.

    This ensemble uses reinforcement learning to learn the best combination weights
    for different driving scenarios by optimizing a reward objective.
    """

    def __init__(
        self,
        lidar_model_path: str,
        grayscale_model_path: str,
        scenario: str = "random",
        learning_rate: float = 0.01,
        weight_init: Optional[List[float]] = None
    ):
        """
        Initialize the trainable ensemble.

        Args:
            lidar_model_path: Path to pretrained lidar model
            grayscale_model_path: Path to pretrained grayscale model
            scenario: Driving scenario for training
            learning_rate: Learning rate for weight optimization
            weight_init: Initial weights [lidar_weight, grayscale_weight]
        """
        # Load expert models
        print(f"Loading lidar model from: {lidar_model_path}")
        self.lidar_model = PPO.load(lidar_model_path)

        print(f"Loading grayscale model from: {grayscale_model_path}")
        self.grayscale_model = PPO.load(grayscale_model_path)

        # Initialize learnable weights
        if weight_init is None:
            weight_init = [0.5, 0.5]  # Start with equal weights

        self.weights = np.array(weight_init, dtype=np.float32)
        self.learning_rate = learning_rate
        self.scenario = scenario

        # Training history
        self.training_history = []
        self.episode_rewards = []
        self.weight_history = []

        print(f"Trainable ensemble initialized with weights: {self.weights}")

    def predict(
        self,
        lidar_obs: np.ndarray,
        grayscale_obs: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Make prediction using current learned weights.

        Args:
            lidar_obs: Lidar observation
            grayscale_obs: Grayscale observation
            deterministic: Whether to use deterministic actions

        Returns:
            Tuple of (action, info_dict)
        """
        # Get predictions from both models
        lidar_action, _ = self.lidar_model.predict(lidar_obs, deterministic=deterministic)
        grayscale_action, _ = self.grayscale_model.predict(grayscale_obs, deterministic=deterministic)

        # Combine using current weights
        lidar_weight, grayscale_weight = self.weights

        if isinstance(self.lidar_model.action_space, spaces.Discrete):
            # Weighted voting for discrete actions
            votes = np.zeros(self.lidar_model.action_space.n)
            votes[lidar_action] += lidar_weight
            votes[grayscale_action] += grayscale_weight
            final_action = np.argmax(votes)
        else:
            # Weighted average for continuous actions
            final_action = lidar_weight * lidar_action + grayscale_weight * grayscale_action

        info = {
            "lidar_weight": lidar_weight,
            "grayscale_weight": grayscale_weight,
            "lidar_action": lidar_action,
            "grayscale_action": grayscale_action,
            "final_action": final_action,
            "ensemble_type": "trainable"
        }

        return final_action, info

    def update_weights(self, reward: float, episode_length: int = None):
        """
        Update ensemble weights based on episode reward.

        Uses a simple reinforcement learning approach to adjust weights
        toward the better performing model.

        Args:
            reward: Episode reward
            episode_length: Episode length (optional)
        """
        # Simple policy gradient style update
        # If reward is positive, reinforce the current weight combination
        # If reward is negative, explore different weights

        reward_signal = np.clip(reward, -1.0, 1.0)  # Normalize reward

        if reward > 0:
            # Small positive updates to current weights (exploitation)
            noise_scale = 0.01
        else:
            # Larger updates to explore different weights (exploration)
            noise_scale = 0.05

        # Add noise to weights
        weight_noise = np.random.normal(0, noise_scale, size=2)
        new_weights = self.weights + self.learning_rate * reward_signal * weight_noise

        # Normalize weights to sum to 1 and keep in [0, 1]
        new_weights = np.clip(new_weights, 0.0, 1.0)
        total = np.sum(new_weights)
        if total > 0:
            new_weights = new_weights / total

        self.weights = new_weights

        # Store history
        self.training_history.append({
            "reward": reward,
            "episode_length": episode_length,
            "old_weights": self.weights.copy(),
            "new_weights": new_weights.copy(),
            "reward_signal": reward_signal
        })

    def train_weights(
        self,
        n_episodes: int = 100,
        scenario: str = None,
        save_progress: bool = True
    ):
        """
        Train the ensemble weights by evaluating performance.

        Args:
            n_episodes: Number of training episodes
            scenario: Scenario to train on (overrides instance scenario)
            save_progress: Whether to save training progress
        """
        train_scenario = scenario or self.scenario

        print(f"Training ensemble weights for {n_episodes} episodes on {train_scenario} scenario...")

        # Create evaluation environments
        lidar_env = UrbanJunctionEnv(scenario=train_scenario, modality="lidar")
        grayscale_env = UrbanJunctionEnv(scenario=train_scenario, modality="grayscale")

        for episode in range(n_episodes):
            # Reset both environments with same seed
            seed = np.random.randint(0, 10000)
            lidar_obs, _ = lidar_env.reset(seed=seed)
            grayscale_obs, _ = grayscale_env.reset(seed=seed)

            episode_reward = 0
            done = False
            steps = 0

            while not done and steps < 200:
                # Get ensemble action
                action, info = self.predict(lidar_obs, grayscale_obs, deterministic=False)

                # Step environments
                lidar_step = lidar_env.step(action)
                grayscale_env.step(action)  # Just for observation, ignore reward

                if len(lidar_step) == 5:
                    next_lidar_obs, reward, terminated, truncated, info = lidar_step
                    done = terminated or truncated
                else:
                    next_lidar_obs, reward, done, info = lidar_step

                episode_reward += reward
                steps += 1
                lidar_obs = next_lidar_obs

                # Get grayscale obs for next step
                grayscale_obs, _ = grayscale_env.reset()

            # Update weights based on episode performance
            self.update_weights(episode_reward, steps)
            self.episode_rewards.append(episode_reward)
            self.weight_history.append(self.weights.copy())

            # Progress logging
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(self.episode_rewards[-20:])
                print(f"Episode {episode + 1}/{n_episodes}: "
                     f"Avg Reward = {avg_reward:.2f}, "
                     f"Weights = [{self.weights[0]:.3f}, {self.weights[1]:.3f}]")

        print("Weight training completed!")
        print(f"Final weights: [{self.weights[0]:.3f}, {self.weights[1]:.3f}]")

        if save_progress:
            self.save_training_progress()

    def save_training_progress(self, filename: str = None):
        """Save training progress to file."""
        if filename is None:
            filename = f"trainable_ensemble_progress_{self.scenario}.json"

        progress_data = {
            "final_weights": self.weights.tolist(),
            "training_history": self.training_history,
            "episode_rewards": self.episode_rewards,
            "weight_history": [w.tolist() for w in self.weight_history],
            "scenario": self.scenario,
            "learning_rate": self.learning_rate
        }

        import json
        with open(filename, 'w') as f:
            json.dump(progress_data, f, indent=2)

        print(f"Training progress saved to {filename}")

    def load_training_progress(self, filename: str):
        """Load training progress from file."""
        import json
        with open(filename, 'r') as f:
            progress_data = json.load(f)

        self.weights = np.array(progress_data["final_weights"])
        self.training_history = progress_data["training_history"]
        self.episode_rewards = progress_data["episode_rewards"]
        self.weight_history = [np.array(w) for w in progress_data["weight_history"]]

        print(f"Loaded weights: {self.weights}")


class MultiModalEnsemble:
    """
    Q-Value Averaging Ensemble for Multi-Modal Observations.

    This ensemble loads pre-trained models for both lidar and grayscale modalities
    and combines their Q-values through weighted averaging to make decisions.

    Features:
    - Automatic weight adaptation based on prediction confidence
    - Graceful degradation when one modality fails
    - Support for different ensemble strategies (uniform, confidence-weighted, adaptive)
    """

    def __init__(
        self,
        lidar_model_path: str,
        grayscale_model_path: str,
        ensemble_strategy: str = "confidence_weighted",
        confidence_threshold: float = 0.1
    ):
        """
        Initialize the multi-modal ensemble.

        Args:
            lidar_model_path: Path to the trained lidar model
            grayscale_model_path: Path to the trained grayscale model
            ensemble_strategy: Strategy for combining predictions
                - "uniform": Equal weighting (0.5 each)
                - "confidence_weighted": Weight by prediction confidence
                - "adaptive": Learn optimal weights during evaluation
            confidence_threshold: Minimum confidence threshold for weighting
        """
        self.ensemble_strategy = ensemble_strategy
        self.confidence_threshold = confidence_threshold

        # Load models
        print(f"Loading lidar model from: {lidar_model_path}")
        self.lidar_model = PPO.load(lidar_model_path)

        print(f"Loading grayscale model from: {grayscale_model_path}")
        self.grayscale_model = PPO.load(grayscale_model_path)

        # Extract action space from models (should be the same)
        self.action_space = self.lidar_model.action_space

        # Initialize adaptive weights if using adaptive strategy
        if ensemble_strategy == "adaptive":
            self.adaptive_weights = {"lidar": 0.5, "grayscale": 0.5}
            self.weight_history = []

        print(f"Ensemble initialized with strategy: {ensemble_strategy}")

    def predict(
        self,
        lidar_obs: np.ndarray,
        grayscale_obs: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Make ensemble prediction using both modalities.

        Args:
            lidar_obs: Lidar observation array
            grayscale_obs: Grayscale observation array
            deterministic: Whether to use deterministic actions

        Returns:
            Tuple of (action, extra_info_dict)
        """
        # Get predictions from both models
        lidar_action, lidar_info = self.lidar_model.predict(lidar_obs, deterministic=deterministic)
        grayscale_action, grayscale_info = self.grayscale_model.predict(grayscale_obs, deterministic=deterministic)

        # Combine predictions based on strategy
        if self.ensemble_strategy == "uniform":
            final_action, ensemble_info = self._uniform_ensemble(lidar_action, grayscale_action,
                                                               lidar_info, grayscale_info)
        elif self.ensemble_strategy == "confidence_weighted":
            final_action, ensemble_info = self._confidence_weighted_ensemble(lidar_action, grayscale_action,
                                                                            lidar_info, grayscale_info)
        elif self.ensemble_strategy == "adaptive":
            final_action, ensemble_info = self._adaptive_ensemble(lidar_action, grayscale_action,
                                                                lidar_info, grayscale_info)
        else:
            raise ValueError(f"Unknown ensemble strategy: {self.ensemble_strategy}")

        return final_action, ensemble_info

    def _uniform_ensemble(self, lidar_action, grayscale_action, lidar_info, grayscale_info):
        """Simple uniform weighting of both models."""
        # For discrete actions, use majority vote
        if isinstance(self.action_space, spaces.Discrete):
            # Count votes for each action
            votes = np.zeros(self.action_space.n)
            votes[lidar_action] += 1
            votes[grayscale_action] += 1

            # Select action with most votes (break ties randomly)
            max_votes = np.max(votes)
            candidates = np.where(votes == max_votes)[0]
            final_action = np.random.choice(candidates)
        else:
            # For continuous actions, average them
            final_action = (lidar_action + grayscale_action) / 2

        ensemble_info = {
            "ensemble_strategy": "uniform",
            "lidar_action": lidar_action,
            "grayscale_action": grayscale_action,
            "final_action": final_action
        }

        return final_action, ensemble_info

    def _confidence_weighted_ensemble(self, lidar_action, grayscale_action, lidar_info, grayscale_info):
        """Weight predictions by model confidence."""
        # Extract confidence scores (using negative log probability as confidence measure)
        lidar_confidence = self._extract_confidence(lidar_info)
        grayscale_confidence = self._extract_confidence(grayscale_info)

        # Normalize confidences
        total_confidence = lidar_confidence + grayscale_confidence
        if total_confidence > 0:
            lidar_weight = lidar_confidence / total_confidence
            grayscale_weight = grayscale_confidence / total_confidence
        else:
            # Fallback to uniform weighting
            lidar_weight = 0.5
            grayscale_weight = 0.5

        # Apply confidence threshold
        if lidar_confidence < self.confidence_threshold:
            lidar_weight = 0.0
            grayscale_weight = 1.0
        if grayscale_confidence < self.confidence_threshold:
            grayscale_weight = 0.0
            lidar_weight = 1.0

        # Renormalize if needed
        total_weight = lidar_weight + grayscale_weight
        if total_weight > 0:
            lidar_weight /= total_weight
            grayscale_weight /= total_weight

        # Combine actions
        if isinstance(self.action_space, spaces.Discrete):
            # Weighted voting for discrete actions
            votes = np.zeros(self.action_space.n)
            votes[lidar_action] += lidar_weight
            votes[grayscale_action] += grayscale_weight
            final_action = np.argmax(votes)
        else:
            # Weighted average for continuous actions
            final_action = lidar_weight * lidar_action + grayscale_weight * grayscale_action

        ensemble_info = {
            "ensemble_strategy": "confidence_weighted",
            "lidar_weight": lidar_weight,
            "grayscale_weight": grayscale_weight,
            "lidar_confidence": lidar_confidence,
            "grayscale_confidence": grayscale_confidence,
            "lidar_action": lidar_action,
            "grayscale_action": grayscale_action,
            "final_action": final_action
        }

        return final_action, ensemble_info

    def _adaptive_ensemble(self, lidar_action, grayscale_action, lidar_info, grayscale_info):
        """Use learned adaptive weights."""
        lidar_weight = self.adaptive_weights["lidar"]
        grayscale_weight = self.adaptive_weights["grayscale"]

        # Combine actions
        if isinstance(self.action_space, spaces.Discrete):
            votes = np.zeros(self.action_space.n)
            votes[lidar_action] += lidar_weight
            votes[grayscale_action] += grayscale_weight
            final_action = np.argmax(votes)
        else:
            final_action = lidar_weight * lidar_action + grayscale_weight * grayscale_action

        ensemble_info = {
            "ensemble_strategy": "adaptive",
            "lidar_weight": lidar_weight,
            "grayscale_weight": grayscale_weight,
            "lidar_action": lidar_action,
            "grayscale_action": grayscale_action,
            "final_action": final_action
        }

        return final_action, ensemble_info

    def _extract_confidence(self, model_info) -> float:
        """Extract confidence score from model prediction info."""
        # Stable Baselines 3 predict() returns (action, states) where states may be None
        if model_info is None:
            return 1.0  # Default confidence

        # Use log probability as confidence measure (higher is better)
        if isinstance(model_info, dict):
            if "log_probability" in model_info:
                return float(model_info["log_probability"])
            elif "probabilities" in model_info:
                # Use max probability as confidence
                probs = np.array(model_info["probabilities"])
                return float(np.max(probs))

        # Default confidence if not available
        return 1.0

    def update_adaptive_weights(self, reward: float, lidar_correct: bool, grayscale_correct: bool):
        """
        Update adaptive weights based on performance feedback.

        Args:
            reward: Episode reward
            lidar_correct: Whether lidar model made correct decision
            grayscale_correct: Whether grayscale model made correct decision
        """
        if self.ensemble_strategy != "adaptive":
            return

        # Simple reinforcement learning update
        learning_rate = 0.01
        reward_signal = 1.0 if reward > 0 else -1.0

        if lidar_correct and not grayscale_correct:
            self.adaptive_weights["lidar"] += learning_rate * reward_signal
            self.adaptive_weights["grayscale"] -= learning_rate * reward_signal
        elif grayscale_correct and not lidar_correct:
            self.adaptive_weights["grayscale"] += learning_rate * reward_signal
            self.adaptive_weights["lidar"] -= learning_rate * reward_signal

        # Normalize weights to sum to 1
        total = self.adaptive_weights["lidar"] + self.adaptive_weights["grayscale"]
        if total > 0:
            self.adaptive_weights["lidar"] /= total
            self.adaptive_weights["grayscale"] /= total
        else:
            # Reset to uniform if weights become invalid
            self.adaptive_weights = {"lidar": 0.5, "grayscale": 0.5}

        # Clip to reasonable bounds
        self.adaptive_weights["lidar"] = np.clip(self.adaptive_weights["lidar"], 0.1, 0.9)
        self.adaptive_weights["grayscale"] = np.clip(self.adaptive_weights["grayscale"], 0.1, 0.9)

        # Store history
        self.weight_history.append(self.adaptive_weights.copy())


class MultiModalLateFusionEnv(gym.Env):
    """
    Late Fusion Environment that combines lidar and grayscale observations from separate environments.

    This environment creates two separate UrbanJunctionEnv instances (one for each modality)
    and concatenates their observations, allowing training of a single policy that learns to
    optimally combine information from both sensors.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        scenario: str = "highway",
        **kwargs
    ):
        """
        Initialize the late fusion environment.

        Args:
            config: Environment configuration
            scenario: Driving scenario
            **kwargs: Additional arguments
        """
        super().__init__()

        # Create separate environments for each modality
        self.lidar_env = UrbanJunctionEnv(config=config, scenario=scenario, modality="lidar", **kwargs)
        self.grayscale_env = UrbanJunctionEnv(config=config, scenario=scenario, modality="grayscale", **kwargs)

        # Get actual observation sizes by doing a test reset
        lidar_obs, _ = self.lidar_env.reset()
        grayscale_obs, _ = self.grayscale_env.reset()

        lidar_size = lidar_obs.size
        grayscale_size = grayscale_obs.size
        total_size = lidar_size + grayscale_size

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(total_size,),
            dtype=np.float32
        )

        print(f"Actual sizes from reset: lidar={lidar_size}, grayscale={grayscale_size}, total={total_size}")

        # Use action space from either environment (they should be the same)
        self.action_space = self.lidar_env.action_space

        print(f"Late fusion environment initialized with combined observation space: {self.observation_space.shape}")
        print(f"Lidar obs shape: {self.lidar_env.observation_space.shape}")
        print(f"Grayscale obs shape: {self.grayscale_env.observation_space.shape}")

    def reset(self, **kwargs):
        """Reset both environments and return combined observation."""
        # Reset both environments with same seed for consistency
        seed = kwargs.get('seed', np.random.randint(0, 10000))

        lidar_obs, lidar_info = self.lidar_env.reset(seed=seed)
        grayscale_obs, grayscale_info = self.grayscale_env.reset(seed=seed)

        # Combine observations
        combined_obs = self._combine_observations(lidar_obs, grayscale_obs)

        return combined_obs, {}

    def step(self, action):
        """Step both environments and return combined observation and shared reward."""
        # Step both environments with same action
        lidar_result = self.lidar_env.step(action)
        grayscale_result = self.grayscale_env.step(action)

        # Use lidar environment for primary results (reward, done, info)
        if len(lidar_result) == 5:
            next_lidar_obs, reward, terminated, truncated, info = lidar_result
            done = terminated or truncated
        else:
            next_lidar_obs, reward, done, info = lidar_result

        # Get next grayscale observation
        if len(grayscale_result) == 5:
            next_grayscale_obs, _, _, _, _ = grayscale_result
        else:
            next_grayscale_obs, _, _, _ = grayscale_result

        # Combine observations
        combined_obs = self._combine_observations(next_lidar_obs, next_grayscale_obs)

        return combined_obs, reward, done, False, info  # Return 5-tuple for gymnasium compatibility

    def _combine_observations(self, lidar_obs, grayscale_obs):
        """Combine lidar and grayscale observations into single array."""
        lidar_flat = lidar_obs.flatten()
        grayscale_flat = grayscale_obs.flatten().astype(np.float32)

        # Normalize grayscale to similar scale as lidar
        grayscale_flat = (grayscale_flat / 127.5) - 1.0  # Normalize to [-1, 1]

        combined_obs = np.concatenate([lidar_flat, grayscale_flat])
        return combined_obs

    def render(self):
        """Render the lidar environment (primary)."""
        return self.lidar_env.render()

    def close(self):
        """Close both environments."""
        self.lidar_env.close()
        self.grayscale_env.close()


class MixtureOfExpertsEnsemble(nn.Module):
    """
    Mixture of Experts Ensemble with learned gating network.

    This approach trains a gating network that learns to weight the contributions
    of different expert models based on the current observation.
    """

    def __init__(
        self,
        lidar_model_path: str,
        grayscale_model_path: str,
        gating_network_dims: List[int] = [256, 128, 2],
        learning_rate: float = 1e-4
    ):
        """
        Initialize the Mixture of Experts ensemble.

        Args:
            lidar_model_path: Path to lidar expert model
            grayscale_model_path: Path to grayscale expert model
            gating_network_dims: Dimensions for the gating network
            learning_rate: Learning rate for gating network training
        """
        super().__init__()

        # Load expert models
        self.lidar_expert = PPO.load(lidar_model_path)
        self.grayscale_expert = PPO.load(grayscale_model_path)

        # Extract observation shapes
        self.lidar_obs_shape = self.lidar_expert.observation_space.shape
        self.grayscale_obs_shape = self.grayscale_expert.observation_space.shape

        # Create gating network
        gating_layers = []
        input_dim = self.lidar_obs_shape[0] + self.grayscale_obs_shape[0]

        for i, dim in enumerate(gating_network_dims):
            if i == 0:
                gating_layers.append(nn.Linear(input_dim, dim))
            else:
                gating_layers.append(nn.Linear(gating_network_dims[i-1], dim))

            if i < len(gating_network_dims) - 1:
                gating_layers.append(nn.ReLU())

        # Final layer outputs weights for each expert (should sum to 1)
        gating_layers.append(nn.Softmax(dim=-1))

        self.gating_network = nn.Sequential(*gating_layers)

        # Training components
        self.optimizer = torch.optim.Adam(self.gating_network.parameters(), lr=learning_rate)
        self.training_history = []

        print(f"Mixture of Experts initialized with gating network: {gating_layers}")

    def forward(self, lidar_obs: torch.Tensor, grayscale_obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the gating network.

        Args:
            lidar_obs: Lidar observation tensor
            grayscale_obs: Grayscale observation tensor

        Returns:
            Expert weights tensor [batch_size, 2]
        """
        combined_obs = torch.cat([lidar_obs, grayscale_obs], dim=-1)
        expert_weights = self.gating_network(combined_obs)
        return expert_weights

    def predict(
        self,
        lidar_obs: np.ndarray,
        grayscale_obs: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Make prediction using the mixture of experts.

        Args:
            lidar_obs: Lidar observation
            grayscale_obs: Grayscale observation
            deterministic: Whether to use deterministic actions

        Returns:
            Tuple of (action, info_dict)
        """
        # Convert to tensors
        lidar_tensor = torch.FloatTensor(lidar_obs).unsqueeze(0)
        grayscale_tensor = torch.FloatTensor(grayscale_obs).unsqueeze(0)

        # Get expert weights
        with torch.no_grad():
            expert_weights = self.forward(lidar_tensor, grayscale_tensor)
            lidar_weight, grayscale_weight = expert_weights[0].cpu().numpy()

        # Get expert predictions
        lidar_action, _ = self.lidar_expert.predict(lidar_obs, deterministic=deterministic)
        grayscale_action, _ = self.grayscale_expert.predict(grayscale_obs, deterministic=deterministic)

        # Combine predictions
        if isinstance(self.lidar_expert.action_space, spaces.Discrete):
            # Weighted voting
            votes = np.zeros(self.lidar_expert.action_space.n)
            votes[lidar_action] += lidar_weight
            votes[grayscale_action] += grayscale_weight
            final_action = np.argmax(votes)
        else:
            # Weighted average
            final_action = lidar_weight * lidar_action + grayscale_weight * grayscale_action

        info = {
            "lidar_weight": lidar_weight,
            "grayscale_weight": grayscale_weight,
            "lidar_action": lidar_action,
            "grayscale_action": grayscale_action,
            "final_action": final_action,
            "ensemble_type": "mixture_of_experts"
        }

        return final_action, info

    def update_gating_network(
        self,
        lidar_obs: np.ndarray,
        grayscale_obs: np.ndarray,
        reward: float,
        next_lidar_obs: np.ndarray = None,
        next_grayscale_obs: np.ndarray = None
    ):
        """
        Update the gating network based on reward feedback.

        Args:
            lidar_obs: Current lidar observation
            grayscale_obs: Current grayscale observation
            reward: Reward received
            next_lidar_obs: Next lidar observation (for temporal difference)
            next_grayscale_obs: Next grayscale observation (for temporal difference)
        """
        # Convert to tensors
        lidar_tensor = torch.FloatTensor(lidar_obs).unsqueeze(0)
        grayscale_tensor = torch.FloatTensor(grayscale_obs).unsqueeze(0)

        # Get current expert weights
        expert_weights = self.forward(lidar_tensor, grayscale_tensor)

        # Simple reward-based update (could be extended with TD learning)
        reward_tensor = torch.FloatTensor([reward])

        # Loss: negative reward weighted by expert weights
        # (encourage high weights when reward is positive)
        loss = -torch.sum(expert_weights * reward_tensor.unsqueeze(-1))

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store training info
        self.training_history.append({
            "loss": loss.item(),
            "reward": reward,
            "lidar_weight": expert_weights[0, 0].item(),
            "grayscale_weight": expert_weights[0, 1].item()
        })


def create_ensemble_policy(lidar_model_path: str, grayscale_model_path: str) -> ActorCriticPolicy:
    """
    Create a custom policy class for late fusion training.

    This policy takes concatenated lidar + grayscale observations and processes them
    through separate feature extractors before combining.

    Args:
        lidar_model_path: Path to pretrained lidar model
        grayscale_model_path: Path to pretrained grayscale model

    Returns:
        Custom ActorCriticPolicy class
    """

    class MultiModalPolicy(ActorCriticPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Load pretrained feature extractors
            lidar_model = PPO.load(lidar_model_path)
            grayscale_model = PPO.load(grayscale_model_path)

            # Extract feature extractors (this is model-specific)
            self.lidar_extractor = lidar_model.policy.features_extractor
            self.grayscale_extractor = grayscale_model.policy.features_extractor

            # Create fusion layer
            lidar_features_dim = lidar_model.policy.features_extractor.features_dim
            grayscale_features_dim = grayscale_model.policy.features_extractor.features_dim

            self.fusion_layer = nn.Sequential(
                nn.Linear(lidar_features_dim + grayscale_features_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU()
            )

            # Override the features_dim
            self.features_dim = 128

        def extract_features(self, obs):
            """Extract features from concatenated observation."""
            # Split observation back into modalities
            lidar_obs, grayscale_obs = self._split_observation(obs)

            # Extract features from each modality
            lidar_features = self.lidar_extractor(lidar_obs)
            grayscale_features = self.grayscale_extractor(grayscale_obs)

            # Fuse features
            combined_features = torch.cat([lidar_features, grayscale_features], dim=-1)
            fused_features = self.fusion_layer(combined_features)

            return fused_features

        def _split_observation(self, obs):
            """Split concatenated observation back into modalities."""
            # This depends on the specific observation shapes used
            # For now, assume lidar comes first, then grayscale
            # TODO: Make this configurable based on actual shapes
            lidar_size = 448  # 64 cells * 7 features
            lidar_obs = obs[:, :lidar_size]
            grayscale_obs = obs[:, lidar_size:]
            return lidar_obs, grayscale_obs

    return MultiModalPolicy
