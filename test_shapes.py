# test_shapes.py - Test tensor shape compatibility for I2A policy

import torch
import gymnasium as gym
from stable_baselines3.common.policies import ActorCriticPolicy
from i2a_policy import I2APolicy
from team2_env.crazy_driver_enviornment import crazy_driver_env

def test_i2a_shapes():
    """Test that I2A policy handles tensor shapes correctly."""

    # Create a minimal environment for testing
    config = crazy_driver_env.default_config()
    config.update({
        "duration": 10,  # Short duration for testing
        "vehicles_count": 5,  # Fewer vehicles for testing
    })

    env = gym.make("CopChase-v0", config=config)

    # Create I2A policy
    policy = I2APolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda x: 3e-4,
        imagination_horizon=3,  # Smaller for testing
        num_imaginations=4,     # Smaller for testing
    )

    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Features extractor output dim: {policy.features_extractor.features_dim}")

    # Test forward pass with dummy data
    batch_size = 2
    obs = torch.randn(batch_size, *env.observation_space.shape)

    try:
        # Extract features
        features = policy.extract_features(obs)
        print(f"Extracted features shape: {features.shape}")

        # Test imagination components
        imagined_actions = policy._sample_imagination_actions(features, obs)
        print(f"Imagined actions shape: {imagined_actions.shape}")

        # Test imagination core
        imagined_states, imagined_rewards, imagined_dones = policy.imagination_core.imagine_trajectories(
            features, imagined_actions
        )
        print(f"Imagined states shape: {imagined_states.shape}")
        print(f"Imagined rewards shape: {imagined_rewards.shape}")
        print(f"Imagined dones shape: {imagined_dones.shape}")

        # Test rollout encoder
        imagination_features = policy.rollout_encoder(
            imagined_states, imagined_actions, imagined_rewards, imagined_dones
        )
        print(f"Imagination features shape: {imagination_features.shape}")

        # Test combined features
        combined_features = torch.cat([features, imagination_features], dim=-1)
        print(f"Combined features shape: {combined_features.shape}")

        # Test MLP extractor (this was failing before)
        latent_pi, latent_vf = policy.mlp_extractor(combined_features)
        print(f"Policy latent shape: {latent_pi.shape}")
        print(f"Value latent shape: {latent_vf.shape}")

        # Test action and value heads
        mean_actions = policy.action_net(latent_pi)
        values = policy.value_net(latent_vf)
        print(f"Mean actions shape: {mean_actions.shape}")
        print(f"Values shape: {values.shape}")

        print("[SUCCESS] All shapes are compatible!")

    except Exception as e:
        print(f"[ERROR] Shape error: {e}")
        raise

    env.close()

if __name__ == "__main__":
    test_i2a_shapes()
