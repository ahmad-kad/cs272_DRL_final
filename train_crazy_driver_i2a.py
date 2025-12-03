# train_crazy_driver_i2a.py

import argparse
import os
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

# Import the custom environment and I2A policy
from team2_env.crazy_driver_enviornment import crazy_driver_env
from i2a_policy import I2APolicy

# Optional: Import wandb helpers if you want logging
try:
    from wandb_single_helpers import make_wandb_single_callback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def make_env(render_mode=None):
    """
    Create the crazy_driver_env for training.

    Uses kinematic observations (the default for this env).
    """
    config = crazy_driver_env.default_config()

    # Training-specific config overrides
    config.update({
        "offscreen_rendering": render_mode == "rgb_array",
        "render_agent": False,  # Don't render during training
        "show_trajectories": False,
    })

    env = gym.make(
        "CopChase-v0",
        render_mode=render_mode,
        config=config,
    )

    if render_mode is None:
        # For training: wrap in Monitor so infos contain 'episode'
        env = Monitor(env)

    return env


def make_env_thunk(render_mode=None):
    """
    Factory for SubprocVecEnv.
    Needs to be top-level so it's picklable.
    """
    def _init():
        return make_env(render_mode=render_mode)
    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train I2A-augmented PPO on the crazy_driver_env."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500k)",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=4,  # Fewer envs due to higher computational cost
        help="Number of parallel envs for training (default: 4)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="If set, do not use Weights & Biases logging.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="crazy_driver_i2a",
        help="Name for saved model (default: crazy_driver_i2a)",
    )
    parser.add_argument(
        "--pretrain_world_model",
        action="store_true",
        help="Pre-train world model before I2A training",
    )
    parser.add_argument(
        "--world_model_steps",
        type=int,
        default=100_000,
        help="Steps to pre-train world model (default: 100k)",
    )
    return parser.parse_args()


def pretrain_world_model(env, device, steps=100_000):
    """
    Pre-train the world model using real environment interactions.
    """
    print(f"[WORLD MODEL] Pre-training for {steps} steps...")

    # Create world model
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    world_model = I2APolicy(
        observation_space=env.observation_space,
        action_space=env.action_space
    ).features_extractor.imagination_module.world_model.to(device)

    optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-4)
    world_model.train()

    obs, _ = env.reset()
    episode_reward = 0
    episode_steps = 0

    for step in range(steps):
        # Random action for exploration
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
        next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(device)
        reward_tensor = torch.FloatTensor([reward]).unsqueeze(0).to(device)
        done_tensor = torch.FloatTensor([terminated or truncated]).unsqueeze(0).to(device)

        # World model prediction
        pred_next_obs, pred_reward, pred_done = world_model(obs_tensor, action_tensor)

        # Compute losses
        obs_loss = F.mse_loss(pred_next_obs, next_obs_tensor)
        reward_loss = F.mse_loss(pred_reward, reward_tensor)
        done_loss = F.binary_cross_entropy(pred_done, done_tensor)

        total_loss = obs_loss + reward_loss + done_loss

        # Update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update for next step
        obs = next_obs
        episode_reward += reward
        episode_steps += 1

        if terminated or truncated:
            obs, _ = env.reset()
            episode_reward = 0
            episode_steps = 0

        if step % 10000 == 0:
            print(f"[WORLD MODEL] Step {step}/{steps}, Loss: {total_loss.item():.4f}")

    print("[WORLD MODEL] Pre-training completed!")
    return world_model


def main():
    args = parse_args()
    total_timesteps = args.steps

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Build SubprocVecEnv for true parallel env stepping
    env_fns = [make_env_thunk(render_mode=None) for _ in range(args.n_envs)]
    env = SubprocVecEnv(env_fns)

    # Decide device (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Optional: Pre-train world model
    if args.pretrain_world_model:
        single_env = make_env()
        pretrain_world_model(single_env, device, args.world_model_steps)
        single_env.close()

    # Model save path
    model_path = f"outputs/models/{args.model_name}_{total_timesteps//1000}k.zip"

    # I2A-augmented PPO
    model = PPO(
        I2APolicy,  # Custom I2A policy
        env,
        verbose=1,
        learning_rate=1e-4,  # Lower LR for complex architecture
        ent_coef=0.01,      # Lower entropy for more deterministic policies
        n_steps=1024,       # Shorter rollouts due to computational cost
        batch_size=32,      # Smaller batches
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./logs/tb_crazy_driver_i2a/",
        device=device,
        policy_kwargs=dict(
            features_extractor_kwargs=dict(features_dim=128),
        )
    )

    # W&B callback (optional)
    if args.no_wandb or not WANDB_AVAILABLE:
        callbacks = None
        if args.no_wandb:
            print("[INFO] W&B logging disabled (--no_wandb set).")
        else:
            print("[INFO] W&B not available.")
    else:
        callbacks = make_wandb_single_callback(
            total_timesteps=total_timesteps,
            env_id="CopChase-v0",
            obs_type="kinematic_i2a",
            project="crazy-driver-i2a",
            run_name=f"crazy_driver_i2a-{total_timesteps//1000}k",
            verbose=1,
        )

    print(f"[TRAIN] I2A-augmented PPO on Crazy Driver environment")
    print(f"  Action space: Continuous (acceleration, steering)")
    print(f"  Observation space: Kinematic ({env.observation_space.shape[0]}D)")
    print(f"  Imagination: {8} trajectories × {5} steps each")
    print(f"  Policy: I2A with MLP features + imagination features")
    print(f"  Steps: {total_timesteps}, n_envs: {args.n_envs}")
    print(f"  Note: ~2-3x slower than baseline due to imagination computations")

    # Train the model
    if callbacks is not None:
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    else:
        model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Save the model
    model.save(model_path)
    env.close()
    print(f"[SAVE] I2A model saved to {model_path}")
    print("[DONE] I2A training finished.")


if __name__ == "__main__":
    main()
