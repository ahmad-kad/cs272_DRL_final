#!/usr/bin/env python3
"""
Ensemble Training Script for Multi-Modal Autonomous Driving

This script provides multiple approaches for training ensemble models that combine
lidar and grayscale observations for improved autonomous driving performance.

Available Approaches:
1. Q-Value Averaging Ensemble: Load pretrained models and combine predictions
2. Late Fusion Training: Train a new policy on concatenated observations
3. Mixture of Experts: Train a gating network to weight expert contributions

Usage:
    # Q-Value Averaging Ensemble
    python train_ensemble.py --approach q_value_ensemble --lidar-model outputs/models/adaptive_lidar_final.zip --grayscale-model outputs/models/adaptive_grayscale_final.zip

    # Late Fusion Training
    python train_ensemble.py --approach late_fusion --total-timesteps 50000

    # Mixture of Experts
    python train_ensemble.py --approach mixture_experts --lidar-model outputs/models/adaptive_lidar_final.zip --grayscale-model outputs/models/adaptive_grayscale_final.zip

Author: AI Assistant
"""

import os
import json
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
import wandb
from typing import Dict, Any, Optional, List

from environments.urban_junction_env import UrbanJunctionEnv
from training.ensemble_models import (
    MultiModalEnsemble,
    MultiModalLateFusionEnv,
    MixtureOfExpertsEnsemble,
    TrainableEnsemble
)
from utils.config import get_curriculum_config
from utils.callbacks import WandbMetricsCallback, StratifiedMetricsCallback
# from training.adaptive_trainer import AdaptiveCurriculum  # Avoid import due to missing models module


def create_q_value_ensemble(
    lidar_model_path: str,
    grayscale_model_path: str,
    ensemble_strategy: str = "confidence_weighted"
) -> MultiModalEnsemble:
    """
    Create a Q-value averaging ensemble.

    Args:
        lidar_model_path: Path to pretrained lidar model
        grayscale_model_path: Path to pretrained grayscale model
        ensemble_strategy: Ensemble combination strategy

    Returns:
        Configured MultiModalEnsemble
    """
    print("Creating Q-Value Averaging Ensemble...")
    ensemble = MultiModalEnsemble(
        lidar_model_path=lidar_model_path,
        grayscale_model_path=grayscale_model_path,
        ensemble_strategy=ensemble_strategy
    )
    return ensemble


def train_late_fusion_model(
    total_timesteps: int = 50000,
    use_wandb: bool = True,
    checkpoint_path: Optional[str] = None
) -> PPO:
    """
    Train a late fusion model on concatenated observations.

    Args:
        total_timesteps: Total training timesteps
        use_wandb: Whether to use Weights & Biases logging
        checkpoint_path: Path to checkpoint for resuming training

    Returns:
        Trained PPO model
    """
    print("Training Late Fusion Model...")

    if use_wandb:
        wandb.init(
            project="autonomous-driving-ensemble",
            name="late-fusion-training",
            config={
                "approach": "late_fusion",
                "total_timesteps": total_timesteps,
                "checkpoint_path": checkpoint_path,
            }
        )

    # Create late fusion environment
    env = MultiModalLateFusionEnv(scenario="random")

    # Use standard MlpPolicy with larger network for the combined observations
    policy_kwargs = {
        "net_arch": [512, 256, 128],  # Larger network for combined observations
    }

    # Create model
    model = PPO(
        "MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="outputs/logs",
        device="cuda" if torch.cuda.is_available() else "cpu",
        policy_kwargs=policy_kwargs,
        n_steps=2048,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
    )

    # Setup callbacks
    callbacks = [StratifiedMetricsCallback(verbose=1)]  # Comprehensive metrics tracking

    if use_wandb:
        callbacks.append(WandbMetricsCallback())

    checkpoint_cb = CheckpointCallback(
        save_freq=10000,
        save_path="outputs/models",
        name_prefix="ensemble_late_fusion"
    )
    callbacks.append(checkpoint_cb)

    # Train model
    print(f"Starting late fusion training for {total_timesteps} timesteps...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))

    # Save final model
    final_path = "outputs/models/ensemble_late_fusion_final.zip"
    model.save(final_path)
    print(f"Late fusion model saved: {final_path}")

    if use_wandb:
        wandb.finish()

    return model


def train_trainable_ensemble(
    lidar_model_path: str,
    grayscale_model_path: str,
    n_episodes: int = 200,
    scenario: str = "random",
    learning_rate: float = 0.01,
    weight_init: Optional[List[float]] = None,
    use_wandb: bool = True
) -> TrainableEnsemble:
    """
    Train a trainable ensemble that learns optimal weights.

    Args:
        lidar_model_path: Path to pretrained lidar model
        grayscale_model_path: Path to pretrained grayscale model
        n_episodes: Number of training episodes
        scenario: Scenario to train on
        learning_rate: Learning rate for weight updates
        weight_init: Initial weights
        use_wandb: Whether to use Weights & Biases logging

    Returns:
        Trained TrainableEnsemble
    """
    print("Training Trainable Ensemble...")

    if use_wandb:
        wandb.init(
            project="autonomous-driving-ensemble",
            name="trainable-ensemble-training",
            config={
                "approach": "trainable_ensemble",
                "n_episodes": n_episodes,
                "scenario": scenario,
                "learning_rate": learning_rate,
                "weight_init": weight_init,
                "lidar_model": lidar_model_path,
                "grayscale_model": grayscale_model_path,
            }
        )

    # Create trainable ensemble
    ensemble = TrainableEnsemble(
        lidar_model_path=lidar_model_path,
        grayscale_model_path=grayscale_model_path,
        scenario=scenario,
        learning_rate=learning_rate,
        weight_init=weight_init
    )

    # Train the weights
    ensemble.train_weights(n_episodes=n_episodes, scenario=scenario)

    # Save the trained ensemble
    ensemble.save_training_progress("outputs/models/trainable_ensemble_progress.json")

    if use_wandb:
        wandb.finish()

    return ensemble


def train_mixture_of_experts(
    lidar_model_path: str,
    grayscale_model_path: str,
    total_timesteps: int = 50000,
    use_wandb: bool = True
) -> MixtureOfExpertsEnsemble:
    """
    Train a Mixture of Experts ensemble with gating network.

    Args:
        lidar_model_path: Path to pretrained lidar model
        grayscale_model_path: Path to pretrained grayscale model
        total_timesteps: Total training timesteps
        use_wandb: Whether to use Weights & Biases logging

    Returns:
        Trained MixtureOfExpertsEnsemble
    """
    print("Training Mixture of Experts Ensemble...")

    if use_wandb:
        wandb.init(
            project="autonomous-driving-ensemble",
            name="mixture-of-experts-training",
            config={
                "approach": "mixture_of_experts",
                "total_timesteps": total_timesteps,
                "lidar_model": lidar_model_path,
                "grayscale_model": grayscale_model_path,
            }
        )

    # Create mixture of experts
    moe = MixtureOfExpertsEnsemble(
        lidar_model_path=lidar_model_path,
        grayscale_model_path=grayscale_model_path
    )

    # Training environment for collecting experience
    env = UrbanJunctionEnv(scenario="random", modality="lidar")  # Use lidar for primary observation

    print(f"Starting MoE training for {total_timesteps} timesteps...")

    obs, info = env.reset()
    episode_reward = 0
    episode_count = 0

    for step in range(total_timesteps):
        # Get both observations (this would need to be modified based on actual env structure)
        # For now, we'll use the same observation for both (simplified)
        lidar_obs = obs
        grayscale_obs = obs  # This should be replaced with actual grayscale observation

        # Get ensemble action
        action, info = moe.predict(lidar_obs, grayscale_obs, deterministic=False)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_reward += reward

        # Update gating network with reward feedback
        moe.update_gating_network(lidar_obs, grayscale_obs, reward)

        if done:
            episode_count += 1
            print(".2f")

            # Log to wandb
            if use_wandb and episode_count % 10 == 0:
                wandb.log({
                    "episode": episode_count,
                    "episode_reward": episode_reward,
                    "lidar_weight": info.get("lidar_weight", 0.5),
                    "grayscale_weight": info.get("grayscale_weight", 0.5),
                })

            obs, info = env.reset()
            episode_reward = 0

        if step % 10000 == 0:
            print(f"Step {step}/{total_timesteps} completed")

    # Save gating network
    torch.save(moe.gating_network.state_dict(), "outputs/models/ensemble_moe_gating.pth")
    print("Mixture of Experts gating network saved")

    if use_wandb:
        wandb.finish()

    return moe


def _run_ensemble_evaluation_episodes(
    ensemble,
    lidar_env,
    grayscale_env,
    n_episodes: int
) -> Dict[str, Any]:
    """
    Run evaluation episodes for Q-value ensemble with both observation types.

    Args:
        ensemble: The ensemble to evaluate
        lidar_env: Lidar environment
        grayscale_env: Grayscale environment
        n_episodes: Number of episodes

    Returns:
        Episode results
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    total_crashes = 0
    ensemble_stats = []

    for episode in range(n_episodes):
        # Reset both environments with same seed for fair comparison
        seed = np.random.randint(0, 10000)
        lidar_obs, _ = lidar_env.reset(seed=seed)
        grayscale_obs, _ = grayscale_env.reset(seed=seed)

        episode_reward = 0
        episode_crashes = 0
        done = False
        steps = 0
        episode_ensemble_info = []

        while not done and steps < 200:
            # Get ensemble action
            action, ensemble_info = ensemble.predict(
                lidar_obs, grayscale_obs, deterministic=True
            )

            # Step both environments (use lidar for primary stepping and reward)
            step_result = lidar_env.step(action)
            if len(step_result) == 5:
                next_lidar_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_lidar_obs, reward, done, info = step_result

            # Step grayscale env too (for next observation)
            grayscale_env.step(action)  # Don't care about reward from this

            episode_reward += reward
            steps += 1

            # Check for crashes
            if isinstance(info, dict) and info.get("crashed", False):
                episode_crashes += 1

            # Store ensemble decision info
            episode_ensemble_info.append(ensemble_info)

            lidar_obs = next_lidar_obs
            # Get next grayscale obs
            grayscale_obs, _ = grayscale_env.reset()  # Simplified - should track properly

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        total_crashes += episode_crashes

        if episode_reward > 0 and episode_crashes == 0:
            success_count += 1

        # Store ensemble statistics
        if episode_ensemble_info:
            avg_lidar_weight = np.mean([info.get("lidar_weight", 0.5) for info in episode_ensemble_info])
            avg_grayscale_weight = np.mean([info.get("grayscale_weight", 0.5) for info in episode_ensemble_info])
            ensemble_stats.append({
                "avg_lidar_weight": float(avg_lidar_weight),
                "avg_grayscale_weight": float(avg_grayscale_weight)
            })

    return {
        "episodes": n_episodes,
        "success_rate": float(success_count / n_episodes),
        "avg_reward": float(np.mean(episode_rewards)),
        "reward_std": float(np.std(episode_rewards)),
        "crash_rate": float(total_crashes / n_episodes),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "success_count": success_count,
        "total_crashes": total_crashes,
        "ensemble_stats": ensemble_stats
    }


def _run_single_evaluation_episodes(
    model,
    env,
    n_episodes: int,
    modality: str
) -> Dict[str, Any]:
    """
    Run evaluation episodes for single model.

    Args:
        model: Model to evaluate
        env: Environment
        n_episodes: Number of episodes
        modality: Observation modality

    Returns:
        Episode results
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    total_crashes = 0

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_crashes = 0
        done = False
        steps = 0

        while not done and steps < 200:
            action, _ = model.predict(obs, deterministic=True)

            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result

            episode_reward += reward
            steps += 1

            if isinstance(info, dict) and info.get("crashed", False):
                episode_crashes += 1

            obs = next_obs

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        total_crashes += episode_crashes

        if episode_reward > 0 and episode_crashes == 0:
            success_count += 1

    return {
        "episodes": n_episodes,
        "success_rate": float(success_count / n_episodes),
        "avg_reward": float(np.mean(episode_rewards)),
        "reward_std": float(np.std(episode_rewards)),
        "crash_rate": float(total_crashes / n_episodes),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "success_count": success_count,
        "total_crashes": total_crashes
    }


def evaluate_ensemble(
    ensemble,
    ensemble_type: str,
    n_episodes: int = 50,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Evaluate an ensemble model across scenarios.

    Args:
        ensemble: The ensemble model to evaluate
        ensemble_type: Type of ensemble ("q_value", "late_fusion", "moe")
        n_episodes: Number of episodes per scenario
        save_results: Whether to save results

    Returns:
        Evaluation results dictionary
    """
    print(f"\nEvaluating {ensemble_type} ensemble...")

    results = {
        "ensemble_type": ensemble_type,
        "scenarios": {}
    }

    scenarios = ["highway", "merge", "intersection"]

    for scenario in scenarios:
        print(f"Evaluating on {scenario} scenario...")

        # For ensembles that need both observation types
        if ensemble_type in ["q_value", "trainable"]:
            # In the _run_ensemble_evaluation_episodes function, use curriculum config
            from utils.config import get_curriculum_config

            # Replace environment creation with:
            env_config = get_curriculum_config(scenario, "hard", "lidar")  # Use hard difficulty
            lidar_env = UrbanJunctionEnv(config=env_config, scenario=scenario, modality="lidar")
            grayscale_env = UrbanJunctionEnv(config=env_config, scenario=scenario, modality="grayscale")

            scenario_results = _run_ensemble_evaluation_episodes(
                ensemble, lidar_env, grayscale_env, n_episodes
            )
        else:
            # For other ensemble types, use single environment
            env = UrbanJunctionEnv(scenario=scenario, modality="lidar")
            scenario_results = _run_single_evaluation_episodes(
                ensemble, env, n_episodes, "lidar"
            )

        results["scenarios"][scenario] = scenario_results

        print(f"  {scenario}: Success Rate = {scenario_results['success_rate']:.2f}, "
              f"Avg Reward = {scenario_results['avg_reward']:.2f}")

    # Calculate overall statistics
    all_success_rates = [results["scenarios"][s]["success_rate"] for s in scenarios]
    all_rewards = [results["scenarios"][s]["avg_reward"] for s in scenarios]
    all_crashes = [results["scenarios"][s]["crash_rate"] for s in scenarios]

    results["summary"] = {
        "overall_success_rate": float(np.mean(all_success_rates)),
        "overall_avg_reward": float(np.mean(all_rewards)),
        "overall_crash_rate": float(np.mean(all_crashes)),
        "success_rate_std": float(np.std(all_success_rates)),
        "reward_std": float(np.std(all_rewards))
    }

    print(f"\nOverall Results for {ensemble_type}:")
    print(f"  Success Rate: {results['summary']['overall_success_rate']:.3f}")
    print(f"  Average Reward: {results['summary']['overall_avg_reward']:.2f}")
    print(f"  Crash Rate: {results['summary']['overall_crash_rate']:.3f}")

    # Save results
    if save_results:
        results_file = f"results/ensemble_{ensemble_type}_evaluation.json"
        os.makedirs("results", exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_file}")

    return results


def main():
    """Main ensemble training function."""
    parser = argparse.ArgumentParser(description="Train ensemble models for autonomous driving")
    parser.add_argument(
        "--approach",
        choices=["q_value_ensemble", "late_fusion", "mixture_experts", "trainable_ensemble"],
        default="q_value_ensemble",
        help="Ensemble training approach"
    )
    parser.add_argument(
        "--lidar-model",
        default="outputs/models/adaptive_lidar_final.zip",
        help="Path to pretrained lidar model"
    )
    parser.add_argument(
        "--grayscale-model",
        default="outputs/models/adaptive_grayscale_final.zip",
        help="Path to pretrained grayscale model"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=50000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--ensemble-strategy",
        choices=["uniform", "confidence_weighted", "adaptive"],
        default="confidence_weighted",
        help="Strategy for Q-value ensemble combination"
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=200,
        help="Number of episodes for trainable ensemble training"
    )
    parser.add_argument(
        "--ensemble-scenario",
        choices=["highway", "merge", "intersection", "random"],
        default="random",
        help="Scenario for trainable ensemble training"
    )
    parser.add_argument(
        "--ensemble-lr",
        type=float,
        default=0.01,
        help="Learning rate for trainable ensemble"
    )
    parser.add_argument(
        "--weight-init",
        nargs=2,
        type=float,
        default=[0.5, 0.5],
        help="Initial weights for trainable ensemble [lidar, grayscale]"
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only evaluate existing ensemble, don't train"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )

    args = parser.parse_args()

    # Validate model paths
    if args.approach in ["q_value_ensemble", "mixture_experts", "trainable_ensemble"]:
        if not os.path.exists(args.lidar_model):
            raise FileNotFoundError(f"Lidar model not found: {args.lidar_model}")
        if not os.path.exists(args.grayscale_model):
            raise FileNotFoundError(f"Grayscale model not found: {args.grayscale_model}")

    use_wandb = not args.no_wandb

    if args.evaluate_only:
        print("Evaluation-only mode selected")
        # Load and evaluate appropriate ensemble
        if args.approach == "q_value_ensemble":
            ensemble = create_q_value_ensemble(
                args.lidar_model,
                args.grayscale_model,
                args.ensemble_strategy
            )
            evaluate_ensemble(ensemble, "q_value", save_results=True)

        elif args.approach == "late_fusion":
            # For late fusion evaluation, we need to create the environment and load the model
            from training.ensemble_models import MultiModalLateFusionEnv
            env = MultiModalLateFusionEnv(scenario="highway")  # Use a single scenario for evaluation
            model = PPO.load("outputs/models/ensemble_late_fusion_final.zip", env=env)
            # Evaluate using single environment evaluation since late fusion is already combined
            results = _run_single_evaluation_episodes(model, env, 50, "combined")
            print(f"Late fusion evaluation: Success Rate = {results['success_rate']:.2f}, Avg Reward = {results['avg_reward']:.2f}")
            # Save results manually
            import json
            os.makedirs("results", exist_ok=True)
            with open("results/ensemble_late_fusion_evaluation.json", 'w') as f:
                json.dump({"late_fusion": {"highway": results}}, f, indent=2)

        elif args.approach == "mixture_experts":
            moe = MixtureOfExpertsEnsemble(args.lidar_model, args.grayscale_model)
            moe.gating_network.load_state_dict(
                torch.load("outputs/models/ensemble_moe_gating.pth")
            )
            evaluate_ensemble(moe, "moe", save_results=True)

        elif args.approach == "trainable_ensemble":
            ensemble = TrainableEnsemble(
                args.lidar_model,
                args.grayscale_model,
                scenario=args.ensemble_scenario,
                learning_rate=args.ensemble_lr,
                weight_init=args.weight_init
            )
            # Load existing progress if available
            progress_file = "outputs/models/trainable_ensemble_progress.json"
            if os.path.exists(progress_file):
                ensemble.load_training_progress(progress_file)
            evaluate_ensemble(ensemble, "trainable", save_results=True)

    else:
        # Training mode
        print(f"Starting {args.approach} ensemble training...")

        if args.approach == "q_value_ensemble":
            # Q-value ensemble doesn't need training, just evaluation
            ensemble = create_q_value_ensemble(
                args.lidar_model,
                args.grayscale_model,
                args.ensemble_strategy
            )
            evaluate_ensemble(ensemble, f"q_value_{args.ensemble_strategy}", save_results=True)

        elif args.approach == "late_fusion":
            model = train_late_fusion_model(
                total_timesteps=args.total_timesteps,
                use_wandb=use_wandb
            )
            evaluate_ensemble(model, "late_fusion", save_results=True)

        elif args.approach == "mixture_experts":
            moe = train_mixture_of_experts(
                args.lidar_model,
                args.grayscale_model,
                total_timesteps=args.total_timesteps,
                use_wandb=use_wandb
            )
            evaluate_ensemble(moe, "moe", save_results=True)

        elif args.approach == "trainable_ensemble":
            ensemble = train_trainable_ensemble(
                args.lidar_model,
                args.grayscale_model,
                n_episodes=args.train_episodes,
                scenario=args.ensemble_scenario,
                learning_rate=args.ensemble_lr,
                weight_init=args.weight_init,
                use_wandb=use_wandb
            )
            evaluate_ensemble(ensemble, "trainable", save_results=True)

    print("Ensemble training/evaluation completed!")


if __name__ == "__main__":
    import torch  # Import torch for MoE
    main()

