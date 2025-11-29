#!/usr/bin/env python3
"""
Training Script for Enhanced Reward Structure

Demonstrates the improved learning performance with the enhanced reward structure
that provides dense, scenario-aware feedback for multi-scenario autonomous driving.

Usage:
    python train_enhanced_rewards.py --scenario highway --timesteps 10000
    python train_enhanced_rewards.py --scenario merge --timesteps 10000
    python train_enhanced_rewards.py --scenario intersection --timesteps 10000
"""

import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

from environments.enhanced_urban_env import EnhancedUrbanJunctionEnv
from utils.callbacks import WandbMetricsCallback, StratifiedMetricsCallback


from stable_baselines3.common.callbacks import BaseCallback

class TqdmCallback(BaseCallback):
    """Custom callback for tqdm progress tracking during training."""

    def __init__(self, total_timesteps, update_freq=100, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.update_freq = update_freq
        self.pbar = None

    def _on_training_start(self):
        """Initialize progress bar when training starts."""
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self):
        """Update progress bar after each step."""
        if self.pbar is not None and self.n_calls % self.update_freq == 0:
            current_step = min(self.num_timesteps, self.total_timesteps)
            self.pbar.n = current_step
            self.pbar.refresh()
        return True

    def _on_training_end(self):
        """Close progress bar when training ends."""
        if self.pbar is not None:
            self.pbar.close()


def create_enhanced_env(scenario="highway", seed=42):
    """Create enhanced environment with improved reward structure."""
    env = EnhancedUrbanJunctionEnv(
        scenario=scenario,
        modality="lidar",  # Use lidar for simplicity
        render_mode=None
    )
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    return env


def create_baseline_env(scenario="highway", seed=42):
    """Create baseline environment with original reward structure."""
    from environments.urban_junction_env import UrbanJunctionEnv

    env = UrbanJunctionEnv(
        scenario=scenario,
        modality="lidar",
        render_mode=None
    )
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    return env


def train_with_rewards(env_factory, scenario_name, total_timesteps=10000, use_wandb=False, exploration_schedule="adaptive"):
    """Train agent with specified reward structure."""

    # Create environment
    env = env_factory(scenario=scenario_name)

    # Create PPO agent with enhanced exploration for generalization
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  # INCREASED from 0.01 - better exploration for generalization
        vf_coef=0.5,
        max_grad_norm=0.5,  # Gradient clipping for stability
        target_kl=0.01,     # More conservative KL divergence for stable learning
        tensorboard_log=None  # Disable tensorboard for testing
    )

    # Setup callbacks
    callbacks = []

    # Comprehensive stratified metrics callback
    stratified_cb = StratifiedMetricsCallback(verbose=1)
    callbacks.append(stratified_cb)

    # Add tqdm progress callback
    tqdm_cb = TqdmCallback(total_timesteps=total_timesteps, update_freq=500)
    callbacks.append(tqdm_cb)

    if use_wandb:
        callbacks.append(WandbMetricsCallback())

    checkpoint_cb = CheckpointCallback(
        save_freq=5000,
        save_path=f"outputs/models/enhanced_{scenario_name}",
        name_prefix=f"enhanced_{scenario_name}"
    )
    callbacks.append(checkpoint_cb)

    print(f"\n[CAR] Training on {scenario_name} scenario with enhanced rewards...")
    print(f"   Environment: {type(env).__name__}")
    print(f"   Total timesteps: {total_timesteps}")
    print(f"   Reward components: Speed, Lane Position, Progress, Completion")
    print("=" * 60)

    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks) if callbacks else None
    )

    # Save final model
    final_path = f"outputs/models/enhanced_{scenario_name}_final.zip"
    model.save(final_path)
    print(f"\n[SAVE] Model saved: {final_path}")

    return model


def evaluate_agent(model, env_factory, scenario_name, n_episodes=10):
    """Evaluate trained agent performance."""
    print(f"\n[CHART] Evaluating {scenario_name} performance...")

    env = env_factory(scenario=scenario_name)

    episode_rewards = []
    episode_lengths = []
    success_count = 0
    crash_count = 0

    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done and episode_length < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            episode_length += 1

            # Check for crashes (info format depends on environment)
            if isinstance(info, list) and len(info) > 0:
                if info[0].get('crashed', False):
                    crash_count += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Define success criteria based on scenario
        if scenario_name == "merge":
            success = episode_reward > 5 and crash_count == 0  # Successfully merged
        elif scenario_name == "intersection":
            success = episode_reward > 10 and crash_count == 0  # Cleared intersection
        else:  # highway
            success = episode_reward > 20 and crash_count == 0  # Good highway driving

        if success:
            success_count += 1

    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    success_rate = success_count / n_episodes
    crash_rate = crash_count / n_episodes

    results = {
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_length": avg_length,
        "success_rate": success_rate,
        "crash_rate": crash_rate,
        "episodes": n_episodes
    }

    print(".2f")
    print(".1f")
    print(".1f")
    return results


def compare_reward_structures(scenario="highway", timesteps=5000):
    """Compare enhanced vs baseline reward structures."""
    print(f"\n[LAB] COMPARING REWARD STRUCTURES: {scenario.upper()} SCENARIO")
    print("=" * 80)

    # Train with enhanced rewards
    print("\n[TARGET] Training with ENHANCED rewards...")
    enhanced_model = train_with_rewards(
        create_enhanced_env,
        f"{scenario}_enhanced",
        total_timesteps=timesteps,
        use_wandb=False
    )

    # Train with baseline rewards
    print("\n[CHART] Training with BASELINE rewards...")
    baseline_model = train_with_rewards(
        create_baseline_env,
        f"{scenario}_baseline",
        total_timesteps=timesteps,
        use_wandb=False
    )

    # Evaluate both
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    enhanced_results = evaluate_agent(enhanced_model, create_enhanced_env, scenario)
    baseline_results = evaluate_agent(baseline_model, create_baseline_env, scenario)

    # Print comparison
    print(f"\n[TREND] PERFORMANCE COMPARISON ({scenario.upper()}):")
    print("-" * 50)
    print("<12")
    print("<12")
    print("<12")
    print("<12")
    print("<12")

    # Calculate improvement percentages
    reward_improvement = ((enhanced_results['avg_reward'] - baseline_results['avg_reward']) /
                         abs(baseline_results['avg_reward'])) * 100
    success_improvement = (enhanced_results['success_rate'] - baseline_results['success_rate']) * 100

    print(".1f")
    print(".1f")

    if reward_improvement > 0:
        print("   [CELEBRATE] Enhanced rewards show better performance!")
    else:
        print("   [THINK] Baseline performed better - may need reward tuning.")

    return {
        "enhanced": enhanced_results,
        "baseline": baseline_results,
        "improvement": {
            "reward_pct": reward_improvement,
            "success_pct": success_improvement
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Train with enhanced reward structure")
    parser.add_argument(
        "--scenario",
        choices=["highway", "merge", "intersection"],
        default="highway",
        help="Driving scenario to train on"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=5000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare enhanced vs baseline rewards"
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only evaluate pre-trained models"
    )

    args = parser.parse_args()

    if args.compare:
        # Run comparison between enhanced and baseline
        results = compare_reward_structures(args.scenario, args.timesteps)

        # Save comparison results
        import json
        os.makedirs("results", exist_ok=True)
        with open(f"results/reward_comparison_{args.scenario}.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n[SAVE] Comparison results saved to: results/reward_comparison_{args.scenario}.json")

    elif args.evaluate_only:
        # Load and evaluate existing models
        print("Evaluating existing models...")

        try:
            enhanced_model = PPO.load(f"outputs/models/enhanced_{args.scenario}_enhanced_final.zip")
            enhanced_results = evaluate_agent(enhanced_model, create_enhanced_env, args.scenario)
        except Exception as e:
            print(f"Enhanced model not found: {e}")
            enhanced_results = None

        try:
            baseline_model = PPO.load(f"outputs/models/enhanced_{args.scenario}_baseline_final.zip")
            baseline_results = evaluate_agent(baseline_model, create_baseline_env, args.scenario)
        except Exception as e:
            print(f"Baseline model not found: {e}")
            baseline_results = None

        if enhanced_results and baseline_results:
            reward_diff = enhanced_results['avg_reward'] - baseline_results['avg_reward']
            success_diff = enhanced_results['success_rate'] - baseline_results['success_rate']
            print(".1f")

    else:
        # Train with enhanced rewards only
        print("[ROCKET] Training with Enhanced Reward Structure")
        print("   Features:")
        print("   - Scenario-aware speed optimization")
        print("   - Lane position rewards")
        print("   - Progress tracking")
        print("   - Completion bonuses")
        print()

        model = train_with_rewards(
            create_enhanced_env,
            args.scenario,
            total_timesteps=args.timesteps
        )

        # Evaluate the trained model
        results = evaluate_agent(model, create_enhanced_env, args.scenario)

        print("\n[TARGET] Training Complete!")
        print(f"   Final performance on {args.scenario}: {results['avg_reward']:.2f} avg reward")


if __name__ == "__main__":
    main()
