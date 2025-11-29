#!/usr/bin/env python3
"""
Analyze Exploration in Enhanced Reward Structure

This script evaluates the exploration capabilities of the PPO algorithm
with enhanced rewards for generalization across driving scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
from environments.enhanced_urban_env import EnhancedUrbanJunctionEnv
from stable_baselines3 import PPO
import torch


def analyze_ppo_exploration():
    """Analyze PPO exploration characteristics."""
    print("🔍 PPO EXPLORATION ANALYSIS")
    print("=" * 50)

    # Create test environment
    env = EnhancedUrbanJunctionEnv(scenario="highway", modality="lidar", render_mode=None)

    # Create PPO with different exploration settings
    configs = [
        {"name": "Conservative", "ent_coef": 0.005, "clip_range": 0.1},
        {"name": "Balanced", "ent_coef": 0.01, "clip_range": 0.2},
        {"name": "Explorative", "ent_coef": 0.02, "clip_range": 0.3},
        {"name": "High Exploration", "ent_coef": 0.05, "clip_range": 0.4}
    ]

    results = {}

    for config in configs:
        print(f"\n[TEST] Testing: {config['name']} Exploration")

        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=512,  # Smaller for testing
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=config["clip_range"],
            ent_coef=config["ent_coef"],
            vf_coef=0.5,
        )

        # Test exploration before training
        pre_train_actions = collect_action_distribution(model, env, n_samples=1000)
        pre_train_entropy = calculate_action_entropy(pre_train_actions)

        # Quick training
        model.learn(total_timesteps=2000)

        # Test exploration after training
        post_train_actions = collect_action_distribution(model, env, n_samples=1000)
        post_train_entropy = calculate_action_entropy(post_train_actions)

        results[config["name"]] = {
            "pre_entropy": pre_train_entropy,
            "post_entropy": post_train_entropy,
            "entropy_change": post_train_entropy - pre_train_entropy,
            "config": config,
            "pre_actions": pre_train_actions,
            "post_actions": post_train_actions
        }

        print(".3f")
        print(".3f")
        print(".3f")
    return results


def collect_action_distribution(model, env, n_samples=1000):
    """Collect action distribution from model."""
    actions = []

    for _ in range(n_samples):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=False)
        actions.append(action)

    return np.array(actions)


def calculate_action_entropy(actions):
    """Calculate entropy of action distribution."""
    if len(actions.shape) > 1:
        # Multi-dimensional actions
        entropy = 0
        for dim in range(actions.shape[1]):
            hist, _ = np.histogram(actions[:, dim], bins=20, density=True)
            hist = hist[hist > 0]  # Remove zeros
            entropy += -np.sum(hist * np.log(hist)) * (np.max(actions[:, dim]) - np.min(actions[:, dim])) / 20
        return entropy / actions.shape[1]  # Average across dimensions
    else:
        # Single dimension
        hist, _ = np.histogram(actions, bins=20, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist))


def analyze_scenario_diversity():
    """Analyze how well the agent explores different scenario behaviors."""
    print("\n[TARGET] SCENARIO-SPECIFIC EXPLORATION ANALYSIS")
    print("=" * 50)

    scenarios = ["highway", "merge", "intersection"]
    results = {}

    for scenario in scenarios:
        print(f"\nTesting {scenario} scenario...")

        # Create environment
        env = EnhancedUrbanJunctionEnv(scenario=scenario, modality="lidar", render_mode=None)

        # Create model
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            ent_coef=0.02,  # Good exploration setting
            clip_range=0.2,
        )

        # Test different behaviors the agent should learn
        behaviors = {
            "safe_driving": test_behavior_pattern(model, env, "safe"),
            "aggressive_driving": test_behavior_pattern(model, env, "aggressive"),
            "defensive_driving": test_behavior_pattern(model, env, "defensive"),
            "random_exploration": test_behavior_pattern(model, env, "random")
        }

        results[scenario] = behaviors

        # Calculate diversity score
        action_diversity = calculate_behavior_diversity(behaviors)
        print(".3f")
    return results


def test_behavior_pattern(model, env, pattern_type):
    """Test specific behavior patterns."""
    actions = []

    for _ in range(100):
        obs, _ = env.reset()

        # Modify observation based on pattern
        if pattern_type == "safe":
            # Simulate safe conditions (good observations)
            pass  # Use normal observations
        elif pattern_type == "aggressive":
            # Simulate risky conditions
            pass  # Could modify observations to seem riskier
        elif pattern_type == "defensive":
            # Simulate dangerous conditions
            pass  # Could modify to trigger proximity penalties
        # elif pattern_type == "random":
            # Use random actions

        if pattern_type == "random":
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=False)

        actions.append(action)

    return np.array(actions)


def calculate_behavior_diversity(behaviors):
    """Calculate diversity across different behavior patterns."""
    entropies = []
    for behavior_actions in behaviors.values():
        entropy = calculate_action_entropy(behavior_actions)
        entropies.append(entropy)

    # Diversity is the variance in entropies across behaviors
    return np.var(entropies)


def evaluate_exploration_quality(results):
    """Evaluate overall exploration quality for generalization."""
    print("\n[CHART] EXPLORATION QUALITY ASSESSMENT")
    print("=" * 50)

    # Analyze entropy changes
    print("\n[CYCLE] Entropy Analysis:")
    for config_name, data in results.items():
        entropy_change = data["entropy_change"]
        if entropy_change > 0.1:
            quality = "[OK] GOOD - Learning to explore"
        elif entropy_change > 0:
            quality = "[WARN]  FAIR - Some exploration learning"
        else:
            quality = "[ERROR] POOR - Losing exploration capability"

        print(".3f")
    # Find best exploration setting
    best_config = max(results.items(), key=lambda x: x[1]["post_entropy"])
    print(f"\n[TARGET] BEST EXPLORATION SETTING: {best_config[0]}")
    print(".3f")
    # Generalization readiness
    avg_post_entropy = np.mean([data["post_entropy"] for data in results.values()])

    print("\n[TROPHY] GENERALIZATION READINESS:")
    if avg_post_entropy > 1.5:
        print("   [OK] EXCELLENT - Agent explores diverse behaviors")
        print("   [TARGET] Ready for cross-scenario generalization")
    elif avg_post_entropy > 1.0:
        print("   [OK] GOOD - Moderate exploration capability")
        print("   [TARGET] Should generalize with longer training")
    elif avg_post_entropy > 0.5:
        print("   [WARN]  FAIR - Limited exploration")
        print("   💡 Consider increasing ent_coef or adding curiosity")
    else:
        print("   [ERROR] POOR - Insufficient exploration")
        print("   🔧 Need to increase ent_coef significantly")

    return avg_post_entropy


def create_exploration_plots(results):
    """Create plots showing exploration analysis."""
    print("\n[TREND] Generating Exploration Analysis Plots...")

    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('PPO Exploration Analysis for Enhanced Rewards', fontsize=16)

        configs = list(results.keys())

        # Entropy comparison
        ax1 = axes[0, 0]
        pre_entropies = [results[c]["pre_entropy"] for c in configs]
        post_entropies = [results[c]["post_entropy"] for c in configs]

        x = np.arange(len(configs))
        ax1.bar(x - 0.2, pre_entropies, 0.4, label='Pre-training', alpha=0.7)
        ax1.bar(x + 0.2, post_entropies, 0.4, label='Post-training', alpha=0.7)

        ax1.set_xlabel('Exploration Configuration')
        ax1.set_ylabel('Action Entropy')
        ax1.set_title('Exploration Entropy Before/After Training')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Entropy change
        ax2 = axes[0, 1]
        entropy_changes = [results[c]["entropy_change"] for c in configs]
        colors = ['red' if x < 0 else 'green' for x in entropy_changes]
        ax2.bar(configs, entropy_changes, color=colors, alpha=0.7)
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Entropy Change')
        ax2.set_title('Learning to Explore (Positive = Good)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        # Action distributions (example)
        ax3 = axes[1, 0]
        best_config = max(results.items(), key=lambda x: x[1]["post_entropy"])[0]
        pre_actions = results[best_config]["pre_actions"]
        post_actions = results[best_config]["post_actions"]

        if len(pre_actions.shape) > 1:
            # Multi-dimensional - show first dimension
            ax3.hist(pre_actions[:, 0], bins=20, alpha=0.5, label='Pre-training', density=True)
            ax3.hist(post_actions[:, 0], bins=20, alpha=0.5, label='Post-training', density=True)
        else:
            ax3.hist(pre_actions, bins=20, alpha=0.5, label='Pre-training', density=True)
            ax3.hist(post_actions, bins=20, alpha=0.5, label='Post-training', density=True)

        ax3.set_xlabel('Action Value')
        ax3.set_ylabel('Density')
        ax3.set_title(f'Action Distribution: {best_config}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Exploration quality radar
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Create a simple quality assessment
        qualities = []
        for config, data in results.items():
            score = min(1.0, data["post_entropy"] / 2.0)  # Normalize to 0-1
            qualities.append(score)

        avg_quality = np.mean(qualities)
        quality_text = ".1f"
        ax4.text(0.5, 0.5, f'Exploration Quality\n{quality_text}',
                ha='center', va='center', fontsize=14, transform=ax4.transAxes)

        plt.tight_layout()
        plt.savefig('exploration_analysis.png', dpi=300, bbox_inches='tight')
        print("   Plots saved to 'exploration_analysis.png'")

    except Exception as e:
        print(f"   Warning: Could not generate plots: {e}")


def main():
    """Run complete exploration analysis."""
    print("[TEST] EXPLORATION ANALYSIS: Enhanced Reward Structure")
    print("Evaluating PPO exploration capabilities for generalization...")

    # Analyze PPO exploration
    exploration_results = analyze_ppo_exploration()

    # Analyze scenario diversity
    scenario_results = analyze_scenario_diversity()

    # Evaluate overall quality
    avg_entropy = evaluate_exploration_quality(exploration_results)

    # Create plots
    create_exploration_plots(exploration_results)

    print("\n" + "=" * 80)
    print("[TARGET] EXPLORATION ANALYSIS COMPLETE!")
    print("=" * 80)

    # Final recommendations
    print("\n💡 RECOMMENDATIONS FOR GENERALIZATION:")

    if avg_entropy > 1.2:
        print("   [OK] EXCELLENT exploration - use current settings")
        print("   [TARGET] Ready for multi-scenario generalization")
    elif avg_entropy > 0.8:
        print("   [OK] GOOD exploration - minor improvements suggested")
        print("   💡 Consider: ent_coef=0.02, curriculum learning")
    else:
        print("   [WARN]  INSUFFICIENT exploration - improvements needed")
        print("   🔧 Try: ent_coef=0.03-0.05, add curiosity bonus")

    print("\n[ROCKET] TRAINING STRATEGY:")
    print("   - Start with single scenario mastery (highway)")
    print("   - Gradually introduce mixed scenarios")
    print("   - Use ent_coef=0.02 for balanced exploration")
    print("   - Train 10k-20k timesteps per scenario")
    print("   - Monitor entropy to ensure exploration is maintained")


if __name__ == "__main__":
    main()
