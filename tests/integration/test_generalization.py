#!/usr/bin/env python3
"""
Test Generalization of Enhanced Reward Structure

This script evaluates whether the enhanced reward structure can generalize
across different driving scenarios (highway, merge, intersection) when trained
on individual scenarios or mixed scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
from environments.enhanced_urban_env import EnhancedUrbanJunctionEnv
from stable_baselines3 import PPO
from tqdm import tqdm
import os


def train_on_single_scenario(scenario, timesteps=1000):
    """Train agent on a single scenario."""
    print(f"\nTraining on {scenario} scenario...")

    # Create environment
    env = EnhancedUrbanJunctionEnv(scenario=scenario, modality="lidar", render_mode=None)

    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
    )

    # Train with progress bar
    for _ in tqdm(range(timesteps // 1024), desc=f"Training on {scenario}"):
        model.learn(total_timesteps=1024)

    # Save model
    model_path = f"outputs/models/generalization_{scenario}_expert.zip"
    os.makedirs("outputs/models", exist_ok=True)
    model.save(model_path)
    print(f"   [SAVE] Saved {scenario} expert to {model_path}")

    return model, model_path


def train_on_mixed_scenarios(timesteps=1500):
    """Train agent on mixed scenarios (curriculum-style)."""
    print(f"\n[CYCLE] Training on mixed scenarios...")

    # Create mixed environment factory
    def create_mixed_env():
        scenario = np.random.choice(['highway', 'merge', 'intersection'],
                                   p=[0.4, 0.35, 0.25])
        return EnhancedUrbanJunctionEnv(scenario=scenario, modality="lidar", render_mode=None)

    # Start with highway-focused training
    env = create_mixed_env()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
    )

    # Train with progress bar
    for _ in tqdm(range(timesteps // 1024), desc="Mixed scenario training"):
        # Recreate environment with different scenario mix over time
        if np.random.random() < 0.1:  # Occasionally switch environment
            env = create_mixed_env()
            try:
                model.set_env(env)
            except Exception as e:
                # Keep current env if incompatible
                pass
        model.learn(total_timesteps=1024)

    # Save model
    model_path = f"outputs/models/generalization_mixed_expert.zip"
    model.save(model_path)
    print(f"   [SAVE] Saved mixed expert to {model_path}")

    return model, model_path


def evaluate_agent_on_scenarios(model_path, test_scenarios=None, n_episodes=3):
    """Evaluate agent performance across different scenarios."""
    if test_scenarios is None:
        test_scenarios = ['highway', 'merge', 'intersection']

    results = {}

    print(f"\n[CHART] Evaluating {model_path}...")

    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"   [ERROR] Could not load model: {e}")
        return None

    for scenario in test_scenarios:
        print(f"   Testing on {scenario}...")

        # Create test environment
        env = EnhancedUrbanJunctionEnv(scenario=scenario, modality="lidar", render_mode=None)

        episode_rewards = []
        episode_lengths = []
        crashes = 0
        completions = 0

        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done and episode_length < 300:
                action, _ = model.predict(obs, deterministic=True)
                step_result = env.step(action)

                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result

                episode_reward += reward
                episode_length += 1

                if hasattr(env, 'vehicle') and env.vehicle.crashed:
                    crashes += 1
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Check for scenario completion
            if scenario == "merge" and hasattr(env, '_is_on_main_road') and env._is_on_main_road():
                completions += 1
            elif scenario == "intersection" and hasattr(env, '_has_cleared_intersection') and env._has_cleared_intersection():
                completions += 1
            elif scenario == "highway" and episode_length >= 200:  # Survived long enough
                completions += 1

        results[scenario] = {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'crash_rate': crashes / n_episodes,
            'completion_rate': completions / n_episodes,
            'episodes': n_episodes
        }

        print(".2f")
    return results


def run_generalization_test():
    """Run comprehensive generalization test."""
    print("=" * 80)
    print("[TEST] GENERALIZATION TEST: Enhanced Reward Structure")
    print("=" * 80)
    print("Testing whether enhanced rewards generalize across scenarios...")

    # Test configurations
    configs = [
        {
            'name': 'Highway Expert',
            'train_scenario': 'highway',
            'test_scenarios': ['highway', 'merge', 'intersection']
        },
        {
            'name': 'Merge Expert',
            'train_scenario': 'merge',
            'test_scenarios': ['highway', 'merge', 'intersection']
        },
        {
            'name': 'Intersection Expert',
            'train_scenario': 'intersection',
            'test_scenarios': ['highway', 'merge', 'intersection']
        },
        {
            'name': 'Mixed Expert',
            'train_scenario': 'mixed',
            'test_scenarios': ['highway', 'merge', 'intersection']
        }
    ]

    results = {}

    for config in configs:
        print(f"\n[TARGET] Testing: {config['name']}")

        # Train or load model
        if config['train_scenario'] == 'mixed':
            model, model_path = train_on_mixed_scenarios()
        else:
            model, model_path = train_on_single_scenario(config['train_scenario'])

        # Evaluate generalization
        eval_results = evaluate_agent_on_scenarios(model_path, config['test_scenarios'])
        if eval_results:
            results[config['name']] = eval_results

    return results


def analyze_generalization(results):
    """Analyze generalization performance."""
    print("\n" + "=" * 80)
    print("🔍 GENERALIZATION ANALYSIS")
    print("=" * 80)

    if not results:
        print("[ERROR] No results to analyze")
        return

    scenarios = ['highway', 'merge', 'intersection']
    metrics = ['avg_reward', 'crash_rate', 'completion_rate']

    # Create comparison table
    print("\n[CHART] PERFORMANCE MATRIX:")
    print("-" * 80)
    print("<12")
    print("-" * 80)

    for agent_name in results.keys():
        row = f"{agent_name:<20}"
        for scenario in scenarios:
            if scenario in results[agent_name]:
                reward = results[agent_name][scenario]['avg_reward']
                crashes = results[agent_name][scenario]['crash_rate']
                completion = results[agent_name][scenario]['completion_rate']
                row += f"{reward:>8.1f}{crashes:>8.1f}{completion:>8.1f}"
            else:
                row += "      -       -       - "
        print(row)

    print("-" * 80)

    # Analyze specialization vs generalization
    print("\n[TARGET] SPECIALIZATION ANALYSIS:")

    for scenario in scenarios:
        print(f"\n{scenario.upper()} Scenario:")
        best_agent = None
        best_reward = -float('inf')

        for agent_name, agent_results in results.items():
            if scenario in agent_results:
                reward = agent_results[scenario]['avg_reward']
                if reward > best_reward:
                    best_reward = reward
                    best_agent = agent_name

        print(f"   Best performer: {best_agent} (Reward: {best_reward:.1f})")

        # Check generalization (performance drop)
        if best_agent and scenario in results[best_agent]:
            expert_reward = results[best_agent][scenario]['avg_reward']

            # Average performance on other scenarios
            other_scenarios = [s for s in scenarios if s != scenario]
            other_rewards = []
            for s in other_scenarios:
                if s in results[best_agent]:
                    other_rewards.append(results[best_agent][s]['avg_reward'])

            if other_rewards:
                avg_other_reward = np.mean(other_rewards)
                generalization_drop = ((expert_reward - avg_other_reward) / expert_reward) * 100
                print(".1f")

    # Overall generalization assessment
    print("\n[TROPHY] OVERALL GENERALIZATION:")

    # Check if mixed training helps
    if 'Mixed Expert' in results:
        mixed_results = results['Mixed Expert']
        avg_mixed_reward = np.mean([mixed_results[s]['avg_reward'] for s in scenarios if s in mixed_results])
        avg_mixed_crashes = np.mean([mixed_results[s]['crash_rate'] for s in scenarios if s in mixed_results])

        print(".1f")
        print(".1f")

        # Compare with specialized agents
        specialized_rewards = []
        specialized_crashes = []
        for agent_name, agent_results in results.items():
            if agent_name != 'Mixed Expert':
                agent_avg_reward = np.mean([agent_results[s]['avg_reward'] for s in scenarios if s in agent_results])
                agent_avg_crashes = np.mean([agent_results[s]['crash_rate'] for s in scenarios if s in agent_results])
                specialized_rewards.append(agent_avg_reward)
                specialized_crashes.append(agent_avg_crashes)

        if specialized_rewards:
            avg_specialized_reward = np.mean(specialized_rewards)
            avg_specialized_crashes = np.mean(specialized_crashes)

            reward_diff = ((avg_mixed_reward - avg_specialized_reward) / avg_specialized_reward) * 100
            crash_diff = ((avg_specialized_crashes - avg_mixed_crashes) / avg_specialized_crashes) * 100

            print("\n[TREND] Mixed Training Impact:")
            print(".1f")
            print(".1f")

    # Final assessment
    generalization_score = 0
    total_tests = 0

    for agent_results in results.values():
        for scenario_results in agent_results.values():
            total_tests += 1
            if scenario_results['crash_rate'] < 0.5 and scenario_results['completion_rate'] > 0.3:
                generalization_score += 1

    generalization_rate = (generalization_score / total_tests) * 100 if total_tests > 0 else 0

    print("\n[TROPHY] GENERALIZATION SCORE:")
    print(".1f")

    if generalization_rate > 80:
        print("   [OK] EXCELLENT: Strong generalization across scenarios!")
    elif generalization_rate > 60:
        print("   [OK] GOOD: Moderate generalization with some specialization benefits.")
    elif generalization_rate > 40:
        print("   [WARN]  FAIR: Generalizes to some scenarios but needs improvement.")
    else:
        print("   [ERROR] POOR: Limited generalization - needs significant tuning.")

    return generalization_rate


def create_generalization_plots(results):
    """Create plots showing generalization performance."""
    if not results:
        return

    print("\n[CHART] Generating Generalization Plots...")

    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Enhanced Reward Structure Generalization Analysis', fontsize=16)

        scenarios = ['highway', 'merge', 'intersection']
        agents = list(results.keys())

        # Reward comparison
        ax1 = axes[0, 0]
        x = np.arange(len(scenarios))
        width = 0.8 / len(agents)

        for i, agent in enumerate(agents):
            rewards = [results[agent].get(s, {}).get('avg_reward', 0) for s in scenarios]
            ax1.bar(x + i*width - width*len(agents)/2, rewards, width, label=agent, alpha=0.7)

        ax1.set_xlabel('Test Scenario')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Reward Generalization')
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenarios)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Crash rate comparison
        ax2 = axes[0, 1]
        for i, agent in enumerate(agents):
            crash_rates = [results[agent].get(s, {}).get('crash_rate', 1.0) for s in scenarios]
            ax2.bar(x + i*width - width*len(agents)/2, crash_rates, width, label=agent, alpha=0.7)

        ax2.set_xlabel('Test Scenario')
        ax2.set_ylabel('Crash Rate')
        ax2.set_title('Safety Generalization')
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenarios)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        # Completion rate comparison
        ax3 = axes[1, 0]
        for i, agent in enumerate(agents):
            completion_rates = [results[agent].get(s, {}).get('completion_rate', 0) for s in scenarios]
            ax3.bar(x + i*width - width*len(agents)/2, completion_rates, width, label=agent, alpha=0.7)

        ax3.set_xlabel('Test Scenario')
        ax3.set_ylabel('Completion Rate')
        ax3.set_title('Task Completion Generalization')
        ax3.set_xticks(x)
        ax3.set_xticklabels(scenarios)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)

        # Performance radar chart (optional)
        ax4 = axes[1, 1]
        ax4.axis('off')
        ax4.text(0.5, 0.5, 'Generalization\nAnalysis\nComplete',
                ha='center', va='center', fontsize=14, transform=ax4.transAxes)

        plt.tight_layout()
        plt.savefig('generalization_analysis.png', dpi=300, bbox_inches='tight')
        print("   Plots saved to 'generalization_analysis.png'")

    except Exception as e:
        print(f"   Warning: Could not generate plots: {e}")


if __name__ == "__main__":
    # Run generalization test
    results = run_generalization_test()

    # Analyze results
    generalization_score = analyze_generalization(results)

    # Create plots
    create_generalization_plots(results)

    print("\n" + "=" * 80)
    print("[TARGET] GENERALIZATION TEST COMPLETE!")
    print("=" * 80)

    if generalization_score > 70:
        print("[OK] SUCCESS: Enhanced reward structure generalizes well across scenarios!")
        print("   The agent can learn collision avoidance that transfers between highway, merge, and intersection scenarios.")
    else:
        print("[WARN]  MIXED: Some generalization achieved but may need further tuning.")
        print("   Consider increasing mixed scenario training or adjusting reward weights.")

    print("\n💡 Recommendations:")
    print("   - Use mixed scenario training for best generalization")
    print("   - Train longer on individual scenarios for specialization")
    print("   - Consider curriculum learning for gradual scenario introduction")
