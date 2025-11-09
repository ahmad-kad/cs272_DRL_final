#!/usr/bin/env python3
"""
Phase 4: Rigorous Validation (The Final Exam)

This script implements comprehensive validation to prove agent robustness.
Unlike training rewards, these tests evaluate generalization under controlled,
adversarial conditions that the agent never saw during training.

Three Validation Tests:

1. Annoyance Gauntlet: Test resilience across annoyance levels (0.1 to 1.0)
2. Zero-Shot Generalization: Test on completely unseen stage sequences
3. Lidar/Grayscale Challenge: Compare performance with different observation types

Key Principle: A robust agent shows graceful degradation, not catastrophic failure.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from gymnasium import spaces

# Stable Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, DummyVecEnv

# Highway environment
from environments.urban_junction_env import UrbanJunctionEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PHASE4_RESULTS_DIR = "results/phase4"
VALIDATION_MODELS_DIR = "models"  # Where trained models are stored

def create_validation_env(annoyance_level: float = 0.5, stage_sequence: list = None,
                         observation_type: str = "Kinematics", adaptive_difficulty: bool = False):
    """
    Create standardized validation environment.

    Args:
        annoyance_level: Fixed annoyance level (0.0 to 1.0)
        stage_sequence: Specific stage sequence for zero-shot testing
        observation_type: "Kinematics", "LidarObservation", or "GrayscaleObservation"
        adaptive_difficulty: Whether to use adaptive difficulty
    """
    config = UrbanJunctionEnv.default_config()

    config.update({
        # Observation type (can be overridden for sensor comparison)
        "observation": {
            "type": observation_type,
            "vehicles_count": 8,   # Optimized: Match training parameters
            "features": ["presence", "x", "y", "vx", "vy"] if observation_type == "Kinematics" else None,
            "absolute": False,
            "normalize": True,
        },

        # Environment settings
        "lanes_count": 2,
        "vehicles_count": 15,
        "vehicles_density": 1.0,
        "duration": 300,  # Long episodes for thorough testing

        # Stage configuration
        "stage_mode": "deterministic" if stage_sequence else "random",
        "stage_length_range": [150, 250],

        # Traffic settings
        "antagonistic_vehicles": annoyance_level > 0.0,
        "annoyance_level": annoyance_level,
        "adaptive_difficulty": adaptive_difficulty,

        # Reward structure (same as training)
        "normalize_reward": True,
        "collision_reward": 1.0,
        "speed_reward": 0.4,
        "speed_penalty_scale": 0.3,
        "progress_reward": 0.2,
        "traffic_light_penalty": 0.4,
        "traffic_light_reward": 0.1,
        "off_road_penalty": 0.3,
        "success_reward": 2.0,
        "stage_completion_reward": 0.5,
        "reward_speed_range": [20, 30],

        "offroad_terminal": False,
    })

    # Override stage sequence if specified
    if stage_sequence:
        config["stage_sequence"] = stage_sequence

    # Create environment
    env = UrbanJunctionEnv(config)
    env = Monitor(env, filename=os.path.join(PHASE4_RESULTS_DIR, f"validation_monitor_{annoyance_level}.csv"))

    # Only apply wrappers for Kinematics (Lidar/Grayscale have different dimensions)
    if observation_type == "Kinematics":
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        env = VecFrameStack(env, n_stack=2, channels_order='last')  # Optimized: Match training

        # Add context wrapper for Kinematics
        env = ValidationContextWrapper(env)

    return env

class ValidationContextWrapper:
    """Add context information for validation (same as training)."""

    def __init__(self, env):
        self.env = env
        original_space = env.observation_space
        context_dims = 3
        new_shape = (original_space.shape[0], original_space.shape[1] + context_dims)

        self.observation_space = spaces.Box(
            low=np.concatenate([original_space.low, np.zeros((original_space.shape[0], context_dims))], axis=1),
            high=np.concatenate([original_space.high, np.ones((original_space.shape[0], context_dims))], axis=1),
            dtype=original_space.dtype
        )

    def _get_context_one_hot(self, phase):
        context_map = {'highway': 0, 'merge': 1, 'intersection': 2}
        one_hot = np.zeros(3)
        if phase in context_map:
            one_hot[context_map[phase]] = 1.0
        return one_hot

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        phase = info.get('phase', 'highway')
        context = self._get_context_one_hot(phase)
        context_expanded = np.tile(context, (obs.shape[0], 1))
        obs_with_context = np.concatenate([obs, context_expanded], axis=1)
        return obs_with_context, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phase = info.get('phase', 'highway')
        context = self._get_context_one_hot(phase)
        context_expanded = np.tile(context, (obs.shape[0], 1))
        obs_with_context = np.concatenate([obs, context_expanded], axis=1)
        return obs_with_context, reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)

def evaluate_agent(model_path: str, env, num_episodes: int = 100,
                  deterministic: bool = True) -> dict:
    """
    Evaluate agent performance on a given environment.

    Returns comprehensive metrics for validation analysis.
    """
    logger.info(f"Evaluating {model_path} for {num_episodes} episodes...")

    # Load model
    model = PPO.load(model_path)

    results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'successes': [],
        'collisions': [],
        'stage_completions': [],
        'annoyance_levels': [],
        'phases_encountered': defaultdict(int),
    }

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        success = False
        collision = False
        stages_completed = 0
        annoyance_level = info.get('annoyance_level', 0.0)

        while not done and episode_length < 1000:  # Safety limit
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += float(reward)
            episode_length += 1

            # Track phase transitions
            if info.get('phase'):
                results['phases_encountered'][info['phase']] += 1

            # Check for stage completion
            if info.get('stage_completion_reward', 0) > 0:
                stages_completed += 1

            done = terminated or truncated

        # Record episode results
        results['episode_rewards'].append(episode_reward)
        results['episode_lengths'].append(episode_length)
        results['successes'].append(1 if info.get('success', False) else 0)
        results['collisions'].append(1 if terminated and not truncated else 0)
        results['stage_completions'].append(stages_completed)
        results['annoyance_levels'].append(annoyance_level)

        if episode % 20 == 0:
            logger.info(f"Episode {episode + 1}/{num_episodes}: "
                       f"Reward={episode_reward:.1f}, Success={success}, Length={episode_length}")

    # Compute summary statistics
    results['avg_reward'] = np.mean(results['episode_rewards'])
    results['std_reward'] = np.std(results['episode_rewards'])
    results['success_rate'] = np.mean(results['successes'])
    results['collision_rate'] = np.mean(results['collisions'])
    results['avg_episode_length'] = np.mean(results['episode_lengths'])

    logger.info(f"Evaluation complete: Avg Reward={results['avg_reward']:.2f} ± {results['std_reward']:.2f}, "
               f"Success Rate={results['success_rate']:.2%}")

    return results

def test_annoyance_gauntlet(model_path: str, annoyance_levels: list = None) -> dict:
    """
    Test 1: Annoyance Gauntlet - Resilience across annoyance levels.

    This test evaluates how gracefully the agent degrades under increasing
    adversarial traffic conditions. A robust agent shows steady decline,
    not catastrophic failure.
    """
    if annoyance_levels is None:
        annoyance_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

    logger.info("=== TEST 1: Annoyance Gauntlet ===")
    logger.info("Evaluating resilience across annoyance levels...")

    results = {}

    for annoyance in annoyance_levels:
        logger.info(f"Testing annoyance level: {annoyance}")

        # Create environment with fixed annoyance
        env = create_validation_env(
            annoyance_level=annoyance,
            adaptive_difficulty=False  # Fixed annoyance for controlled testing
        )

        # Evaluate agent
        eval_results = evaluate_agent(model_path, env, num_episodes=50)
        results[annoyance] = eval_results

        env.close()

        logger.info(f"Annoyance {annoyance}: Success Rate = {eval_results['success_rate']:.2%}")

    return results

def test_zero_shot_generalization(model_path: str) -> dict:
    """
    Test 2: Zero-Shot Generalization - Unseen stage sequences.

    This test evaluates whether the agent truly learned driving concepts
    rather than memorizing specific sequences from training.
    """
    logger.info("=== TEST 2: Zero-Shot Generalization ===")
    logger.info("Testing on completely unseen stage sequences...")

    # Define novel stage sequences never seen in training
    test_sequences = [
        [('intersection', 200), ('merge', 150), ('intersection', 200)],  # Hard start
        [('merge', 300), ('highway', 100), ('merge', 200)],            # Merge sandwich
        [('highway', 150), ('intersection', 100), ('merge', 150), ('highway', 100)],  # Complex mix
        [('intersection', 250), ('highway', 200), ('intersection', 150)], # Traffic light heavy
    ]

    results = {}

    for i, sequence in enumerate(test_sequences):
        logger.info(f"Testing sequence {i + 1}: {[s[0] for s in sequence]}")

        env = create_validation_env(
            annoyance_level=0.5,  # Moderate difficulty
            stage_sequence=sequence,
            adaptive_difficulty=False
        )

        eval_results = evaluate_agent(model_path, env, num_episodes=30)
        results[f"sequence_{i+1}"] = eval_results

        env.close()

        logger.info(f"Sequence {i+1}: Success Rate = {eval_results['success_rate']:.2%}")

    return results

def test_sensor_modalities(model_path: str) -> dict:
    """
    Test 3: Lidar/Grayscale Challenge - Sensor comparison.

    This test quantifies the performance cost of moving from perfect
    kinematic information to realistic sensor inputs.
    """
    logger.info("=== TEST 3: Lidar/Grayscale Challenge ===")
    logger.info("Comparing performance across observation types...")

    observation_types = ["Kinematics", "LidarObservation", "GrayscaleObservation"]
    results = {}

    for obs_type in observation_types:
        logger.info(f"Testing observation type: {obs_type}")

        try:
            env = create_validation_env(
                annoyance_level=0.3,  # Moderate difficulty for fair comparison
                observation_type=obs_type,
                adaptive_difficulty=False
            )

            # Note: For Lidar/Grayscale, we need different model loading logic
            # This is a simplified version - in practice you'd train separate models
            if obs_type == "Kinematics":
                eval_results = evaluate_agent(model_path, env, num_episodes=30)
            else:
                logger.warning(f"Skipping {obs_type} evaluation (requires separate trained model)")
                eval_results = {"skipped": True, "reason": "Requires separate model training"}

            results[obs_type] = eval_results
            env.close()

        except Exception as e:
            logger.warning(f"Failed to evaluate {obs_type}: {e}")
            results[obs_type] = {"error": str(e)}

    return results

def plot_validation_results(results: dict, test_name: str):
    """Create visualization plots for validation results."""
    try:
        plt.figure(figsize=(12, 8))

        if test_name == "annoyance_gauntlet":
            # Plot success rate vs annoyance level
            annoyance_levels = list(results.keys())
            success_rates = [results[level]['success_rate'] for level in annoyance_levels]

            plt.subplot(2, 2, 1)
            plt.plot(annoyance_levels, success_rates, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Annoyance Level')
            plt.ylabel('Success Rate')
            plt.title('Annoyance Gauntlet: Success Rate vs Difficulty')
            plt.grid(True, alpha=0.3)

            # Plot average reward vs annoyance level
            avg_rewards = [results[level]['avg_reward'] for level in annoyance_levels]

            plt.subplot(2, 2, 2)
            plt.plot(annoyance_levels, avg_rewards, 'ro-', linewidth=2, markersize=8)
            plt.xlabel('Annoyance Level')
            plt.ylabel('Average Reward')
            plt.title('Annoyance Gauntlet: Reward vs Difficulty')
            plt.grid(True, alpha=0.3)

        elif test_name == "zero_shot":
            # Bar chart of success rates for different sequences
            sequences = list(results.keys())
            success_rates = [results[seq]['success_rate'] for seq in sequences]

            plt.subplot(2, 2, 1)
            bars = plt.bar(range(len(sequences)), success_rates)
            plt.xlabel('Test Sequence')
            plt.ylabel('Success Rate')
            plt.title('Zero-Shot Generalization: Success by Sequence')
            plt.xticks(range(len(sequences)), [f'Seq {i+1}' for i in range(len(sequences))])
            plt.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, rate in zip(bars, success_rates):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{rate:.1%}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(PHASE4_RESULTS_DIR, f"{test_name}_results.png"), dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved {test_name} plots to {PHASE4_RESULTS_DIR}")

    except Exception as e:
        logger.warning(f"Failed to create plots for {test_name}: {e}")

def save_validation_report(results: dict, test_name: str, model_path: str):
    """Save detailed validation results to CSV and JSON."""
    try:
        # Convert results to DataFrame-friendly format
        report_data = []

        if test_name == "annoyance_gauntlet":
            for annoyance, metrics in results.items():
                report_data.append({
                    'annoyance_level': annoyance,
                    'avg_reward': metrics['avg_reward'],
                    'std_reward': metrics['std_reward'],
                    'success_rate': metrics['success_rate'],
                    'collision_rate': metrics['collision_rate'],
                    'avg_episode_length': metrics['avg_episode_length'],
                })

        elif test_name == "zero_shot":
            for sequence, metrics in results.items():
                report_data.append({
                    'sequence': sequence,
                    'avg_reward': metrics['avg_reward'],
                    'success_rate': metrics['success_rate'],
                    'collision_rate': metrics['collision_rate'],
                })

        # Save to CSV
        df = pd.DataFrame(report_data)
        csv_path = os.path.join(PHASE4_RESULTS_DIR, f"{test_name}_report.csv")
        df.to_csv(csv_path, index=False)

        logger.info(f"Saved {test_name} report to {csv_path}")

    except Exception as e:
        logger.warning(f"Failed to save {test_name} report: {e}")

def run_phase4_validation(model_path: str = "models/phase3/ppo_stage_c_final.zip"):
    """Execute complete Phase 4 validation suite."""

    logger.info("=== PHASE 4: Rigorous Validation (The Final Exam) ===")
    logger.info("Proving agent robustness through controlled adversarial testing")
    logger.info("=" * 60)

    # Create results directory
    Path(PHASE4_RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Test 1: Annoyance Gauntlet
    logger.info("Running Test 1: Annoyance Gauntlet...")
    gauntlet_results = test_annoyance_gauntlet(model_path)
    all_results['annoyance_gauntlet'] = gauntlet_results

    plot_validation_results(gauntlet_results, "annoyance_gauntlet")
    save_validation_report(gauntlet_results, "annoyance_gauntlet", model_path)

    # Test 2: Zero-Shot Generalization
    logger.info("Running Test 2: Zero-Shot Generalization...")
    generalization_results = test_zero_shot_generalization(model_path)
    all_results['zero_shot_generalization'] = generalization_results

    plot_validation_results(generalization_results, "zero_shot")
    save_validation_report(generalization_results, "zero_shot_generalization", model_path)

    # Test 3: Sensor Modalities (Kinematics only for now - requires separate training)
    logger.info("Running Test 3: Sensor Modalities...")
    sensor_results = test_sensor_modalities(model_path)
    all_results['sensor_modalities'] = sensor_results

    # Generate comprehensive validation report
    generate_validation_summary(all_results, model_path)

    logger.info("✓ Phase 4 validation completed!")
    logger.info(f"✓ Results saved to {PHASE4_RESULTS_DIR}")
    logger.info("✓ Check annoyance_gauntlet_results.png for robustness visualization")

    return all_results

def generate_validation_summary(all_results: dict, model_path: str):
    """Generate comprehensive validation summary."""
    summary_path = os.path.join(PHASE4_RESULTS_DIR, "validation_summary.txt")

    with open(summary_path, 'w') as f:
        f.write("PHASE 4 VALIDATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_path}\n\n")

        # Annoyance Gauntlet Summary
        if 'annoyance_gauntlet' in all_results:
            f.write("TEST 1: ANNOYANCE GAUNTLET\n")
            f.write("-" * 30 + "\n")

            gauntlet = all_results['annoyance_gauntlet']
            annoyance_levels = sorted(gauntlet.keys())

            f.write("Annoyance Level → Success Rate:\n")
            for level in annoyance_levels:
                success_rate = gauntlet[level]['success_rate']
                f.write(".1f")

            # Check for graceful degradation
            success_rates = [gauntlet[level]['success_rate'] for level in annoyance_levels]
            degradation_rate = (success_rates[0] - success_rates[-1]) / len(annoyance_levels)

            if degradation_rate < 0.3:  # Less than 30% drop per level
                robustness = "EXCELLENT: Graceful degradation"
            elif degradation_rate < 0.5:
                robustness = "GOOD: Steady decline"
            else:
                robustness = "CONCERNING: Sharp drop-off"

            f.write(f"\nRobustness Assessment: {robustness}\n\n")

        # Zero-Shot Generalization Summary
        if 'zero_shot_generalization' in all_results:
            f.write("TEST 2: ZERO-SHOT GENERALIZATION\n")
            f.write("-" * 35 + "\n")

            generalization = all_results['zero_shot_generalization']
            avg_success = np.mean([metrics['success_rate'] for metrics in generalization.values()])

            f.write(f"{avg_success:.1%}\n")
            f.write("Individual Sequence Results:\n")

            for seq_name, metrics in generalization.items():
                f.write(f"  {seq_name}: {metrics['success_rate']:.1%}\n")

            if avg_success > 0.8:
                generalization_assessment = "EXCELLENT: Strong generalization"
            elif avg_success > 0.6:
                generalization_assessment = "GOOD: Adequate generalization"
            else:
                generalization_assessment = "NEEDS IMPROVEMENT: Weak generalization"

            f.write(f"\nGeneralization Assessment: {generalization_assessment}\n\n")

        f.write("OVERALL ASSESSMENT\n")
        f.write("-" * 20 + "\n")

        # Provide final recommendation
        if ('annoyance_gauntlet' in all_results and
            'zero_shot_generalization' in all_results):

            gauntlet_robust = all_results['annoyance_gauntlet']
            final_success = gauntlet_robust[max(gauntlet_robust.keys())]['success_rate']
            generalization_avg = np.mean([m['success_rate'] for m in generalization.values()])

            if final_success > 0.7 and generalization_avg > 0.8:
                final_assessment = "EXCELLENT: Production-ready autonomous driver!"
            elif final_success > 0.5 and generalization_avg > 0.6:
                final_assessment = "GOOD: Capable autonomous driver with limitations"
            else:
                final_assessment = "NEEDS WORK: Not yet ready for complex scenarios"

            f.write(f"Final Assessment: {final_assessment}\n")

    logger.info(f"Validation summary saved to {summary_path}")

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/phase3/ppo_stage_c_final.zip"

    run_phase4_validation(model_path)
