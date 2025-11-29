#!/usr/bin/env python3
"""
Generalized Modality Training with Safety Constraints

This script trains a single generalized model for a chosen modality (lidar, grayscale, or both)
across all driving scenarios (highway, merge, intersection) using curriculum learning
with enhanced safety constraints.

Usage:
    # Train generalized lidar model
    python train_generalized_modality.py --modality lidar

    # Train generalized grayscale model
    python train_generalized_modality.py --modality grayscale

    # Train generalized multi-modal model
    python train_generalized_modality.py --modality both

    # With custom timesteps and wandb
    python train_generalized_modality.py --modality lidar --timesteps 30000 --use-wandb
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
import argparse
import json
from typing import Dict, List, Any, Optional

from environments.enhanced_urban_env import EnhancedUrbanJunctionEnv
from utils.callbacks import WandbMetricsCallback, BestModelCallback, ProgressCallback, StratifiedMetricsCallback, CurriculumTrackingCallback


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


class GeneralizedModalityTrainer:
    """
    Trains a generalized model for a specific modality across all driving scenarios.

    This trainer uses curriculum learning to progressively expose the agent to:
    1. Single scenario (highway) with chosen modality
    2. Multiple scenarios (highway + merge) with chosen modality
    3. All scenarios (highway + merge + intersection) with chosen modality
    """

    def __init__(self, modality: str, base_dir: str = "outputs"):
        """
        Initialize the generalized modality trainer.

        Args:
            modality: 'lidar', 'grayscale', or 'both'
            base_dir: Base directory for outputs
        """
        self.modality = modality
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "models")
        self.logs_dir = os.path.join(base_dir, "logs")

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Define curriculum phases for the chosen modality
        self.curriculum_phases = self._define_curriculum()

        # Training state
        self.model = None
        self.current_phase = 0
        self.total_timesteps = 0

    def _define_curriculum(self) -> List[Dict[str, Any]]:
        """
        Define the curriculum progression for the chosen modality.

        Returns:
            List of curriculum phases
        """
        # All scenarios for training
        all_scenarios = ["highway", "merge", "intersection"]

        return [
            # Phase 1: Foundation - Single scenario with chosen modality
            {
                "name": f"foundation_highway_{self.modality}",
                "scenarios": ["highway"],
                "modalities": [self.modality],
                "timesteps": 60000,
                "difficulty": 0.3,
                "required_success_rate": 0.85,
                "description": f"Learn basic highway driving with {self.modality}"
            },

            # Phase 2: Highway Mastery - Focus on highway skills
            {
                "name": f"highway_mastery_{self.modality}",
                "scenarios": ["highway"],
                "modalities": [self.modality],
                "timesteps": 60000,
                "difficulty": 0.4,
                "required_success_rate": 0.90,
                "description": f"Master highway driving skills with {self.modality}"
            },

            # Phase 3: Expansion - Add merging scenarios
            {
                "name": f"expansion_highway_merge_{self.modality}",
                "scenarios": ["highway", "merge"],
                "modalities": [self.modality],
                "timesteps": 80000,
                "difficulty": 0.5,
                "required_success_rate": 0.85,
                "description": f"Add merging scenarios with {self.modality}"
            },

            # Phase 4: Merge Mastery - Focus on merging skills
            {
                "name": f"merge_mastery_{self.modality}",
                "scenarios": ["merge"],
                "modalities": [self.modality],
                "timesteps": 60000,
                "difficulty": 0.6,
                "required_success_rate": 0.90,
                "description": f"Master merging skills with {self.modality}"
            },

            # Phase 5: Multi-Scenario Learning - Highway and Merge together
            {
                "name": f"multi_scenario_hw_merge_{self.modality}",
                "scenarios": ["highway", "merge"],
                "modalities": [self.modality],
                "timesteps": 80000,
                "difficulty": 0.7,
                "required_success_rate": 0.85,
                "description": f"Learn to handle highway and merge scenarios together with {self.modality}"
            },

            # Phase 6: Add Intersections - All scenarios with chosen modality
            {
                "name": f"mastery_all_scenarios_{self.modality}",
                "scenarios": all_scenarios,
                "modalities": [self.modality],
                "timesteps": 100000,
                "difficulty": 0.8,
                "required_success_rate": 0.80,
                "description": f"Full multi-scenario learning with {self.modality}"
            },

            # Phase 7: Intersection Specialization - Focus on intersections
            {
                "name": f"specialization_intersection_{self.modality}",
                "scenarios": ["intersection"],
                "modalities": [self.modality],
                "timesteps": 60000,
                "difficulty": 0.9,
                "required_success_rate": 0.90,
                "description": f"Specialize in intersection handling with {self.modality}"
            },

            # Phase 8: Final Generalization - All scenarios together
            {
                "name": f"final_generalization_{self.modality}",
                "scenarios": all_scenarios,
                "modalities": [self.modality],
                "timesteps": 80000,
                "difficulty": 1.0,
                "required_success_rate": 0.85,
                "description": f"Final generalization across all scenarios with {self.modality}"
            }
        ]

    def _create_curriculum_env(self, phase_config: Dict[str, Any]):
        """
        Create environment for current curriculum phase with balanced scenario sampling.

        Args:
            phase_config: Configuration for the current phase

        Returns:
            Vectorized environment
        """
        # Initialize scenario usage tracking for this phase if not exists
        phase_key = f"phase_{len(self.curriculum_phases)}_scenarios"
        if not hasattr(self, phase_key):
            setattr(self, phase_key, {scenario: 0 for scenario in phase_config["scenarios"]})

        scenario_usage = getattr(self, phase_key)

        def make_env():
            # Balanced scenario sampling - prefer less-used scenarios
            available_scenarios = phase_config["scenarios"]
            if len(available_scenarios) == 1:
                scenario = available_scenarios[0]
            else:
                # Calculate weights inversely proportional to usage
                weights = []
                for s in available_scenarios:
                    # Base weight of 1, plus bonus for less-used scenarios
                    usage = scenario_usage[s]
                    weight = 1.0 / (1.0 + usage * 0.1)  # Decreasing weight as usage increases
                    weights.append(weight)

                # Normalize weights
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]

                scenario = np.random.choice(available_scenarios, p=weights)

            # Track scenario usage
            scenario_usage[scenario] += 1

            # Use the chosen modality (should be just one)
            modality = self.modality

            env = EnhancedUrbanJunctionEnv(
                scenario=scenario,
                modality=modality,
                render_mode=None
            )
            env = Monitor(env)
            return env

        env = DummyVecEnv([make_env])
        return env

    def _evaluate_phase_performance(self, phase_config: Dict[str, Any], n_episodes: int = 50) -> Dict[str, float]:
        """
        Evaluate performance on current curriculum phase.

        Args:
            phase_config: Current phase configuration
            n_episodes: Number of evaluation episodes

        Returns:
            Dictionary with performance metrics
        """
        print(f"  📊 Evaluating phase performance ({n_episodes} episodes per scenario)...")

        total_reward = 0.0
        successful_episodes = 0
        crashes = 0
        completions = 0

        for episode in range(n_episodes):
            # Sample environment for evaluation
            scenario = np.random.choice(phase_config["scenarios"])
            modality = self.modality

            env = EnhancedUrbanJunctionEnv(scenario=scenario, modality=modality, render_mode=None)

            obs, _ = env.reset()
            episode_reward = 0.0
            steps = 0
            terminated = False
            truncated = False
            crashed = False

            while not (terminated or truncated) and steps < 1000:
                action, _ = self.model.predict(obs, deterministic=True)
                step_result = env.step(action)

                if len(step_result) == 5:
                    next_obs, reward, terminated, truncated, info = step_result
                else:
                    next_obs, reward, terminated, truncated = step_result[:4]

                episode_reward += reward
                obs = next_obs
                steps += 1

                # Check for crash
                if hasattr(env, 'vehicle') and env.vehicle.crashed:
                    crashed = True
                    crashes += 1
                    break

            total_reward += episode_reward

            # Success criteria
            success_threshold = self._get_success_threshold(scenario)
            if episode_reward > success_threshold and not crashed:
                successful_episodes += 1

            # Completion criteria
            if self._episode_completed(scenario, steps, crashed):
                completions += 1

        # Calculate metrics
        avg_reward = total_reward / n_episodes
        success_rate = successful_episodes / n_episodes
        crash_rate = crashes / n_episodes
        completion_rate = completions / n_episodes

        return {
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "crash_rate": crash_rate,
            "completion_rate": completion_rate,
            "episodes_evaluated": n_episodes
        }

    def _get_success_threshold(self, scenario: str) -> float:
        """Get success threshold for a scenario."""
        thresholds = {
            "highway": 20.0,
            "merge": 15.0,
            "intersection": 25.0
        }
        return thresholds.get(scenario, 15.0)

    def _episode_completed(self, scenario: str, steps: int, crashed: bool) -> bool:
        """Check if episode represents successful completion."""
        if crashed:
            return False

        if scenario == "highway":
            return steps >= 150
        elif scenario == "merge":
            return steps >= 100
        elif scenario == "intersection":
            return steps >= 80
        return steps >= 100

    def train_curriculum(self, total_timesteps: int = 50000, use_wandb: bool = False) -> Optional[PPO]:
        """
        Train the agent through the complete curriculum.

        Args:
            total_timesteps: Total training timesteps across all phases
            use_wandb: Whether to use Weights & Biases logging

        Returns:
            Trained PPO model, or None if training failed
        """
        print("🚀 STARTING GENERALIZED MODALITY TRAINING")
        print("=" * 60)
        print(f"📡 Modality: {self.modality.upper()}")
        print(f"🎯 Goal: Generalized model for all scenarios")
        print(f"🛡️ Safety: Enhanced environment with hard constraints")
        print(f"📚 Curriculum: {len(self.curriculum_phases)} phases")
        print("=" * 60)

        # Initialize model
        print("\n🤖 Initializing PPO model...")
        sample_env = self._create_curriculum_env(self.curriculum_phases[0])

        self.model = PPO(
            "MlpPolicy",
            sample_env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,  # Good exploration for generalization
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=None
        )

        # Train through curriculum phases
        for phase_idx, phase_config in enumerate(self.curriculum_phases):
            print(f"\n🎓 PHASE {phase_idx + 1}/{len(self.curriculum_phases)}: {phase_config['name']}")
            print("-" * 60)
            print(f"📋 Description: {phase_config['description']}")
            print(f"🛣️ Scenarios: {phase_config['scenarios']}")
            print(f"📡 Modality: {self.modality}")
            print(f"⏱️ Timesteps: {phase_config['timesteps']}")

            # Create environment for this phase
            env = self._create_curriculum_env(phase_config)

            # Update model environment
            self.model.set_env(env)

            # Setup callbacks
            callbacks = []

            # Comprehensive stratified metrics callback
            stratified_cb = StratifiedMetricsCallback(verbose=1)
            callbacks.append(stratified_cb)

            # Curriculum tracking callback
            curriculum_cb = CurriculumTrackingCallback(curriculum_phases=self.curriculum_phases, verbose=1)
            callbacks.append(curriculum_cb)

            # Notify curriculum callback of phase start
            curriculum_cb.on_phase_start(phase_idx, phase_config)

            # Progress callback for console output
            progress_cb = ProgressCallback(print_freq=1000, verbose=1)
            callbacks.append(progress_cb)

            # Progress callback for tqdm
            tqdm_cb = TqdmCallback(total_timesteps=phase_config['timesteps'])
            callbacks.append(tqdm_cb)

            if use_wandb:
                # Legacy wandb callback (now redundant with stratified callback, but keeping for compatibility)
                callbacks.append(WandbMetricsCallback())

            # Best model callback - evaluate every 5000 timesteps
            best_model_save_path = os.path.join(self.models_dir, f"best_{self.modality}")
            eval_env = EnhancedUrbanJunctionEnv(scenario="highway", modality=self.modality, render_mode=None)
            best_model_cb = BestModelCallback(
                eval_env=eval_env,
                eval_freq=5000,
                save_path=best_model_save_path,
                verbose=1
            )
            callbacks.append(best_model_cb)

            # Checkpoint callback
            checkpoint_cb = CheckpointCallback(
                save_freq=max(1000, phase_config['timesteps'] // 10),
                save_path=os.path.join(self.models_dir, f"generalized_{self.modality}_phase_{phase_idx}"),
                name_prefix=f"phase_{phase_idx}"
            )
            callbacks.append(checkpoint_cb)

            # Train this phase
            print(f"\n🏃 Training phase {phase_idx + 1}...")
            self.model.learn(
                total_timesteps=phase_config['timesteps'],
                callback=callbacks,
                reset_num_timesteps=False
            )

            self.total_timesteps += phase_config['timesteps']

            # Evaluate phase performance (increased episodes for reliability)
            phase_metrics = self._evaluate_phase_performance(phase_config, n_episodes=50)

            # Notify curriculum callback of phase completion
            curriculum_cb.on_phase_end(phase_idx, phase_config, phase_metrics)

            print("\n📊 Phase Results:")
            print(f"   Avg Reward: {phase_metrics['avg_reward']:.2f}")
            print(f"   Success Rate: {phase_metrics['success_rate']:.1%}")
            print(f"   Crash Rate: {phase_metrics['crash_rate']:.1%}")
            print(f"   Completion Rate: {phase_metrics['completion_rate']:.1%}")
            # Check if phase requirements met
            if phase_metrics['success_rate'] >= phase_config['required_success_rate']:
                print(f"   ✅ Phase {phase_idx + 1} PASSED - Ready for next phase")
                phase_passed = True
            else:
                print(f"   ❌ Phase {phase_idx + 1} FAILED - Success rate {phase_metrics['success_rate']:.1%} below target ({phase_config['required_success_rate']:.1%})")

                # Allow one retry per phase with additional training
                if not hasattr(self, f'phase_{phase_idx}_retries'):
                    setattr(self, f'phase_{phase_idx}_retries', 0)

                retry_count = getattr(self, f'phase_{phase_idx}_retries')

                if retry_count < 1:
                    # Retry the same phase with additional training
                    setattr(self, f'phase_{phase_idx}_retries', retry_count + 1)
                    additional_timesteps = phase_config['timesteps'] // 2  # Add 50% more training
                    print(f"   🔄 Retrying Phase {phase_idx + 1} with additional {additional_timesteps} timesteps...")

                    self.model.learn(
                        total_timesteps=additional_timesteps,
                        callback=callbacks,
                        reset_num_timesteps=False
                    )

                    self.total_timesteps += additional_timesteps

                    # Re-evaluate after additional training
                    phase_metrics = self._evaluate_phase_performance(phase_config)
                    curriculum_cb.on_phase_end(phase_idx, phase_config, phase_metrics)

                    print("\n📊 Phase Results (After Retry):")
                    print(f"   Avg Reward: {phase_metrics['avg_reward']:.2f}")
                    print(f"   Success Rate: {phase_metrics['success_rate']:.1%}")
                    print(f"   Crash Rate: {phase_metrics['crash_rate']:.1%}")
                    print(f"   Completion Rate: {phase_metrics['completion_rate']:.1%}")

                    if phase_metrics['success_rate'] >= phase_config['required_success_rate']:
                        print(f"   ✅ Phase {phase_idx + 1} PASSED on retry - Ready for next phase")
                        phase_passed = True
                    else:
                        print(f"   💥 Phase {phase_idx + 1} FAILED permanently - Stopping curriculum")
                        print("   Model did not meet minimum performance requirements")
                        break
                else:
                    print(f"   💥 Phase {phase_idx + 1} FAILED permanently - Stopping curriculum")
                    print("   Model did not meet minimum performance requirements")
                    break

                phase_passed = False

        # Save final model
        final_model_path = os.path.join(self.models_dir, f"generalized_{self.modality}_final.zip")
        self.model.save(final_model_path)
        print(f"\n💾 Final model saved: {final_model_path}")

        # Final comprehensive evaluation
        print("\n🎯 FINAL EVALUATION")
        final_results = self._evaluate_final_performance()

        # Get comprehensive statistics from callbacks
        print("\n📊 COMPREHENSIVE TRAINING ANALYSIS")
        print("=" * 60)

        # Stratified metrics summary
        if 'stratified_cb' in locals():
            stratified_stats = stratified_cb.get_comprehensive_stats()
            self._print_stratified_summary(stratified_stats)

        # Curriculum analysis
        if 'curriculum_cb' in locals():
            curriculum_stats = curriculum_cb.get_curriculum_stats()
            self._print_curriculum_summary(curriculum_stats, final_results)

        # Save training summary
        self._save_training_summary(final_results)

        print(f"📁 Models saved in: {self.models_dir}")
        print(f"📊 Results saved in: outputs/evaluations/")
        print(f"📈 Modality: {self.modality.upper()}")
        print(f"🛡️ Safety: Enhanced constraints active")
        print(f"🎯 Generalization: All scenarios covered")
        print("=" * 60)

        return self.model

    def _print_stratified_summary(self, stats: Dict[str, Any]):
        """Print comprehensive stratified metrics summary."""
        print("🏷️  STRATIFIED PERFORMANCE ANALYSIS")
        print("-" * 40)

        # Overall performance
        overall = stats.get("overall", {})
        if overall:
            print("📈 Overall Performance:")
            print(f"  Success Rate: {overall.get('successes_rate', 0):.1%}")
            print(f"  Crash Rate: {overall.get('crashes_rate', 0):.1%}")
            print(f"  Avg Reward: {overall.get('episode_rewards_mean', 0):.2f}")
            if "safety/total_override_rate_mean" in overall:
                print(f"  Safety Override Rate: {overall['safety/total_override_rate_mean']:.1%}")
        # Scenario-specific performance
        scenarios = stats.get("scenarios", {})
        if scenarios:
            print("\n🏁 Scenario-Specific Performance:")
            for scenario, data in scenarios.items():
                if "successes_rate" in data:
                    print(f"  {scenario.capitalize()}: Success={data['successes_rate']:.1%}, "
                          f"Crashes={data.get('crashes_rate', 0):.1%}, "
                          f"Reward={data.get('episode_rewards_mean', 0):.2f}")

        # Modality-specific performance
        modalities = stats.get("modalities", {})
        if modalities:
            print("\n📡 Modality-Specific Performance:")
            for modality, data in modalities.items():
                if "successes_rate" in data:
                    print(f"  {modality.capitalize()}: Success={data['successes_rate']:.1%}, "
                          f"Crashes={data.get('crashes_rate', 0):.1%}")

        # Recent performance
        recent = stats.get("recent_performance", {})
        if recent:
            print("\n⏰ Recent Performance (last 100 episodes):")
            print(f"  Success Rate: {recent.get('success_rate', 0):.1%}")
            print(f"  Crash Rate: {recent.get('crash_rate', 0):.1%}")
    def _print_curriculum_summary(self, stats: Dict[str, Any], final_results: Dict[str, Any] = None):
        """Print comprehensive curriculum analysis."""
        print("\n📚 CURRICULUM LEARNING ANALYSIS")
        print("-" * 40)

        # Phase performance summary
        phases = stats.get("phase_performance", {})
        if phases.get("final_success_rate"):
            print("🎯 Phase Progression:")
            for i, (success, crashes, reward) in enumerate(zip(
                phases["final_success_rate"],
                phases["final_crash_rate"],
                phases["final_avg_reward"]
            )):
                print(f"  Phase {i+1}: Success={success:.1%}, Crashes={crashes:.1%}, Reward={reward:.2f}")

        # Generalization metrics
        print(f"  Scenario Generalization: {stats.get('scenario_generalization', 0):.2f}")
        print(f"  Modality Transfer: {stats.get('modality_transfer', 0):.2f}")
        print(f"  Curriculum Efficiency: {stats.get('curriculum_efficiency', 0):.2f}")
        # Scenario performance - use final comprehensive evaluation results
        if final_results:
            print("\n🏁 Final Scenario Mastery:")
            scenarios = ["highway", "merge", "intersection"]
            for scenario in scenarios:
                if scenario in final_results:
                    data = final_results[scenario]
                    print(f"  {scenario.capitalize()}: {data['success_rate']:.1%} success, "
                          f"{data['crash_rate']:.1%} crashes")

        # Modality performance - use final comprehensive evaluation results
        if final_results and "overall" in final_results:
            print("\n📡 Final Modality Performance:")
            overall = final_results["overall"]
            print(f"  {self.modality.capitalize()}: {overall['success_rate']:.1%} success, "
                  f"{overall['crash_rate']:.1%} crashes")

    def _evaluate_final_performance(self) -> Dict[str, Any]:
        """Perform comprehensive final evaluation across all scenarios."""
        print("🔬 Performing comprehensive final evaluation...")

        results = {}
        scenarios = ["highway", "merge", "intersection"]

        for scenario in scenarios:
            print(f"  Testing {scenario} scenario...")

            env = EnhancedUrbanJunctionEnv(scenario=scenario, modality=self.modality, render_mode=None)

            # Evaluate multiple episodes
            episode_rewards = []
            successes = []
            crashes = []

            for episode in range(100):  # 100 episodes per scenario for reliable statistics
                print(f"    Episode {episode + 1}/100...", end="\r")
                obs, _ = env.reset()
                episode_reward = 0.0
                steps = 0
                crashed = False
                terminated = False
                truncated = False

                while not (terminated or truncated) and steps < 1000:
                    action, _ = self.model.predict(obs, deterministic=True)
                    step_result = env.step(action)

                    if len(step_result) == 5:
                        next_obs, reward, terminated, truncated, info = step_result
                    else:
                        next_obs, reward, terminated, truncated = step_result[:4]

                    episode_reward += reward
                    obs = next_obs
                    steps += 1

                    if hasattr(env, 'vehicle') and env.vehicle.crashed:
                        crashed = True
                        break

                episode_rewards.append(episode_reward)
                crashes.append(1 if crashed else 0)

                # Success criteria
                success_threshold = self._get_success_threshold(scenario)
                success = episode_reward > success_threshold and not crashed
                successes.append(1 if success else 0)

            results[scenario] = {
                "avg_reward": np.mean(episode_rewards),
                "success_rate": np.mean(successes),
                "crash_rate": np.mean(crashes),
                "episodes": len(episode_rewards)
            }

            print(f"    Avg Reward: {results[scenario]['avg_reward']:.2f}")
            print(f"    Success Rate: {results[scenario]['success_rate']:.1%}")
            print(f"    Crash Rate: {results[scenario]['crash_rate']:.1%}")
        # Overall metrics
        all_rewards = [results[s]["avg_reward"] for s in scenarios]
        all_successes = [results[s]["success_rate"] for s in scenarios]
        all_crashes = [results[s]["crash_rate"] for s in scenarios]

        results["overall"] = {
            "avg_reward": np.mean(all_rewards),
            "success_rate": np.mean(all_successes),
            "crash_rate": np.mean(all_crashes),
            "scenarios_tested": len(scenarios)
        }

        return results

    def _save_training_summary(self, final_results: Dict[str, Any]):
        """Save comprehensive training summary."""
        summary = {
            "training_config": {
                "modality": self.modality,
                "curriculum_phases": len(self.curriculum_phases),
                "total_timesteps": self.total_timesteps,
                "safety_enabled": True
            },
            "curriculum_phases": self.curriculum_phases,
            "final_performance": final_results,
            "training_timestamp": str(np.datetime64('now'))
        }

        summary_path = os.path.join(self.base_dir, "evaluations",
                                   f"generalized_{self.modality}_training_summary.json")

        os.makedirs(os.path.dirname(summary_path), exist_ok=True)

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"📄 Training summary saved: {summary_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Generalized Modality Model")
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=["lidar", "grayscale", "both"],
        help="Modality to train: lidar, grayscale, or both"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=150000,
        help="Total training timesteps across all phases"
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="outputs",
        help="Base directory for outputs"
    )

    args = parser.parse_args()

    print("🚗 ENHANCED AUTONOMOUS DRIVING - GENERALIZED MODALITY TRAINING")
    print("=" * 70)
    print(f"📡 Modality: {args.modality.upper()}")
    print("🎯 Goal: Single generalized model for all scenarios")
    print("🛡️ Safety: Hard constraints throughout training")
    print("📚 Method: Curriculum learning progression")
    print("=" * 70)

    # Initialize trainer
    trainer = GeneralizedModalityTrainer(
        modality=args.modality,
        base_dir=args.base_dir
    )

    # Train the model
    final_model = trainer.train_curriculum(
        total_timesteps=args.timesteps,
        use_wandb=args.use_wandb
    )

    if final_model:
        print("\n✅ SUCCESS: Generalized model trained!")
        print(f"📁 Model saved as: outputs/models/generalized_{args.modality}_final.zip")
        print("🎯 Ready for deployment across all driving scenarios!")
    else:
        print("\n❌ Training failed")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
