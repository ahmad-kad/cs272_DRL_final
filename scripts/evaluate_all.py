#!/usr/bin/env python3
"""
Comprehensive Evaluation System for Autonomous Driving Agents

This script provides thorough evaluation of trained models across all scenarios,
modalities, and performance metrics with detailed reporting and visualization.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from stable_baselines3 import PPO
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Import project modules
import sys
sys.path.append('..')
from environments.enhanced_urban_env import EnhancedUrbanJunctionEnv


class ComprehensiveEvaluator:
    """Comprehensive evaluation system for autonomous driving models."""

    def __init__(self, results_dir="results/evaluations"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Evaluation scenarios and modalities
        self.scenarios = ["highway", "merge", "intersection"]
        self.modalities = ["lidar", "grayscale", "both"]
        self.models_dir = Path("results/models")

        # Performance metrics to track
        self.metrics = [
            "avg_reward", "success_rate", "crash_rate", "completion_rate",
            "avg_episode_length", "safety_score", "efficiency_score"
        ]

        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")

    def discover_models(self):
        """Discover all trained models in the models directory."""
        models = {}

        if not self.models_dir.exists():
            print("[ERROR] No models directory found")
            return models

        # Look for .zip model files
        for model_file in self.models_dir.glob("**/*.zip"):
            model_name = model_file.stem
            model_path = str(model_file)

            # Extract metadata from filename
            metadata = self._parse_model_metadata(model_name)

            if metadata:
                models[model_name] = {
                    "path": model_path,
                    "metadata": metadata,
                    "file": model_file
                }

        print(f"[FOLDER] Found {len(models)} trained models")
        return models

    def _parse_model_metadata(self, model_name):
        """Parse model metadata from filename."""
        metadata = {
            "name": model_name,
            "scenario": None,
            "modality": None,
            "training_type": "unknown"
        }

        # Parse common patterns
        if "enhanced" in model_name:
            metadata["training_type"] = "enhanced"
        elif "curriculum" in model_name:
            metadata["training_type"] = "curriculum"
        elif "baseline" in model_name:
            metadata["training_type"] = "baseline"

        # Extract scenario
        for scenario in self.scenarios:
            if scenario in model_name:
                metadata["scenario"] = scenario
                break

        # Extract modality
        for modality in self.modalities:
            if modality in model_name:
                metadata["modality"] = modality
                break

        return metadata

    def evaluate_model(self, model_path, model_name, n_episodes=20):
        """Evaluate a single model comprehensively."""
        print(f"\n[LAB] Evaluating {model_name}...")

        try:
            model = PPO.load(model_path)
        except Exception as e:
            print(f"[ERROR] Failed to load model {model_path}: {e}")
            return None

        results = {}

        # Evaluate on all scenario-modality combinations
        for scenario in self.scenarios:
            for modality in self.modalities:
                test_name = f"{scenario}_{modality}"

                print(f"  Testing {test_name}...")

                # Create test environment
                env = EnhancedUrbanJunctionEnv(
                    scenario=scenario,
                    modality=modality,
                    render_mode=None
                )

                # Run evaluation episodes
                metrics = self._evaluate_on_environment(model, env, n_episodes)
                results[test_name] = metrics

        # Calculate aggregate metrics
        results["aggregate"] = self._calculate_aggregate_metrics(results)

        return results

    def _evaluate_on_environment(self, model, env, n_episodes):
        """Evaluate model on a specific environment."""
        episode_rewards = []
        episode_lengths = []
        successful_episodes = 0
        crashes = 0
        completions = 0
        safety_overrides = []

        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            crashed = False

            while not done and episode_length < 500:  # Max episode length
                action, _ = model.predict(obs, deterministic=True)

                # Execute action
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_obs, reward, done, info = step_result

                episode_reward += reward
                episode_length += 1
                obs = next_obs

                # Track crashes
                if hasattr(env, 'vehicle') and env.vehicle.crashed and not crashed:
                    crashed = True
                    crashes += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Success criteria (adjust based on scenario)
            success_threshold = self._get_success_threshold(env.current_scenario)
            if episode_reward > success_threshold and not crashed:
                successful_episodes += 1

            # Completion criteria
            if self._episode_completed(env, episode_length, crashed):
                completions += 1

            # Track safety overrides
            if hasattr(env, 'get_safety_stats'):
                safety_stats = env.get_safety_stats()
                safety_overrides.append(safety_stats['safety_override_count'])

        # Calculate metrics
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_length = np.mean(episode_lengths)
        success_rate = successful_episodes / n_episodes
        crash_rate = crashes / n_episodes
        completion_rate = completions / n_episodes

        # Safety and efficiency scores
        safety_score = 1.0 - crash_rate  # Higher is better
        efficiency_score = min(1.0, avg_reward / 50.0)  # Normalize efficiency

        return {
            "episodes": n_episodes,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "avg_episode_length": avg_length,
            "success_rate": success_rate,
            "crash_rate": crash_rate,
            "completion_rate": completion_rate,
            "safety_score": safety_score,
            "efficiency_score": efficiency_score,
            "avg_safety_overrides": np.mean(safety_overrides) if safety_overrides else 0
        }

    def _get_success_threshold(self, scenario):
        """Get success threshold for a scenario."""
        thresholds = {
            "highway": 20.0,      # Basic highway navigation
            "merge": 15.0,        # Successful merging
            "intersection": 25.0  # Complex intersection navigation
        }
        return thresholds.get(scenario, 15.0)

    def _episode_completed(self, env, episode_length, crashed):
        """Check if episode represents successful completion."""
        if crashed:
            return False

        scenario = env.current_scenario
        if scenario == "highway":
            return episode_length >= 150  # Survived reasonable distance
        elif scenario == "merge":
            return episode_length >= 100  # Made it through merge zone
        elif scenario == "intersection":
            return episode_length >= 80   # Cleared intersection
        return episode_length >= 100      # Default

    def _calculate_aggregate_metrics(self, results):
        """Calculate aggregate metrics across all test scenarios."""
        scenario_results = [metrics for key, metrics in results.items() if key != "aggregate"]

        if not scenario_results:
            return {}

        # Weighted average across all scenarios (equal weight per scenario-modality)
        avg_reward = np.mean([r["avg_reward"] for r in scenario_results])
        success_rate = np.mean([r["success_rate"] for r in scenario_results])
        crash_rate = np.mean([r["crash_rate"] for r in scenario_results])
        completion_rate = np.mean([r["completion_rate"] for r in scenario_results])
        safety_score = np.mean([r["safety_score"] for r in scenario_results])
        efficiency_score = np.mean([r["efficiency_score"] for r in scenario_results])

        return {
            "overall_avg_reward": avg_reward,
            "overall_success_rate": success_rate,
            "overall_crash_rate": crash_rate,
            "overall_completion_rate": completion_rate,
            "overall_safety_score": safety_score,
            "overall_efficiency_score": efficiency_score,
            "scenarios_tested": len([k for k in results.keys() if k != "aggregate"])
        }

    def run_full_evaluation(self):
        """Run comprehensive evaluation of all discovered models."""
        print("[ROCKET] STARTING COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)

        # Discover models
        models = self.discover_models()
        if not models:
            print("[ERROR] No models found to evaluate")
            return None

        # Evaluate each model
        evaluation_results = {}

        for model_name, model_info in tqdm(models.items(), desc="Evaluating Models"):
            results = self.evaluate_model(
                model_info["path"],
                model_name
            )

            if results:
                evaluation_results[model_name] = {
                    "results": results,
                    "metadata": model_info["metadata"],
                    "evaluation_timestamp": datetime.now().isoformat()
                }

        # Save results
        self.save_evaluation_results(evaluation_results)

        # Generate reports and visualizations
        self.generate_evaluation_report(evaluation_results)
        self.create_visualizations(evaluation_results)

        print("\n" + "=" * 60)
        print("[CELEBRATE] EVALUATION COMPLETE!")
        print("=" * 60)

        return evaluation_results

    def save_evaluation_results(self, results):
        """Save evaluation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_evaluation_{timestamp}.json"

        filepath = self.results_dir / filename

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"[SAVE] Results saved to: {filepath}")

    def generate_evaluation_report(self, results):
        """Generate a comprehensive evaluation report."""
        print("\n[CHART] GENERATING EVALUATION REPORT")

        if not results:
            print("[ERROR] No results to report")
            return

        # Create summary DataFrame
        summary_data = []
        for model_name, data in results.items():
            agg = data["results"].get("aggregate", {})
            metadata = data["metadata"]

            summary_data.append({
                "Model": model_name,
                "Training_Type": metadata.get("training_type", "unknown"),
                "Primary_Scenario": metadata.get("scenario", "mixed"),
                "Primary_Modality": metadata.get("modality", "mixed"),
                "Avg_Reward": agg.get("overall_avg_reward", 0),
                "Success_Rate": agg.get("overall_success_rate", 0),
                "Crash_Rate": agg.get("overall_crash_rate", 0),
                "Safety_Score": agg.get("overall_safety_score", 0),
                "Efficiency_Score": agg.get("overall_efficiency_score", 0)
            })

        df = pd.DataFrame(summary_data)

        # Sort by overall performance (weighted score)
        df["Overall_Score"] = (
            df["Safety_Score"] * 0.4 +      # Safety most important
            df["Success_Rate"] * 0.3 +      # Task completion
            df["Efficiency_Score"] * 0.2 +  # Efficiency
            (1 - df["Crash_Rate"]) * 0.1    # Low crashes
        )

        df = df.sort_values("Overall_Score", ascending=False)

        # Print summary report
        print("\n[TROPHY] MODEL PERFORMANCE RANKING:")
        print("=" * 80)
        print(df.to_string(index=False, float_format="%.3f"))

        # Best model analysis
        if len(df) > 0:
            best_model = df.iloc[0]
            print("\n[TARGET] BEST PERFORMING MODEL:")
            print(f"   Name: {best_model['Model']}")
            print(".3f")
            print(".1%")
            print(".1%")
            print(".3f")

        # Save detailed report
        report_file = self.results_dir / "evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write("COMPREHENSIVE MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models Evaluated: {len(results)}\n\n")
            f.write("PERFORMANCE RANKING:\n")
            f.write("-" * 50 + "\n")
            f.write(df.to_string(index=False, float_format="%.3f"))
            f.write("\n\nDETAILED RESULTS:\n")
            f.write("-" * 50 + "\n")
            f.write(json.dumps(results, indent=2, default=str))

        print(f"\n[DOC] Detailed report saved to: {report_file}")

    def _prepare_visualization_data(self, results):
        """Prepare data for visualization from evaluation results."""
        plot_data = []
        for model_name, data in results.items():
            metadata = data["metadata"]
            agg = data["results"].get("aggregate", {})

            plot_data.append({
                "model": model_name,
                "training_type": metadata.get("training_type", "unknown"),
                "avg_reward": agg.get("overall_avg_reward", 0),
                "success_rate": agg.get("overall_success_rate", 0),
                "crash_rate": agg.get("overall_crash_rate", 0),
                "safety_score": agg.get("overall_safety_score", 0),
                "efficiency_score": agg.get("overall_efficiency_score", 0)
            })

        return pd.DataFrame(plot_data)

    def _create_reward_comparison_plot(self, df, ax):
        """Create reward comparison bar plot."""
        bars = ax.bar(range(len(df)), df['avg_reward'], color='skyblue')
        ax.set_title('Average Reward by Model')
        ax.set_ylabel('Reward')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['model'], rotation=45, ha='right')

        # Add value labels
        for bar, value in zip(bars, df['avg_reward']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    '.1f', ha='center', va='bottom', fontsize=8)

    def _create_success_rate_plot(self, df, ax):
        """Create success rate comparison bar plot."""
        success_bars = ax.bar(range(len(df)), df['success_rate'], color='lightgreen')
        ax.set_title('Success Rate by Model')
        ax.set_ylabel('Success Rate')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['model'], rotation=45, ha='right')

        for bar, value in zip(success_bars, df['success_rate']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    '.1%', ha='center', va='bottom', fontsize=8)

    def _create_safety_metrics_plot(self, df, ax):
        """Create safety metrics plot."""
        safety_bars = ax.bar(range(len(df)), df['safety_score'], color='salmon')
        crash_bars = ax.bar(range(len(df)), -df['crash_rate'], color='red', alpha=0.7)
        ax.set_title('Safety Metrics')
        ax.set_ylabel('Safety Score / -Crash Rate')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['model'], rotation=45, ha='right')

    def _create_training_type_comparison(self, df, ax):
        """Create training type comparison plot."""
        if 'training_type' in df.columns:
            training_comparison = df.groupby('training_type')[['avg_reward', 'success_rate']].mean()
            training_comparison.plot(kind='bar', ax=ax, colormap='viridis')
            ax.set_title('Performance by Training Type')
            ax.set_ylabel('Score')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    def _create_radar_chart(self, df, ax):
        """Create radar chart for best model."""
        best_model = df.loc[df['avg_reward'].idxmax()]
        categories = ['Safety', 'Success Rate', 'Efficiency']
        values = [best_model['safety_score'], best_model['success_rate'], best_model['efficiency_score']]

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Close the circle
        angles += angles[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=best_model['model'])
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title(f'Best Model Profile: {best_model["model"]}')
        ax.grid(True)

    def _create_performance_distribution_plot(self, df, ax):
        """Create performance distribution box plot."""
        df[['avg_reward', 'success_rate', 'safety_score']].boxplot(ax=ax)
        ax.set_title('Performance Distribution')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)

    def _create_model_comparison_dashboard(self, df):
        """Create the main model comparison dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Model Evaluation Dashboard', fontsize=16)

        # Create individual plots
        self._create_reward_comparison_plot(df, axes[0, 0])
        self._create_success_rate_plot(df, axes[0, 1])
        self._create_safety_metrics_plot(df, axes[0, 2])
        self._create_training_type_comparison(df, axes[1, 0])
        self._create_radar_chart(df, axes[1, 1])
        self._create_performance_distribution_plot(df, axes[1, 2])

        plt.tight_layout()
        return fig

    def create_visualizations(self, results):
        """Create comprehensive visualizations of evaluation results."""
        print("\n[TREND] GENERATING VISUALIZATIONS")

        if not results:
            return

        # Prepare data
        df = self._prepare_visualization_data(results)

        if df.empty:
            print("[ERROR] No data for visualization")
            return

        # Create comprehensive dashboard
        fig = self._create_model_comparison_dashboard(df)

        # Save dashboard
        dashboard_file = self.results_dir / "evaluation_dashboard.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        print(f"[CHART] Dashboard saved to: {dashboard_file}")

        # Create individual scenario plots
        self._create_scenario_comparison_plots(results)

    def _create_scenario_comparison_plots(self, results):
        """Create detailed scenario-by-scenario comparison plots."""
        print("  Creating scenario comparison plots...")

        # Collect data by scenario
        scenario_data = {}
        for model_name, data in results.items():
            model_results = data["results"]
            for test_name, metrics in model_results.items():
                if test_name == "aggregate":
                    continue

                scenario = test_name.split('_')[0]
                modality = test_name.split('_')[1]

                if scenario not in scenario_data:
                    scenario_data[scenario] = {}

                if model_name not in scenario_data[scenario]:
                    scenario_data[scenario][model_name] = {}

                scenario_data[scenario][model_name][modality] = metrics

        # Create scenario plots
        for scenario, model_data in scenario_data.items():
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'{scenario.upper()} Scenario Performance', fontsize=14)

            modalities = ['lidar', 'grayscale', 'both']

            for i, modality in enumerate(modalities):
                ax = axes[i]

                model_names = []
                rewards = []
                success_rates = []

                for model_name, modalities_data in model_data.items():
                    if modality in modalities_data:
                        metrics = modalities_data[modality]
                        model_names.append(model_name)
                        rewards.append(metrics.get('avg_reward', 0))
                        success_rates.append(metrics.get('success_rate', 0))

                if model_names:
                    x = np.arange(len(model_names))
                    ax.bar(x - 0.2, rewards, 0.4, label='Reward', alpha=0.7)
                    ax.bar(x + 0.2, success_rates, 0.4, label='Success Rate', alpha=0.7)

                    ax.set_title(f'{modality.upper()} Modality')
                    ax.set_ylabel('Score')
                    ax.set_xticks(x)
                    ax.set_xticklabels(model_names, rotation=45, ha='right')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

            plt.tight_layout()
            scenario_file = self.results_dir / f"{scenario}_comparison.png"
            plt.savefig(scenario_file, dpi=300, bbox_inches='tight')
            plt.close()

        print(f"  [TREND] Created {len(scenario_data)} scenario comparison plots")


def main():
    """Run comprehensive evaluation."""
    evaluator = ComprehensiveEvaluator()
    results = evaluator.run_full_evaluation()

    if results:
        print(f"\n[OK] Successfully evaluated {len(results)} models")
        print("[CHART] Check results/evaluations/ for detailed reports and visualizations")
    else:
        print("\n[ERROR] Evaluation failed - check for trained models")


if __name__ == "__main__":
    main()
