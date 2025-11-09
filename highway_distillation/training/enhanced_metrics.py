#!/usr/bin/env python3
"""
Enhanced Metrics Collection for Professional RL Training

This module provides comprehensive metrics collection for production RL training,
including statistical analysis, convergence diagnostics, and advanced RL metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import time
import psutil
from pathlib import Path
import json
import scipy.stats as stats


class RLTrainingMetrics:
    """
    Comprehensive metrics collection for RL training.

    Collects traditional metrics plus advanced RL-specific measurements
    with statistical analysis capabilities.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

        # Core RL metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []

        # Advanced RL metrics
        self.value_function_estimates = []
        self.policy_entropies = []
        self.q_value_distributions = []
        self.gradient_norms = []
        self.action_distributions = []
        self.value_losses = []
        self.policy_losses = []
        self.kl_divergences = []
        self.exploration_rates = []

        # Statistical analysis
        self.reward_bootstrap_cis = []
        self.convergence_scores = []
        self.stationarity_tests = []

        # Infrastructure metrics
        self.training_times = []
        self.memory_usage = []
        self.cpu_usage = []
        self.gpu_memory = []

        # Metadata
        self.start_time = time.time()
        self.episode_count = 0

    def update_episode(self, reward: float, length: int, success: bool = False,
                      phase: str = None, **kwargs):
        """Update metrics for a completed episode."""
        self.episode_count += 1

        # Core metrics
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_successes.append(success)

        # Advanced metrics (if provided)
        if 'value_estimate' in kwargs:
            self.value_function_estimates.append(kwargs['value_estimate'])
        if 'policy_entropy' in kwargs:
            self.policy_entropies.append(kwargs['policy_entropy'])
        if 'q_values' in kwargs:
            self.q_value_distributions.append(kwargs['q_values'])
        if 'gradient_norm' in kwargs:
            self.gradient_norms.append(kwargs['gradient_norm'])
        if 'action_dist' in kwargs:
            self.action_distributions.append(kwargs['action_dist'])
        if 'value_loss' in kwargs:
            self.value_losses.append(kwargs['value_loss'])
        if 'policy_loss' in kwargs:
            self.policy_losses.append(kwargs['policy_loss'])
        if 'kl_divergence' in kwargs:
            self.kl_divergences.append(kwargs['kl_divergence'])
        if 'exploration_rate' in kwargs:
            self.exploration_rates.append(kwargs['exploration_rate'])

        # Infrastructure metrics
        self._collect_infrastructure_metrics()

        # Statistical analysis (every 50 episodes)
        if self.episode_count % 50 == 0:
            self._update_statistical_analysis()

    def _collect_infrastructure_metrics(self):
        """Collect system resource usage."""
        try:
            self.training_times.append(time.time() - self.start_time)
            self.memory_usage.append(psutil.virtual_memory().percent)
            self.cpu_usage.append(psutil.cpu_percent(interval=None))

            # GPU metrics (if available)
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                    self.gpu_memory.append(gpu_mem)
                else:
                    self.gpu_memory.append(0.0)
            except ImportError:
                self.gpu_memory.append(0.0)

        except Exception as e:
            # Graceful degradation if system monitoring fails
            self.memory_usage.append(0.0)
            self.cpu_usage.append(0.0)
            self.gpu_memory.append(0.0)

    def _update_statistical_analysis(self):
        """Update statistical analysis metrics."""
        if len(self.episode_rewards) < 30:
            return

        # Bootstrap confidence intervals
        recent_rewards = np.array(self.episode_rewards[-self.window_size:])
        ci_lower, ci_upper = self._bootstrap_confidence_interval(recent_rewards, n_bootstrap=1000)
        self.reward_bootstrap_cis.append((ci_lower, ci_upper))

        # Convergence assessment
        convergence_score = self._assess_convergence(recent_rewards)
        self.convergence_scores.append(convergence_score)

        # Stationarity test (Dickey-Fuller)
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(recent_rewards)
            stationarity_p_value = adf_result[1]
            self.stationarity_tests.append(stationarity_p_value)
        except ImportError:
            self.stationarity_tests.append(None)

    def _bootstrap_confidence_interval(self, data: np.ndarray, n_bootstrap: int = 1000,
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        bootstrapped_means = []
        n = len(data)

        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrapped_means.append(np.mean(sample))

        alpha = 1 - confidence
        lower = np.percentile(bootstrapped_means, alpha/2 * 100)
        upper = np.percentile(bootstrapped_means, (1 - alpha/2) * 100)

        return lower, upper

    def _assess_convergence(self, rewards: np.ndarray, threshold: float = 0.05) -> float:
        """
        Assess convergence based on reward stability.

        Returns convergence score between 0 (not converged) and 1 (fully converged).
        """
        if len(rewards) < 20:
            return 0.0

        # Method 1: Coefficient of variation (lower is better)
        cv = np.std(rewards) / abs(np.mean(rewards)) if np.mean(rewards) != 0 else float('inf')
        cv_score = max(0, 1 - cv)  # Normalize to 0-1

        # Method 2: Trend analysis (should be flat for convergence)
        if len(rewards) >= 50:
            recent_trend = np.polyfit(range(len(rewards)), rewards, 1)[0]
            trend_score = max(0, 1 - abs(recent_trend) * 10)  # Scale trend sensitivity
        else:
            trend_score = 0.5  # Neutral for short sequences

        # Method 3: Autocorrelation (should be high for stable series)
        if len(rewards) >= 30:
            autocorr = np.corrcoef(rewards[:-1], rewards[1:])[0, 1]
            autocorr_score = (autocorr + 1) / 2  # Convert [-1,1] to [0,1]
        else:
            autocorr_score = 0.5

        # Combine scores
        convergence_score = (cv_score + trend_score + autocorr_score) / 3
        return min(1.0, max(0.0, convergence_score))

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics."""
        if not self.episode_rewards:
            return {}

        rewards = np.array(self.episode_rewards)

        summary = {
            # Basic statistics
            'total_episodes': len(rewards),
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'median_reward': float(np.median(rewards)),

            # Success metrics
            'success_rate': float(np.mean(self.episode_successes)),
            'total_successes': int(np.sum(self.episode_successes)),

            # Episode statistics
            'mean_episode_length': float(np.mean(self.episode_lengths)),
            'std_episode_length': float(np.std(self.episode_lengths)),

            # Advanced metrics (if available)
            'latest_convergence_score': self.convergence_scores[-1] if self.convergence_scores else None,
            'mean_policy_entropy': float(np.mean(self.policy_entropies)) if self.policy_entropies else None,
            'mean_value_loss': float(np.mean(self.value_losses)) if self.value_losses else None,
            'mean_policy_loss': float(np.mean(self.policy_losses)) if self.policy_losses else None,

            # Statistical analysis
            'reward_confidence_interval': self.reward_bootstrap_cis[-1] if self.reward_bootstrap_cis else None,
            'stationarity_p_value': self.stationarity_tests[-1] if self.stationarity_tests else None,

            # Training efficiency
            'training_time_seconds': time.time() - self.start_time,
            'episodes_per_second': len(rewards) / (time.time() - self.start_time),
            'mean_memory_usage': float(np.mean(self.memory_usage)) if self.memory_usage else None,
            'mean_cpu_usage': float(np.mean(self.cpu_usage)) if self.cpu_usage else None,
        }

        return summary

    def detect_training_issues(self) -> List[str]:
        """Detect potential training issues and return alerts."""
        alerts = []

        if len(self.episode_rewards) < 10:
            return alerts  # Not enough data

        recent_rewards = self.episode_rewards[-20:]

        # Reward collapse detection
        if len(recent_rewards) >= 10:
            recent_mean = np.mean(recent_rewards)
            overall_mean = np.mean(self.episode_rewards)
            if recent_mean < overall_mean * 0.5:
                alerts.append(".2f"
                            f"Overall: {overall_mean:.2f})")

        # High variance (unstable training)
        if np.std(recent_rewards) > abs(np.mean(recent_rewards)) * 2:
            alerts.append(".2f"
                        f"Std: {np.std(recent_rewards):.2f})")

        # Policy gradient explosion
        if self.gradient_norms and len(self.gradient_norms) >= 5:
            recent_gradients = self.gradient_norms[-5:]
            if np.mean(recent_gradients) > 10:  # Arbitrary threshold
                alerts.append(".2f")

        # Convergence issues
        if self.convergence_scores and self.convergence_scores[-1] < 0.3:
            alerts.append(".3f")

        # Resource issues
        if self.memory_usage and np.mean(self.memory_usage[-10:]) > 90:
            alerts.append(".1f")

        return alerts

    def export_metrics(self, filepath: str) -> str:
        """Export all metrics to a comprehensive JSON file."""
        metrics_data = {
            'summary': self.get_summary_stats(),
            'raw_data': {
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'episode_successes': self.episode_successes,
                'policy_entropies': self.policy_entropies,
                'value_losses': self.value_losses,
                'policy_losses': self.policy_losses,
                'gradient_norms': self.gradient_norms,
                'training_times': self.training_times,
                'memory_usage': self.memory_usage,
                'cpu_usage': self.cpu_usage,
            },
            'analysis': {
                'convergence_scores': self.convergence_scores,
                'reward_confidence_intervals': self.reward_bootstrap_cis,
                'stationarity_tests': self.stationarity_tests,
                'detected_issues': self.detect_training_issues(),
            },
            'metadata': {
                'export_time': time.time(),
                'total_episodes': self.episode_count,
                'collection_window': self.window_size,
            }
        }

        # Save to outputs/metrics directory
        metrics_dir = Path("outputs/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_filepath = metrics_dir / f"{Path(filepath).name}"

        with open(metrics_filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)

        return str(metrics_filepath)


class ExperimentTracker:
    """
    Track experiments with hyperparameters, configurations, and results.

    Provides systematic experiment management for RL research.
    """

    def __init__(self, experiments_dir: str = "outputs/experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.current_experiment = None

    def start_experiment(self, name: str, config: Dict[str, Any],
                        tags: List[str] = None) -> str:
        """Start a new experiment with configuration."""
        experiment_id = f"{name}_{int(time.time())}"
        experiment_dir = self.experiments_dir / experiment_id
        experiment_dir.mkdir(exist_ok=True)

        self.current_experiment = {
            'id': experiment_id,
            'name': name,
            'config': config,
            'tags': tags or [],
            'start_time': time.time(),
            'status': 'running',
            'directory': str(experiment_dir),
        }

        # Save experiment configuration
        with open(experiment_dir / 'config.json', 'w') as f:
            json.dump(self.current_experiment, f, indent=2, default=str)

        return experiment_id

    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters for the current experiment."""
        if not self.current_experiment:
            raise ValueError("No active experiment")

        exp_dir = Path(self.current_experiment['directory'])
        with open(exp_dir / 'hyperparameters.json', 'w') as f:
            json.dump(params, f, indent=2)

    def save_checkpoint(self, model_path: str, metrics: Dict[str, Any],
                       episode: int) -> str:
        """Save a model checkpoint with metadata."""
        if not self.current_experiment:
            raise ValueError("No active experiment")

        exp_dir = Path(self.current_experiment['directory'])
        checkpoints_dir = exp_dir / 'checkpoints'
        checkpoints_dir.mkdir(exist_ok=True)

        checkpoint_info = {
            'episode': episode,
            'model_path': model_path,
            'metrics': metrics,
            'timestamp': time.time(),
        }

        checkpoint_file = checkpoints_dir / f'checkpoint_ep{episode}.json'
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_info, f, indent=2, default=str)

        return str(checkpoint_file)

    def end_experiment(self, final_metrics: Dict[str, Any], status: str = 'completed'):
        """End the current experiment."""
        if not self.current_experiment:
            raise ValueError("No active experiment")

        self.current_experiment.update({
            'end_time': time.time(),
            'duration': time.time() - self.current_experiment['start_time'],
            'final_metrics': final_metrics,
            'status': status,
        })

        exp_dir = Path(self.current_experiment['directory'])
        with open(exp_dir / 'experiment_summary.json', 'w') as f:
            json.dump(self.current_experiment, f, indent=2, default=str)

        # Update experiment registry
        self._update_experiment_registry()

        experiment_id = self.current_experiment['id']
        self.current_experiment = None
        return experiment_id

    def _update_experiment_registry(self):
        """Update the global experiment registry."""
        registry_file = self.experiments_dir / 'experiment_registry.json'

        # Load existing registry
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                registry = json.load(f)
        else:
            registry = {'experiments': []}

        # Add current experiment
        registry['experiments'].append({
            'id': self.current_experiment['id'],
            'name': self.current_experiment['name'],
            'status': self.current_experiment.get('status', 'unknown'),
            'start_time': self.current_experiment['start_time'],
            'end_time': self.current_experiment.get('end_time'),
            'tags': self.current_experiment['tags'],
            'directory': self.current_experiment['directory'],
        })

        # Save updated registry
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2, default=str)


# Example usage integration with existing training
def create_enhanced_training_monitor(phase_name: str) -> Tuple[RLTrainingMetrics, ExperimentTracker]:
    """Create enhanced monitoring for RL training."""

    # Initialize comprehensive metrics collection
    metrics = RLTrainingMetrics(window_size=100)

    # Initialize experiment tracking
    tracker = ExperimentTracker()

    # Example experiment configuration
    experiment_config = {
        'phase': phase_name,
        'algorithm': 'PPO',
        'environment': 'UrbanJunctionEnv',
        'multi_modal': True,
        'antagonistic_vehicles': True,
    }

    # Start experiment tracking
    experiment_id = tracker.start_experiment(
        name=f"{phase_name}_training",
        config=experiment_config,
        tags=['rl', 'autonomous-driving', phase_name]
    )

    return metrics, tracker


# Demonstration of enhanced metrics usage
if __name__ == "__main__":
    print("Enhanced RL Metrics Collection Demo")
    print("=" * 50)

    # Create enhanced monitoring
    metrics, tracker = create_enhanced_training_monitor("phase1")

    # Simulate training episodes with rich metrics
    np.random.seed(42)
    for episode in range(1, 151):
        # Simulate realistic reward progression
        base_reward = 2.0 + episode * 0.02 + np.random.normal(0, 0.5)
        reward = max(-5.0, min(15.0, base_reward))  # Clip to realistic range

        length = int(np.random.normal(25, 5))
        success = reward > 8.0

        # Add advanced RL metrics
        metrics.update_episode(
            reward=reward,
            length=length,
            success=success,
            phase='highway',
            value_estimate=np.random.normal(5.0, 1.0),
            policy_entropy=np.random.beta(2, 5),  # Low entropy = deterministic policy
            q_values=np.random.normal(0, 2, 5),  # 5 actions
            gradient_norm=np.random.exponential(0.1),
            value_loss=np.random.exponential(0.5),
            policy_loss=np.random.exponential(0.3),
            exploration_rate=max(0.01, 0.5 * np.exp(-episode / 100)),
        )

        # Periodic checkpointing
        if episode % 50 == 0:
            checkpoint_metrics = metrics.get_summary_stats()
            tracker.save_checkpoint(
                model_path=f"model_ep{episode}.zip",
                metrics=checkpoint_metrics,
                episode=episode
            )

        # Progress reporting
        if episode % 25 == 0:
            summary = metrics.get_summary_stats()
            alerts = metrics.detect_training_issues()
            conv_score = summary.get('latest_convergence_score') or 0.0
            print(f"Episode {episode}: Reward={summary['mean_reward']:.2f} "
                  f"Success={summary['success_rate']:.1%} "
                  f"Convergence={conv_score:.2f}")
            if alerts:
                print(f"  Alerts: {alerts}")

    # Final analysis
    final_metrics = metrics.get_summary_stats()
    alerts = metrics.detect_training_issues()

    print("\nFinal Training Summary:")
    print(f"  Episodes: {final_metrics['total_episodes']}")
    print(".2f")
    print(".2f")
    print(".1%")
    print(".2f")
    if final_metrics.get('latest_convergence_score'):
        print(".3f")
    if alerts:
        print(f"  Training Issues: {alerts}")

    # Export comprehensive data
    metrics_file = metrics.export_metrics("enhanced_training_metrics.json")
    tracker.end_experiment(final_metrics, status='completed')

    print(f"\nData exported to: {metrics_file}")
    print(f"Experiment tracking complete: {tracker.experiments_dir}")

    print("\nThis demonstrates professional-grade RL monitoring with:")
    print("- Comprehensive metrics collection")
    print("- Statistical analysis and confidence intervals")
    print("- Automated issue detection")
    print("- Experiment tracking and versioning")
    print("- Infrastructure monitoring")
    print("- Convergence assessment")
