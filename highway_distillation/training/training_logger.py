#!/usr/bin/env python3
"""
Enhanced Training Logger for RL Highway Distillation

Provides comprehensive logging for training phases with:
- Real-time progress tracking
- Performance metrics visualization
- Environment state monitoring
- Curriculum progress tracking
- Resource usage monitoring
- Training milestone detection
"""

import os
import logging
import time
import psutil
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


class TrainingLogger:
    """
    Enhanced training logger with comprehensive metrics and visualization.

    Features:
    - Real-time progress tracking with milestones
    - Performance metrics (rewards, success rates, episode lengths)
    - Environment state monitoring (phases, difficulty levels)
    - Resource usage tracking (CPU, memory, training speed)
    - Curriculum progress visualization
    - Training milestone detection and logging
    """

    def __init__(self, phase_name: str, outputs_dir: str = "outputs"):
        """
        Initialize insight-focused training logger.

        Args:
            phase_name: Name of training phase (e.g., "phase1", "phase2")
            outputs_dir: Base directory for all outputs
        """
        self.phase_name = phase_name
        self.outputs_dir = Path(outputs_dir)
        self.logs_dir = self.outputs_dir / "logs" / phase_name
        self.data_dir = self.outputs_dir / "data"
        self.plots_dir = self.outputs_dir / "plots"
        self.experiments_dir = self.outputs_dir / "experiments"

        # Create all output directories
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(f"{phase_name}_training")
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # File handler for insight logs
        log_file = self.logs_dir / f"{phase_name}_training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler for insights only
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')  # Clean output
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Metrics tracking (silent)
        self.start_time = time.time()
        self.episode_metrics = []
        self.checkpoint_times = []
        self.milestones = []

        # Insight tracking
        self._logged_first_success = False
        self._last_logged_improvement = None

        # Resource monitoring
        self.initial_memory = psutil.virtual_memory().used / (1024**3)  # GB
        self.process = psutil.Process()

        # Silent initialization - no log spam

    def log_training_start(self, config: Dict[str, Any]) -> None:
        """Log essential training start information."""
        key_config = {k: v for k, v in config.items()
                     if k in ['total_timesteps', 'learning_rate', 'batch_size']}
        if key_config:
            config_str = ", ".join(f"{k}={v}" for k, v in key_config.items())
            self.logger.info(f"[START] {self.phase_name.upper()}: {config_str}")
        else:
            self.logger.info(f"[START] {self.phase_name.upper()} training initiated")

    def log_environment_setup(self, env_config: Dict[str, Any]) -> None:
        """Log environment configuration."""
        self.logger.info("Environment Setup:")

        # Log key environment parameters
        key_params = [
            'observation_type', 'vehicles_count', 'stage_mode',
            'antagonistic_vehicles', 'annoyance_level', 'duration'
        ]

        for param in key_params:
            if param in env_config:
                value = env_config[param]
                if isinstance(value, float):
                    self.logger.info(f"  {param}: {value:.3f}")
                else:
                    self.logger.info(f"  {param}: {value}")

        self.logger.info("Environment configured successfully")

    def log_model_architecture(self, model_info: Dict[str, Any]) -> None:
        """Log model architecture details."""
        self.logger.info("Model Architecture:")

        if 'policy_type' in model_info:
            self.logger.info(f"  Policy: {model_info['policy_type']}")

        if 'network_arch' in model_info:
            self.logger.info(f"  Network: {model_info['network_arch']}")

        if 'parameters' in model_info:
            params = model_info['parameters']
            if isinstance(params, (int, float)) and params > 1000:
                self.logger.info(",")
            else:
                self.logger.info(f"  Parameters: {params}")

        self.logger.info("Model initialized successfully")

    def log_episode_progress(self, episode: int, reward: float, length: int,
                           success: bool = False, phase: str = None) -> None:
        """Log only when insights are gained - minimal approach."""
        # Store metrics for analysis (silent operation)
        episode_data = {
            'episode': episode,
            'reward': reward,
            'length': length,
            'success': success,
            'phase': phase,
            'timestamp': time.time()
        }
        self.episode_metrics.append(episode_data)

        # Only log significant insights - never routine updates
        should_log = self._should_log_insight(episode, reward, success)
        if should_log:
            insight_type, message = should_log
            self.logger.info(f"[{insight_type}] {message}")

    def _should_log_insight(self, episode: int, reward: float, success: bool) -> Optional[Tuple[str, str]]:
        """Determine if this episode provides a logging-worthy insight."""

        # Always log first few episodes for baseline
        if episode <= 3:
            return "BASELINE", f"Episode {episode}: {reward:+.1f} reward"

        # Log major milestones (reduced frequency)
        if episode in [100, 500, 1000, 2500, 5000]:
            avg_reward = np.mean([ep['reward'] for ep in self.episode_metrics[-50:]])
            return "MILESTONE", f"Episode {episode}: {avg_reward:+.2f} avg reward"

        # Log significant reward improvements (only major changes)
        if len(self.episode_metrics) >= 100:  # Need substantial history
            recent_avg = np.mean([ep['reward'] for ep in self.episode_metrics[-50:]])
            older_avg = np.mean([ep['reward'] for ep in self.episode_metrics[-100:-50]])

            improvement = recent_avg - older_avg
            # Only log if improvement > 1.0 AND we haven't logged recently
            if abs(improvement) > 1.0:
                # Hysteresis: don't log if we logged a similar trend recently
                if (not hasattr(self, '_last_trend_episode') or
                    episode - self._last_trend_episode > 100):  # Min 100 episodes between trend logs
                    self._last_trend_episode = episode
                    direction = "↗" if improvement > 0 else "↘"
                    return "TREND", f"Reward {direction} {improvement:+.2f} over 50 episodes"

        # Log convergence detection (very rare but important)
        if len(self.episode_metrics) >= 200:  # Need lots of data for convergence
            convergence_score = self._calculate_convergence_insight()
            if convergence_score > 0.9:  # Very high threshold for convergence
                # Hysteresis: only log convergence once, not repeatedly
                if not hasattr(self, '_convergence_logged') or not self._convergence_logged:
                    self._convergence_logged = True
                    return "CONVERGED", f"Training converged (stability: {convergence_score:.1f})"

        # Log first success (important behavioral milestone)
        if success and not hasattr(self, '_logged_first_success'):
            self._logged_first_success = True
            return "SUCCESS", f"First successful episode at {episode}"

        # Log anomalies (very rare, very important)
        if self._detect_anomaly(reward, episode):
            return "ANOMALY", f"Unusual reward: {reward:+.1f} at episode {episode}"

        # No insight to log - silent operation
        return None

    def _calculate_convergence_insight(self) -> float:
        """Calculate if training has meaningfully converged."""
        if len(self.episode_metrics) < 50:
            return 0.0

        # Simple convergence heuristic: low variance + stable trend
        recent_rewards = np.array([ep['reward'] for ep in self.episode_metrics[-50:]])

        # Coefficient of variation (lower = more stable)
        cv = np.std(recent_rewards) / abs(np.mean(recent_rewards))
        stability_score = max(0, 1 - cv * 2)  # Scale CV to 0-1

        # Trend stability (how flat the recent trend is)
        if len(recent_rewards) >= 20:
            slope = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
            trend_stability = max(0, 1 - abs(slope) * 20)  # Scale slope to 0-1
        else:
            trend_stability = 0.5

        return (stability_score + trend_stability) / 2

    def _detect_anomaly(self, reward: float, episode: int) -> bool:
        """Detect truly anomalous reward values (very rare)."""
        if len(self.episode_metrics) < 50:  # Need substantial history
            return False

        # Compare to recent distribution (last 50 episodes)
        recent_rewards = [ep['reward'] for ep in self.episode_metrics[-50:-1]]
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)

        if std_reward == 0:
            return False

        # Flag as anomaly only if > 4 standard deviations (very rare)
        z_score = abs(reward - mean_reward) / std_reward
        return z_score > 4.0


    def log_checkpoint_saved(self, checkpoint_path: str, timestep: int) -> None:
        """Log checkpoint saving."""
        checkpoint_time = time.time()
        self.checkpoint_times.append(checkpoint_time)

        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        self.logger.info(f"   Timestep: {timestep:,}")
        self.logger.info(f"   Episodes completed: {len(self.episode_metrics)}")

        # Calculate checkpoint frequency
        if len(self.checkpoint_times) > 1:
            time_since_last = checkpoint_time - self.checkpoint_times[-2]
            self.logger.info(".2f")
    def log_curriculum_progress(self, current_annoyance: float,
                              performance_trend: str = None) -> None:
        """Log curriculum learning progress."""
        self.logger.info("Curriculum Progress:")
        self.logger.info(".3f")
        if performance_trend:
            self.logger.info(f"   Trend: {performance_trend}")

        # Track annoyance progression
        self.metrics['annoyance_levels'].append(current_annoyance)

    def log_training_progress(self, timestep: int, total_timesteps: int,
                            episodes_completed: int, recent_rewards: List[float] = None) -> None:
        """Log periodic training progress."""
        progress_pct = (timestep / total_timesteps) * 100

        self.logger.info(f"Progress: {progress_pct:5.1f}% ({timestep:,}/{total_timesteps:,} steps)")

        if episodes_completed > 0:
            self.logger.info(f"   Episodes completed: {episodes_completed}")

        if recent_rewards:
            avg_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            self.logger.info(".2f")
        # Resource usage
        memory_usage = self.process.memory_info().rss / (1024**3)  # GB
        memory_delta = memory_usage - self.initial_memory
        self.logger.info(".2f")
        # Training speed (if we have episode data)
        if len(self.episode_metrics) > 10:
            recent_episodes = self.episode_metrics[-10:]
            timespan = recent_episodes[-1]['timestamp'] - recent_episodes[0]['timestamp']
            if timespan > 0:
                episodes_per_sec = len(recent_episodes) / timespan
                self.logger.info(".3f")
        # Create periodic visualization if we have enough data
        if len(self.episode_metrics) >= 25 and len(self.episode_metrics) % 50 == 0:
            self._create_intermediate_plot()
    def log_training_complete(self, final_timestep: int, total_episodes: int,
                            best_reward: float = None) -> None:
        """Log training completion with summary."""
        training_time = time.time() - self.start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        self.logger.info("=" * 80)
        self.logger.info("TRAINING COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 80)

        self.logger.info("Final Statistics:")
        self.logger.info(f"  Total timesteps: {final_timestep:,}")
        self.logger.info(f"  Episodes completed: {total_episodes}")
        if best_reward is not None:
            self.logger.info(".2f")
        self.logger.info(".2f")
        self.logger.info(f"  Training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

        if self.episode_metrics:
            all_rewards = [ep['reward'] for ep in self.episode_metrics]
            self.logger.info(".2f")
            self.logger.info(".2f")
            success_rate = sum(1 for ep in self.episode_metrics if ep['success']) / len(self.episode_metrics)
            self.logger.info(".1%")
        self.logger.info("=" * 80)

    def log_error(self, error: Exception, context: str = "") -> None:
        """Log training errors with context."""
        self.logger.error(f"ERROR in {context}: {error}")
        import traceback
        self.logger.error(f"Stack trace:\n{traceback.format_exc()}")

    def _create_intermediate_plot(self) -> None:
        """Create intermediate training progress plot during training."""
        if not self.episode_metrics:
            return

        try:
            # Extract data
            episodes = [ep['episode'] for ep in self.episode_metrics]
            rewards = [ep['reward'] for ep in self.episode_metrics]

            # Create simple progress plot
            plt.figure(figsize=(12, 6))

            # Reward progression
            plt.subplot(1, 2, 1)
            plt.plot(episodes, rewards, 'b-', alpha=0.7, linewidth=1)
            plt.title(f'{self.phase_name.upper()} - Reward Progress')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)

            # Moving average if enough data
            if len(rewards) > 20:
                window_size = min(20, len(rewards) // 5)
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(episodes[window_size-1:], moving_avg, 'r-', linewidth=2,
                        label=f'Moving Avg ({window_size} ep)')
                plt.legend()

            # Recent reward distribution
            plt.subplot(1, 2, 2)
            if len(rewards) > 50:
                recent_rewards = rewards[-50:]
                plt.hist(recent_rewards, bins=15, alpha=0.7, color='green', edgecolor='black')
                plt.axvline(np.mean(recent_rewards), color='red', linestyle='--',
                           label=f'Mean: {np.mean(recent_rewards):.2f}')
                plt.legend()
            plt.title('Recent Reward Distribution')
            plt.xlabel('Reward')
            plt.ylabel('Frequency')

            plt.tight_layout()

            # Save intermediate plot
            intermediate_plot_path = self.plots_dir / f"{self.phase_name}_progress_ep{len(episodes)}.png"
            plt.savefig(intermediate_plot_path, dpi=100, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Intermediate plot saved: {intermediate_plot_path}")

        except Exception as e:
            self.logger.warning(f"Could not create intermediate plot: {e}")

    def create_training_plots(self) -> None:
        """Create final training progress visualization plots."""
        if not self.episode_metrics:
            return

        try:
            # Extract data
            episodes = [ep['episode'] for ep in self.episode_metrics]
            rewards = [ep['reward'] for ep in self.episode_metrics]
            lengths = [ep['length'] for ep in self.episode_metrics]
            successes = [1 if ep['success'] else 0 for ep in self.episode_metrics]

            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{self.phase_name.upper()} Training Progress', fontsize=16)

            # Reward progression
            ax1.plot(episodes, rewards, 'b-', alpha=0.7, linewidth=1)
            ax1.set_title('Episode Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.grid(True, alpha=0.3)

            # Moving average reward
            if len(rewards) > 50:
                window_size = min(50, len(rewards) // 10)
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                ax1.plot(episodes[window_size-1:], moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window_size} ep)')
                ax1.legend()

            # Episode lengths
            ax2.plot(episodes, lengths, 'g-', alpha=0.7, linewidth=1)
            ax2.set_title('Episode Lengths')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Steps')
            ax2.grid(True, alpha=0.3)

            # Success rate
            success_rate = np.cumsum(successes) / np.arange(1, len(successes) + 1)
            ax3.plot(episodes, success_rate, 'orange', linewidth=2)
            ax3.set_title('Cumulative Success Rate')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Success Rate')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1)

            # Reward distribution (recent episodes)
            if len(rewards) > 100:
                recent_rewards = rewards[-100:]
                ax4.hist(recent_rewards, bins=20, alpha=0.7, color='purple', edgecolor='black')
                ax4.axvline(np.mean(recent_rewards), color='red', linestyle='--',
                           label=f'Mean: {np.mean(recent_rewards):.2f}')
                ax4.legend()
            ax4.set_title('Recent Reward Distribution (Last 100 Episodes)')
            ax4.set_xlabel('Reward')
            ax4.set_ylabel('Frequency')

            plt.tight_layout()

            # Save plot
            plot_path = self.plots_dir / f"{self.phase_name}_training_progress.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Training plots saved to: {plot_path}")

        except Exception as e:
            self.logger.warning(f"Could not create training plots: {e}")

    def _check_milestones(self, episode: int, reward: float, success: bool) -> None:
        """Check for training milestones - integrated into insight logging."""
        # Milestones are now handled in _should_log_insight for cleaner integration
        pass

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        if not self.episode_metrics:
            return {}

        rewards = [ep['reward'] for ep in self.episode_metrics]
        lengths = [ep['length'] for ep in self.episode_metrics]
        successes = [ep['success'] for ep in self.episode_metrics]

        return {
            'total_episodes': len(self.episode_metrics),
            'avg_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'std_reward': np.std(rewards),
            'avg_length': np.mean(lengths),
            'success_rate': np.mean(successes),
            'training_time_hours': (time.time() - self.start_time) / 3600,
            'milestones_reached': len(self.milestones)
        }

    def export_convergence_data(self, filename: str = None) -> str:
        """
        Export convergence data for plotting.

        Returns:
            Path to exported CSV file
        """
        if not self.episode_metrics:
            self.logger.warning("No episode data to export")
            return None

        if filename is None:
            filename = f"{self.phase_name}_convergence_data.csv"

        filepath = self.data_dir / filename

        try:
            import csv

            # Prepare data for CSV export
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = ['episode', 'reward', 'length', 'success', 'phase',
                            'timestamp', 'time_elapsed', 'moving_avg_reward_50',
                            'moving_avg_reward_100', 'cumulative_success_rate']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                start_time = self.episode_metrics[0]['timestamp'] if self.episode_metrics else time.time()

                for i, ep in enumerate(self.episode_metrics):
                    # Calculate moving averages
                    ma_50 = np.mean([e['reward'] for e in self.episode_metrics[max(0, i-49):i+1]]) if i >= 0 else ep['reward']
                    ma_100 = np.mean([e['reward'] for e in self.episode_metrics[max(0, i-99):i+1]]) if i >= 0 else ep['reward']

                    # Calculate cumulative success rate
                    success_rate = np.mean([e['success'] for e in self.episode_metrics[:i+1]]) if i >= 0 else ep['success']

                    row = {
                        'episode': ep['episode'],
                        'reward': ep['reward'],
                        'length': ep['length'],
                        'success': 1 if ep['success'] else 0,
                        'phase': ep['phase'] or 'unknown',
                        'timestamp': ep['timestamp'],
                        'time_elapsed': ep['timestamp'] - start_time,
                        'moving_avg_reward_50': ma_50,
                        'moving_avg_reward_100': ma_100,
                        'cumulative_success_rate': success_rate
                    }
                    writer.writerow(row)

            self.logger.info(f"Convergence data exported to: {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Failed to export convergence data: {e}")
            return None


class EpisodeProgressCallback:
    """
    Callback to track episode progress and log to TrainingLogger.

    This integrates with Stable-Baselines3's callback system to track
    episode completion and report to the enhanced training logger.
    """

    def __init__(self, training_logger: TrainingLogger):
        """
        Initialize episode progress callback.

        Args:
            training_logger: TrainingLogger instance to report to
        """
        self.training_logger = training_logger
        self.episode_count = 0

    def __call__(self, locals, globals):
        """Callback called during training."""
        # Check if an episode just ended
        if locals.get('done', False) or locals.get('terminated', False):
            self.episode_count += 1

            # Extract episode information
            reward = locals.get('episode_reward', 0)
            length = locals.get('episode_length', 0)

            # Try to determine if episode was successful
            # This is a heuristic - you might want to customize based on your success criteria
            success = reward > 5.0  # Example threshold

            # Try to get current phase from environment
            phase = "unknown"
            try:
                env = locals.get('self').env
                if hasattr(env, 'phase'):
                    phase = env.phase
                elif hasattr(env, 'env') and hasattr(env.env, 'phase'):
                    phase = env.env.phase
            except:
                pass

            # Log episode progress
            self.training_logger.log_episode_progress(
                episode=self.episode_count,
                reward=float(reward),
                length=int(length),
                success=success,
                phase=phase
            )


def create_episode_callback(phase_name: str) -> EpisodeProgressCallback:
    """
    Create episode progress callback for a training phase.

    Args:
        phase_name: Name of the training phase

    Returns:
        Configured EpisodeProgressCallback
    """
    if phase_name == "phase1":
        logger = create_phase1_logger()
    elif phase_name == "phase2":
        logger = create_phase2_logger()
    elif phase_name == "phase3":
        logger = create_phase3_logger()
    else:
        raise ValueError(f"Unknown phase: {phase_name}")

    return EpisodeProgressCallback(logger)


# Convenience functions for each phase
def create_phase1_logger():
    """Create insight-focused logger for Phase 1 training."""
    return TrainingLogger("phase1")

def create_phase2_logger():
    """Create insight-focused logger for Phase 2 training."""
    return TrainingLogger("phase2")

def create_phase3_logger():
    """Create insight-focused logger for Phase 3 training."""
    return TrainingLogger("phase3")
