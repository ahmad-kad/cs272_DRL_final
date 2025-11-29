from stable_baselines3.common.callbacks import BaseCallback
import wandb
import numpy as np
from collections import deque
import os
from typing import Dict, Any, List

class WandbMetricsCallback(BaseCallback):
    """
    Logs custom driving metrics to WandB:
    - Crash Rate
    - Average Speed
    - Success Rate (reached goal/survived)
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_crashes = []
        self.episode_speeds = []
        self.episode_successes = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        
        # Check for crashes and successes
        for idx, info in enumerate(infos):
            if "speed" in info:
                self.episode_speeds.append(info["speed"])
            
            if dones[idx]:
                crashed = info.get("crashed", False)
                self.episode_crashes.append(1 if crashed else 0)
                
                # Definition of success varies by env, but generally not crashing + positive reward
                # Or specific 'arrived_reward' in intersection
                reward = self.locals["rewards"][idx]
                success = (not crashed) and (reward > 0)
                self.episode_successes.append(1 if success else 0)

        return True

    def _on_rollout_end(self) -> None:
        # Compute metrics for the rollout
        metrics = {}
        
        if self.episode_speeds:
            metrics["rollout/mean_speed"] = np.mean(self.episode_speeds)
            
        if self.episode_crashes:
            metrics["rollout/crash_rate"] = np.mean(self.episode_crashes)
            
        if self.episode_successes:
            metrics["rollout/success_rate"] = np.mean(self.episode_successes)
            
        if metrics and wandb.run is not None:
            wandb.log(metrics, step=self.num_timesteps)
            
        # Reset buffers
        self.episode_speeds = []
        self.episode_crashes = []
        self.episode_successes = []

class AdaptiveCurriculumCallback(BaseCallback):
    """
    Stops training if success rate exceeds threshold for a sustained period.
    """
    def __init__(self, success_threshold=0.9, window_size=100, min_steps=5000, verbose=0):
        super().__init__(verbose)
        self.success_threshold = success_threshold
        self.success_buffer = deque(maxlen=window_size)
        self.min_steps = min_steps
        self.goal_reached = False

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        
        for idx, done in enumerate(dones):
            if done:
                info = infos[idx]
                crashed = info.get("crashed", False)
                reward = self.locals["rewards"][idx]
                # Simple success heuristic
                success = 1.0 if (not crashed and reward > 0) else 0.0
                self.success_buffer.append(success)

        # Check condition
        if self.num_timesteps > self.min_steps and len(self.success_buffer) >= self.success_buffer.maxlen:
            avg_success = np.mean(self.success_buffer)
            if avg_success >= self.success_threshold:
                if self.verbose > 0:
                    print(f"\nAdaptive Curriculum: Milestone reached! Success rate {avg_success:.2f} >= {self.success_threshold}")
                self.goal_reached = True
                return False # Stop training

        return True


class BestModelCallback(BaseCallback):
    """
    Callback to save the best model based on evaluation performance.
    Evaluates the model periodically and saves when a new best is achieved.
    """

    def __init__(self, eval_env, eval_freq=5000, save_path="./best_model", verbose=0):
        """
        Args:
            eval_env: Environment to use for evaluation
            eval_freq: How often to evaluate (in timesteps)
            save_path: Directory to save the best model
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.best_score = -np.inf
        self.best_timestep = 0

        os.makedirs(save_path, exist_ok=True)

    def _evaluate_model(self) -> float:
        """Evaluate the model and return average reward over multiple episodes."""
        episode_rewards = []
        n_eval_episodes = 10

        for _ in range(n_eval_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0.0
            terminated = False
            truncated = False
            steps = 0

            while not (terminated or truncated) and steps < 1000:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                episode_reward += reward
                steps += 1

                # Check for crash
                if hasattr(self.eval_env, 'vehicle') and self.eval_env.vehicle.crashed:
                    break

            episode_rewards.append(episode_reward)

        return np.mean(episode_rewards)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current model
            current_score = self._evaluate_model()

            if self.verbose > 0:
                print(f"Timestep {self.num_timesteps}: Evaluation score = {current_score:.2f}")

            # Check if this is the best score so far
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_timestep = self.num_timesteps

                # Save the model
                model_path = os.path.join(self.save_path, "best_model.zip")
                self.model.save(model_path)

                if self.verbose > 0:
                    print(f"New best model saved! Score: {current_score:.2f} at timestep {self.num_timesteps}")

        return True


class StratifiedMetricsCallback(BaseCallback):
    """
    Comprehensive callback that tracks metrics stratified by scenario, modality,
    and includes safety override tracking, reward decomposition, and performance analysis.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

        # Scenario-stratified metrics
        self.scenarios = ["highway", "merge", "intersection"]
        self.modalities = ["lidar", "grayscale", "both"]

        # Initialize metric buffers for each scenario
        self.scenario_metrics = {}
        for scenario in self.scenarios:
            self.scenario_metrics[scenario] = {
                "episode_rewards": [],
                "episode_lengths": [],
                "crashes": [],
                "successes": [],
                "completions": [],
                "safety_overrides": [],
                "emergency_brakes": [],
                "speed_limits": [],
                "road_recoveries": [],
                "lane_corrections": [],
                "speeds": [],
                "reward_components": {}
            }

        # Overall metrics
        self.overall_metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "crashes": [],
            "successes": [],
            "completions": [],
            "safety_overrides": [],
            "emergency_brakes": [],
            "speed_limits": [],
            "road_recoveries": [],
            "lane_corrections": [],
            "speeds": []
        }

        # Reward component tracking (initialize with common components)
        reward_components = [
            "speed_reward", "lane_position_reward", "progress_reward",
            "completion_bonus", "safe_maneuver_bonus", "collision_penalty",
            "proximity_penalty", "on_road_reward", "offroad_penalty",
            "lane_change_penalty", "time_penalty"
        ]

        for scenario in self.scenarios:
            for component in reward_components:
                self.scenario_metrics[scenario]["reward_components"][component] = []

        # Modality tracking
        self.modality_metrics = {}
        for modality in self.modalities:
            self.modality_metrics[modality] = {
                "episode_rewards": [],
                "successes": [],
                "crashes": []
            }

        # Rolling window for recent performance
        self.window_size = 100
        self.recent_rewards = deque(maxlen=self.window_size)
        self.recent_successes = deque(maxlen=self.window_size)
        self.recent_crashes = deque(maxlen=self.window_size)

    def _on_step(self) -> bool:
        """Collect metrics from each step."""
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for idx, info in enumerate(infos):
            if dones[idx]:  # Episode ended
                reward = self.locals.get("rewards", [0])[idx] if idx < len(self.locals.get("rewards", [])) else 0
                self._record_episode_metrics(info, reward)

        return True

    def _record_episode_metrics(self, info: Dict[str, Any], reward: float = None):
        """Record comprehensive metrics for a completed episode."""
        scenario = info.get("scenario", "unknown")
        modality = info.get("modality", "unknown")
        crashed = info.get("crashed", False)

        # Use provided reward or extract from info
        if reward is None:
            reward = info.get("rewards/total_reward", 0)

        # Calculate success and completion
        success = (not crashed) and (reward > 0)
        completed = info.get("episode/completed", False)

        # Safety override metrics
        safety_overrides = info.get("safety/safety_override_count", 0)
        override_rate = info.get("safety/override_rate", 0)
        emergency_brakes = 1 if info.get("safety/last_safety_override") == "emergency_brake" else 0
        speed_limits = 1 if info.get("safety/last_safety_override") == "speed_limit" else 0
        road_recoveries = 1 if info.get("safety/last_safety_override") == "road_recovery" else 0
        lane_corrections = 1 if info.get("safety/last_safety_override") == "lane_correction" else 0

        # Vehicle metrics
        speed = info.get("vehicle/speed", 0)
        episode_length = info.get("episode_step", 0)

        # Record overall metrics
        self.overall_metrics["episode_rewards"].append(reward)
        self.overall_metrics["episode_lengths"].append(episode_length)
        self.overall_metrics["crashes"].append(1 if crashed else 0)
        self.overall_metrics["successes"].append(1 if success else 0)
        self.overall_metrics["completions"].append(1 if completed else 0)
        self.overall_metrics["safety_overrides"].append(safety_overrides)
        self.overall_metrics["emergency_brakes"].append(emergency_brakes)
        self.overall_metrics["speed_limits"].append(speed_limits)
        self.overall_metrics["road_recoveries"].append(road_recoveries)
        self.overall_metrics["lane_corrections"].append(lane_corrections)
        self.overall_metrics["speeds"].append(speed)

        # Record scenario-specific metrics
        if scenario in self.scenario_metrics:
            metrics = self.scenario_metrics[scenario]
            metrics["episode_rewards"].append(reward)
            metrics["episode_lengths"].append(episode_length)
            metrics["crashes"].append(1 if crashed else 0)
            metrics["successes"].append(1 if success else 0)
            metrics["completions"].append(1 if completed else 0)
            metrics["safety_overrides"].append(safety_overrides)
            metrics["emergency_brakes"].append(emergency_brakes)
            metrics["speed_limits"].append(speed_limits)
            metrics["road_recoveries"].append(road_recoveries)
            metrics["lane_corrections"].append(lane_corrections)
            metrics["speeds"].append(speed)

            # Record reward components
            for component_key, component_value in info.items():
                if component_key.startswith("rewards/") and component_key.endswith("_raw"):
                    component_name = component_key.replace("rewards/", "").replace("_raw", "")
                    if component_name in metrics["reward_components"]:
                        metrics["reward_components"][component_name].append(component_value)

        # Record modality-specific metrics
        if modality in self.modality_metrics:
            mod_metrics = self.modality_metrics[modality]
            mod_metrics["episode_rewards"].append(reward)
            mod_metrics["successes"].append(1 if success else 0)
            mod_metrics["crashes"].append(1 if crashed else 0)

        # Update rolling window
        self.recent_rewards.append(reward)
        self.recent_successes.append(1 if success else 0)
        self.recent_crashes.append(1 if crashed else 0)

    def _on_rollout_end(self) -> None:
        """Compute and log aggregated metrics."""
        if not self.overall_metrics["episode_rewards"]:
            return  # No episodes completed yet

        metrics = {}

        # Overall performance metrics
        if self.overall_metrics["episode_rewards"]:
            metrics["rollout/ep_rew_mean"] = np.mean(self.overall_metrics["episode_rewards"][-10:])  # Last 10 episodes
            metrics["rollout/ep_len_mean"] = np.mean(self.overall_metrics["episode_lengths"][-10:])

        if self.overall_metrics["successes"]:
            metrics["rollout/success_rate"] = np.mean(self.overall_metrics["successes"][-20:])  # Last 20 episodes

        if self.overall_metrics["crashes"]:
            metrics["rollout/crash_rate"] = np.mean(self.overall_metrics["crashes"][-20:])

        if self.overall_metrics["completions"]:
            metrics["rollout/completion_rate"] = np.mean(self.overall_metrics["completions"][-20:])

        # Safety metrics
        if self.overall_metrics["safety_overrides"]:
            metrics["safety/total_override_rate"] = np.mean(self.overall_metrics["safety_overrides"][-20:])
            metrics["safety/emergency_brake_rate"] = np.mean(self.overall_metrics["emergency_brakes"][-20:])
            metrics["safety/speed_limit_rate"] = np.mean(self.overall_metrics["speed_limits"][-20:])
            metrics["safety/road_recovery_rate"] = np.mean(self.overall_metrics["road_recoveries"][-20:])
            metrics["safety/lane_correction_rate"] = np.mean(self.overall_metrics["lane_corrections"][-20:])

        # Vehicle metrics
        if self.overall_metrics["speeds"]:
            metrics["vehicle/avg_speed"] = np.mean(self.overall_metrics["speeds"][-20:])

        # Scenario-specific metrics
        for scenario in self.scenarios:
            scenario_data = self.scenario_metrics[scenario]
            if scenario_data["episode_rewards"]:
                metrics[f"{scenario}/success_rate"] = np.mean(scenario_data["successes"][-10:])
                metrics[f"{scenario}/crash_rate"] = np.mean(scenario_data["crashes"][-10:])
                metrics[f"{scenario}/avg_reward"] = np.mean(scenario_data["episode_rewards"][-10:])
                metrics[f"{scenario}/safety_override_rate"] = np.mean(scenario_data["safety_overrides"][-10:])

        # Modality-specific metrics
        for modality in self.modalities:
            mod_data = self.modality_metrics[modality]
            if mod_data["episode_rewards"]:
                metrics[f"{modality}/success_rate"] = np.mean(mod_data["successes"][-10:])
                metrics[f"{modality}/crash_rate"] = np.mean(mod_data["crashes"][-10:])
                metrics[f"{modality}/avg_reward"] = np.mean(mod_data["episode_rewards"][-10:])

        # Rolling window performance (recent 100 episodes)
        if len(self.recent_rewards) >= 10:
            metrics["recent/avg_reward"] = np.mean(list(self.recent_rewards))
            metrics["recent/success_rate"] = np.mean(list(self.recent_successes))
            metrics["recent/crash_rate"] = np.mean(list(self.recent_crashes))

        # Log to wandb if available
        if wandb.run is not None:
            wandb.log(metrics, step=self.num_timesteps)

        # Console output for important metrics
        if self.verbose > 0 and len(self.overall_metrics["episode_rewards"]) % 20 == 0:
            print("📊 Metrics Update:")
            print(f"  Recent Reward: {metrics.get('rollout/ep_rew_mean', 0):.2f}")
            print(f"  Success Rate: {metrics.get('rollout/success_rate', 0):.1%}")
            if "safety/total_override_rate" in metrics:
                print(f"  Safety Override Rate: {metrics['safety/total_override_rate']:.1%}")

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for analysis."""
        stats = {
            "overall": self._compute_stats_for_data(self.overall_metrics),
            "scenarios": {},
            "modalities": {},
            "recent_performance": {}
        }

        # Scenario stats
        for scenario in self.scenarios:
            stats["scenarios"][scenario] = self._compute_stats_for_data(self.scenario_metrics[scenario])

        # Modality stats
        for modality in self.modalities:
            stats["modalities"][modality] = self._compute_stats_for_data(self.modality_metrics[modality])

        # Recent performance
        if len(self.recent_rewards) > 0:
            stats["recent_performance"] = {
                "avg_reward": np.mean(list(self.recent_rewards)),
                "success_rate": np.mean(list(self.recent_successes)),
                "crash_rate": np.mean(list(self.recent_crashes)),
                "window_size": len(self.recent_rewards)
            }

        return stats

    def _compute_stats_for_data(self, data: Dict[str, List]) -> Dict[str, float]:
        """Compute statistics for a data dictionary."""
        stats = {}
        for key, values in data.items():
            if isinstance(values, list) and values:
                if key == "reward_components":
                    # Handle nested reward components
                    for comp_name, comp_values in values.items():
                        if comp_values:
                            stats[f"{comp_name}_mean"] = np.mean(comp_values)
                            stats[f"{comp_name}_std"] = np.std(comp_values)
                else:
                    stats[f"{key}_mean"] = np.mean(values)
                    stats[f"{key}_std"] = np.std(values)
                    stats[f"{key}_count"] = len(values)

                    if key in ["crashes", "successes", "completions", "emergency_brakes", "speed_limits", "road_recoveries", "lane_corrections"]:
                        stats[f"{key}_rate"] = np.mean(values)

        return stats


class CurriculumTrackingCallback(BaseCallback):
    """
    Callback to track curriculum learning progress, generalization across scenarios,
    and learning transfer between modalities.
    """

    def __init__(self, curriculum_phases=None, verbose=0):
        super().__init__(verbose)
        self.curriculum_phases = curriculum_phases or []
        self.current_phase_idx = 0
        self.phase_start_timestep = 0

        # Phase performance tracking
        self.phase_metrics = {
            "start_timestep": [],
            "end_timestep": [],
            "duration": [],
            "final_success_rate": [],
            "final_crash_rate": [],
            "final_avg_reward": [],
            "peak_success_rate": [],
            "peak_avg_reward": []
        }

        # Generalization tracking
        self.scenario_performance = {
            "highway": {"rewards": [], "successes": [], "crashes": []},
            "merge": {"rewards": [], "successes": [], "crashes": []},
            "intersection": {"rewards": [], "successes": [], "crashes": []}
        }

        self.modality_performance = {
            "lidar": {"rewards": [], "successes": [], "crashes": []},
            "grayscale": {"rewards": [], "successes": [], "crashes": []},
            "both": {"rewards": [], "successes": [], "crashes": []}
        }

        # Learning transfer metrics
        self.transfer_metrics = {
            "scenario_generalization": [],
            "modality_transfer": [],
            "curriculum_efficiency": []
        }

    def _on_step(self) -> bool:
        """Track curriculum progress and generalization."""
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for idx, info in enumerate(infos):
            if dones[idx]:  # Episode completed
                self._record_episode_for_curriculum(info, self.locals.get("rewards", [0])[idx] if idx < len(self.locals.get("rewards", [])) else 0)

        return True

    def _record_episode_for_curriculum(self, info: Dict[str, Any], reward: float):
        """Record episode data for curriculum analysis."""
        scenario = info.get("scenario", "unknown")
        modality = info.get("modality", "unknown")
        crashed = info.get("crashed", False)
        success = (not crashed) and (reward > 0)

        # Record scenario performance
        if scenario in self.scenario_performance:
            self.scenario_performance[scenario]["rewards"].append(reward)
            self.scenario_performance[scenario]["successes"].append(1 if success else 0)
            self.scenario_performance[scenario]["crashes"].append(1 if crashed else 0)

        # Record modality performance
        if modality in self.modality_performance:
            self.modality_performance[modality]["rewards"].append(reward)
            self.modality_performance[modality]["successes"].append(1 if success else 0)
            self.modality_performance[modality]["crashes"].append(1 if crashed else 0)

    def on_phase_start(self, phase_idx: int, phase_config: Dict[str, Any]):
        """Called when a new curriculum phase starts."""
        self.current_phase_idx = phase_idx
        self.phase_start_timestep = self.num_timesteps

        if self.verbose > 0:
            print(f"📚 Curriculum Phase {phase_idx + 1} Started: {phase_config.get('name', 'Unknown')}")
            print(f"   Scenarios: {phase_config.get('scenarios', [])}")
            print(f"   Modality: {phase_config.get('modalities', [])}")

    def on_phase_end(self, phase_idx: int, phase_config: Dict[str, Any], phase_metrics: Dict[str, float]):
        """Called when a curriculum phase ends."""
        phase_duration = self.num_timesteps - self.phase_start_timestep

        # Record phase metrics
        self.phase_metrics["start_timestep"].append(self.phase_start_timestep)
        self.phase_metrics["end_timestep"].append(self.num_timesteps)
        self.phase_metrics["duration"].append(phase_duration)
        self.phase_metrics["final_success_rate"].append(phase_metrics.get("success_rate", 0))
        self.phase_metrics["final_crash_rate"].append(phase_metrics.get("crash_rate", 0))
        self.phase_metrics["final_avg_reward"].append(phase_metrics.get("avg_reward", 0))

        # Calculate peak performance during phase
        self._calculate_phase_peaks(phase_idx)

        # Calculate generalization metrics
        self._update_generalization_metrics()

        if self.verbose > 0:
            print(f"📚 Curriculum Phase {phase_idx + 1} Completed:")
            print(f"   Success Rate: {phase_metrics.get('success_rate', 0):.1%}")
            print(f"   Crash Rate: {phase_metrics.get('crash_rate', 0):.1%}")
            print(f"   Avg Reward: {phase_metrics.get('avg_reward', 0):.2f}")
    def _calculate_phase_peaks(self, phase_idx: int):
        """Calculate peak performance metrics for the completed phase."""
        # For now, use final metrics as peak (could be enhanced with rolling windows)
        self.phase_metrics["peak_success_rate"].append(self.phase_metrics["final_success_rate"][-1])
        self.phase_metrics["peak_avg_reward"].append(self.phase_metrics["final_avg_reward"][-1])

    def _update_generalization_metrics(self):
        """Calculate generalization and transfer learning metrics."""
        # Scenario generalization: variance in performance across scenarios
        scenario_success_rates = []
        for scenario, data in self.scenario_performance.items():
            if data["successes"]:
                success_rate = np.mean(data["successes"][-50:])  # Last 50 episodes per scenario
                scenario_success_rates.append(success_rate)

        if len(scenario_success_rates) > 1:
            scenario_generalization = 1.0 - np.std(scenario_success_rates)  # Lower variance = better generalization
            self.transfer_metrics["scenario_generalization"].append(scenario_generalization)

        # Modality transfer: performance improvement across modalities
        modality_success_rates = []
        for modality, data in self.modality_performance.items():
            if data["successes"]:
                success_rate = np.mean(data["successes"][-50:])
                modality_success_rates.append(success_rate)

        if len(modality_success_rates) > 1:
            modality_transfer = np.mean(modality_success_rates)  # Average performance across modalities
            self.transfer_metrics["modality_transfer"].append(modality_transfer)

        # Curriculum efficiency: success rate improvement over phases
        if len(self.phase_metrics["final_success_rate"]) > 1:
            recent_phases = self.phase_metrics["final_success_rate"][-3:]  # Last 3 phases
            if len(recent_phases) > 1:
                efficiency = np.mean(recent_phases) / max(0.01, recent_phases[0])  # Improvement ratio
                self.transfer_metrics["curriculum_efficiency"].append(efficiency)

    def _on_rollout_end(self) -> None:
        """Log curriculum and generalization metrics."""
        metrics = {}

        # Current generalization metrics
        if self.transfer_metrics["scenario_generalization"]:
            metrics["curriculum/scenario_generalization"] = self.transfer_metrics["scenario_generalization"][-1]

        if self.transfer_metrics["modality_transfer"]:
            metrics["curriculum/modality_transfer"] = self.transfer_metrics["modality_transfer"][-1]

        if self.transfer_metrics["curriculum_efficiency"]:
            metrics["curriculum/efficiency"] = self.transfer_metrics["curriculum_efficiency"][-1]

        # Current phase performance
        if self.phase_metrics["final_success_rate"]:
            metrics["curriculum/current_phase_success"] = self.phase_metrics["final_success_rate"][-1]
            metrics["curriculum/current_phase_crash_rate"] = self.phase_metrics["final_crash_rate"][-1]
            metrics["curriculum/current_phase_reward"] = self.phase_metrics["final_avg_reward"][-1]

        # Scenario diversity metrics
        for scenario, data in self.scenario_performance.items():
            if data["successes"]:
                metrics[f"curriculum/{scenario}_success_rate"] = np.mean(data["successes"][-20:])

        # Modality diversity metrics
        for modality, data in self.modality_performance.items():
            if data["successes"]:
                metrics[f"curriculum/{modality}_success_rate"] = np.mean(data["successes"][-20:])

        # Log to wandb
        if wandb.run is not None:
            wandb.log(metrics, step=self.num_timesteps)

    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get comprehensive curriculum statistics."""
        stats = {
            "phase_performance": dict(self.phase_metrics),
            "scenario_generalization": self.transfer_metrics["scenario_generalization"][-1] if self.transfer_metrics["scenario_generalization"] else 0,
            "modality_transfer": self.transfer_metrics["modality_transfer"][-1] if self.transfer_metrics["modality_transfer"] else 0,
            "curriculum_efficiency": self.transfer_metrics["curriculum_efficiency"][-1] if self.transfer_metrics["curriculum_efficiency"] else 0,
            "scenario_performance": {},
            "modality_performance": {}
        }

        # Current scenario performance
        for scenario, data in self.scenario_performance.items():
            if data["successes"]:
                stats["scenario_performance"][scenario] = {
                    "success_rate": np.mean(data["successes"]),
                    "crash_rate": np.mean(data["crashes"]),
                    "avg_reward": np.mean(data["rewards"])
                }

        # Current modality performance
        for modality, data in self.modality_performance.items():
            if data["successes"]:
                stats["modality_performance"][modality] = {
                    "success_rate": np.mean(data["successes"]),
                    "crash_rate": np.mean(data["crashes"]),
                    "avg_reward": np.mean(data["rewards"])
                }

        return stats


class ProgressCallback(BaseCallback):
    """
    Callback to print training progress to console every N steps.
    """

    def __init__(self, print_freq=1000, verbose=1):
        """
        Args:
            print_freq: How often to print progress (in timesteps)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0 and self.verbose > 0:
            print(f"Training Progress: {self.num_timesteps} timesteps completed")
        return True

