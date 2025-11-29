import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
import wandb
import os
import numpy as np
import torch
from collections import deque
from typing import Dict, Any, Optional

from utils.config import get_curriculum_config
from utils.callbacks import WandbMetricsCallback

from environments.urban_junction_env import UrbanJunctionEnv

# from models.attention import AttentiveLidarExtractor  # Module not available

class AdaptiveCurriculum:
    """
    Manages difficulty scaling based on agent performance for maximum generalization.
    Includes modality progression for multi-modal learning.
    """

    def __init__(self, min_difficulty=0.0, max_difficulty=1.0, enable_modality_curriculum=True):
        """
        Initialize the difficulty manager.

        Args:
            min_difficulty: Minimum difficulty level (0.0 = easiest)
            max_difficulty: Maximum difficulty level (1.0 = hardest)
            enable_modality_curriculum: Whether to enable progressive modality learning
        """
        self.difficulty_level = min_difficulty
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.enable_modality_curriculum = enable_modality_curriculum

        # Performance tracking (sliding window)
        self.performance_window = deque(maxlen=50)  # Last 50 evaluations
        self.evaluation_interval = 5000  # Steps between evaluations

        # Aggressive thresholds for reaching maximum difficulty
        self.excellent_threshold = 0.85  # Need 85% success for increases
        self.good_threshold = 0.75       # Need 75% success for gradual increases
        self.struggle_threshold = 0.4    # Success rate for difficulty decrease
        self.crash_penalty_threshold = 0.25  # Crash rate for difficulty decrease

        # Higher threshold for maximum difficulty push
        self.high_difficulty_threshold = 0.8

        # Scenario mixing (highway/merge/intersection ratios)
        self.scenario_mix = self._get_scenario_mix(self.difficulty_level)

        # Modality mixing for multi-modal curriculum
        self.modality_mix = self._get_modality_mix(self.difficulty_level)

        # Curriculum progression tracking
        self.progression_history = []

    def get_current_config(self) -> Dict[str, Any]:
        """Generate environment configuration based on current difficulty."""
        d = self.difficulty_level

        # Scale parameters with difficulty
        config = {
            # fewer cars at low difficulty
            "vehicle_density": int(1 + d * 4),        # 1 → 5 vehicles instead of 5→15

            # keep traffic a bit slower
            "traffic_speed_factor": 0.6 + d * 0.2,   # 60% → 80% speed

            # softer collision penalty
            "collision_penalty": -1.0 - d * 3.0,     # -1.0 → -4.0

            # small arrival bonus
            "success_bonus": 1.0 + d * 2.0,          # 1.0 → 3.0

            "scenario_mix": self._get_scenario_mix(d),
            "difficulty_level": d,
        }

        if self.enable_modality_curriculum:
            config["modality_mix"] = self._get_modality_mix(d)

        return config

    def _get_scenario_mix(self, difficulty: float) -> Dict[str, float]:
        """Intersection-focused scenario mixing for enhanced intersection performance."""

        # ONLY MERGE
        return {"highway": 0.0, "merge": 1.0, "intersection": 0.0}

        if difficulty < 0.3:
            # Foundation: Pure highway mastery
            return {"highway": 1.0, "merge": 0.0, "intersection": 0.0}
        elif difficulty < 0.5:
            # Integration: Highway + merge learning
            return {"highway": 0.7, "merge": 0.3, "intersection": 0.0}
        elif difficulty < 0.7:
            # Challenge: Early intersection introduction with more emphasis
            return {"highway": 0.5, "merge": 0.2, "intersection": 0.3}
        elif difficulty < 0.9:
            # Deep learning: Intersection priority over highway
            return {"highway": 0.3, "merge": 0.2, "intersection": 0.5}
        else:
            # Maximum generalization: Intersection dominant for mastery
            return {"highway": 0.2, "merge": 0.2, "intersection": 0.6}

    def _get_modality_mix(self, difficulty: float) -> Dict[str, float]:
        """Multi-modal curriculum: gradually introduce combined modalities."""

        # ONLY LIDAR
        return {"lidar": 1.0, "grayscale": 0.0, "both": 0.0}

        if not self.enable_modality_curriculum:
            return {"both": 1.0}  # Default to both if curriculum disabled

        if difficulty < 0.2:
            # Foundation: Single modality focus for basic skills
            # Alternate between lidar and grayscale to build diverse foundations
            return {"lidar": 1.0, "grayscale": 0.0, "both": 0.0}
        elif difficulty < 0.4:
            # Introduction: Mix single modalities
            return {"lidar": 0.6, "grayscale": 0.4, "both": 0.0}
        elif difficulty < 0.6:
            # Transition: Introduce combined modality alongside singles
            return {"lidar": 0.3, "grayscale": 0.3, "both": 0.4}
        elif difficulty < 0.8:
            # Integration: Combined modality becomes primary
            return {"lidar": 0.2, "grayscale": 0.2, "both": 0.6}
        else:
            # Mastery: Full multi-modal learning
            return {"lidar": 0.1, "grayscale": 0.1, "both": 0.8}

    def update_difficulty(self, performance: Dict[str, float]) -> bool:
        """
        Update difficulty based on recent performance.
        Returns True if difficulty changed.
        """
        self.performance_window.append(performance)

        # Require more evaluations at high difficulty for stable generalization
        min_evaluations = 8 if self.difficulty_level > self.high_difficulty_threshold else 3
        if len(self.performance_window) < min_evaluations:
            return False  # Need more data points for stable assessment

        # Calculate smoothed metrics
        recent_performances = list(self.performance_window)[-10:]  # Last 10 evaluations
        avg_success = np.mean([p["success_rate"] for p in recent_performances])
        avg_crashes = np.mean([p["crash_rate"] for p in recent_performances])
        avg_reward = np.mean([p["avg_reward"] for p in recent_performances])

        old_difficulty = self.difficulty_level
        difficulty_changed = False

        # Adjust thresholds based on difficulty level - less conservative near maximum
        if self.difficulty_level > self.high_difficulty_threshold:
            # At very high difficulty, still be careful but allow progress to max
            current_excellent_threshold = 0.90  # Need 90% success
            current_good_threshold = 0.85       # Need 85% for gradual increases
            current_crash_threshold = 0.08      # Allow <8% crashes
        else:
            # Standard thresholds for lower difficulties
            current_excellent_threshold = self.excellent_threshold
            current_good_threshold = self.good_threshold
            current_crash_threshold = 0.10

        # Difficulty adjustment logic - larger steps when approaching maximum
        if avg_success > current_excellent_threshold and avg_crashes < current_crash_threshold:
            # Excellent performance - increase step size when close to max
            if self.difficulty_level < self.max_difficulty:
                step_size = 0.10 if self.difficulty_level > 0.8 else 0.08
                self.difficulty_level = min(self.max_difficulty, self.difficulty_level + step_size)
                difficulty_changed = True
                print(".2f")

        elif avg_success > current_good_threshold and avg_crashes < current_crash_threshold:
            # Good performance - increase step size when close to max
            if self.difficulty_level < self.max_difficulty:
                step_size = 0.08 if self.difficulty_level > 0.8 else 0.05
                self.difficulty_level = min(self.max_difficulty, self.difficulty_level + step_size)
                difficulty_changed = True
                print(".2f")

        elif avg_success < self.struggle_threshold or avg_crashes > self.crash_penalty_threshold:
            # Struggling - decrease difficulty to build foundation
            self.difficulty_level = max(self.min_difficulty, self.difficulty_level - 0.08)
            difficulty_changed = True
            print("Agent struggling. Decreasing difficulty to build foundation.")

        # At maximum difficulty, continue training until we achieve mastery
        elif self.difficulty_level >= self.max_difficulty:
            # Require excellent performance at maximum difficulty before stopping
            if avg_success >= 0.95 and avg_crashes <= 0.05:
                print("MASTERED: Excellent performance achieved at maximum difficulty!")
                return True  # Allow difficulty to stay at max and continue training briefly
            else:
                print("At maximum difficulty - continuing training until mastery.")
                return False

        # Log progression
        if difficulty_changed:
            self.progression_history.append({
                "old_difficulty": old_difficulty,
                "new_difficulty": self.difficulty_level,
                "performance": performance,
                "reason": "automatic_adjustment",
                "thresholds_used": {
                    "excellent": current_excellent_threshold,
                    "good": current_good_threshold,
                    "crash_limit": current_crash_threshold
                }
            })

            # Update scenario mix
            self.scenario_mix = self._get_scenario_mix(self.difficulty_level)

            # Update modality mix if enabled
            if self.enable_modality_curriculum:
                self.modality_mix = self._get_modality_mix(self.difficulty_level)

        return difficulty_changed

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of curriculum progression."""
        return {
            "current_difficulty": self.difficulty_level,
            "scenario_mix": self.scenario_mix,
            "modality_mix": self.modality_mix if self.enable_modality_curriculum else None,
            "progression_events": len(self.progression_history),
            "performance_trend": list(self.performance_window)[-5:] if self.performance_window else []
        }

class AdaptiveTrainer:
    """
    Adaptive curriculum learning trainer for autonomous driving agents.

    This class implements progressive learning strategies that adapt the training
    difficulty and curriculum based on agent performance. It supports multiple
    scenarios (highway, merge, intersection) and modalities (lidar, grayscale, both).

    Features:
    - Progressive curriculum advancement
    - Performance-based difficulty scaling
    - Multi-scenario and multi-modality support
    - Comprehensive logging and monitoring

    Attributes:
        curriculum_phases: List of curriculum phases with difficulty progression
        current_phase: Current training phase index
        performance_history: Historical performance metrics
    """
    """
    Trainer that uses performance-driven curriculum learning.
    """

    def __init__(self, modality="both", base_dir="outputs", use_attention=True, checkpoint_path=None, enable_modality_curriculum=True):
        self.base_modality = modality  # Store the base modality
        self.current_modality = modality  # Track current modality for model compatibility
        self.base_dir = base_dir
        self.use_attention = use_attention
        self.checkpoint_path = checkpoint_path
        self.enable_modality_curriculum = enable_modality_curriculum

        # Setup directories
        self.models_dir = os.path.join(base_dir, "models")
        self.logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Initialize curriculum (model created dynamically)
        self.curriculum = AdaptiveCurriculum(enable_modality_curriculum=enable_modality_curriculum)
        self.model = None

        # Training state
        self.total_timesteps = 0
        self.evaluation_interval = 5000

    def _create_model(self, env):
        """Create PPO model with optional attention."""
        policy = "MlpPolicy" if self.current_modality in ["lidar", "both"] else "CnnPolicy"

        # Base hyperparameters
        if self.current_modality in ["lidar", "both"]:
            policy_kwargs = {"net_arch": [256, 128]}  # Larger network for combined observations
        else:
            policy_kwargs = {}  # Default CNN architecture

        # Add attention if requested
        if self.use_attention and self.current_modality == "lidar":
            # Attention module not available - using standard MLP
            # from models.attention import AttentiveLidarExtractor
            # policy_kwargs["features_extractor_class"] = AttentiveLidarExtractor
            # policy_kwargs["features_extractor_kwargs"] = {"features_dim": 128}
            policy_kwargs["net_arch"] = [128]  # Smaller network
            print("Warning: Attention mechanisms disabled (module not available)")

        # Create model with environment
        model = PPO(
            policy,
            env=env,
            verbose=1,
            tensorboard_log=self.logs_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",  # Enable CUDA if available
            policy_kwargs=policy_kwargs,
            n_steps=2048,
            batch_size=256,
            gae_lambda=0.95,
            gamma=0.99,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
        )

        return model

    def _restore_curriculum_state(self):
        """Attempt to restore curriculum state from checkpoint filename."""
        if self.checkpoint_path:
            checkpoint_name = os.path.basename(self.checkpoint_path)

            # Special handling for final model - start at 0.96 for mastery training
            if "final" in checkpoint_name:
                self.curriculum.difficulty_level = 0.50
                print("Starting mastery training at difficulty 0.96")
                return

            # Special handling for retry from specific checkpoint - force difficulty 0.36
            if "adaptive_grayscale_100000.zip" in checkpoint_name:
                self.curriculum.difficulty_level = 0.36
                print("Forcing difficulty level to 0.36 for retry training")
                return

            # Try to extract difficulty level from filename like "adaptive_lidar_25.zip"
            import re
            match = re.search(r'adaptive_.*?_(\d+)\.zip$', checkpoint_name)
            if match:
                difficulty_int = int(match.group(1))
                # If it's a step number (like 50000), estimate difficulty based on training progress
                # From our previous training, 50k steps reached ~0.64 difficulty
                if difficulty_int > 100:  # It's a step number, not difficulty
                    # Rough estimation: difficulty increases roughly linearly with steps
                    estimated_difficulty = min(1.0, (difficulty_int / 100000) * 1.2)  # Conservative estimate
                    self.curriculum.difficulty_level = estimated_difficulty
                    print(f"Estimated difficulty level from {difficulty_int} steps: {self.curriculum.difficulty_level}")
                else:
                    # It's already a difficulty level (0-100)
                    self.curriculum.difficulty_level = difficulty_int / 100.0
                    print(f"Restored difficulty level: {self.curriculum.difficulty_level}")

    def train_adaptive_curriculum(self, total_timesteps=100000, resume_timesteps=0, target_difficulty=1.0):
        """
        Train with adaptive curriculum that scales difficulty based on performance.
        Continues until target_difficulty is reached or total_timesteps is exceeded.
        """
        # print("Starting Adaptive Curriculum Training")
        # print(f"Base Modality: {self.base_modality}")
        print(f"Base Modality: forced Lidar")
        # print(f"Modality Curriculum: {'Enabled' if self.enable_modality_curriculum else 'Disabled'}")
        # print(f"Attention: {'Enabled' if self.use_attention else 'Disabled'}")
        print(f"Target Difficulty: {target_difficulty}")
        print(f"Total timesteps: {total_timesteps}")
        print("=" * 60)

        timesteps_completed = resume_timesteps

        while timesteps_completed < total_timesteps:
            # 1. Get current curriculum configuration
            curr_config = self.curriculum.get_current_config()

            # 2. Create mixed environment based on difficulty
            env = self._create_adaptive_environment(curr_config)

            # 3. Check if we need a new model due to modality change
            # Determine the primary modality from the mix
            if "modality_mix" in curr_config:
                primary_modality = max(curr_config["modality_mix"], key=curr_config["modality_mix"].get)
            else:
                primary_modality = self.base_modality

            needs_new_model = (self.model is None or
                             primary_modality != self.current_modality)

            if needs_new_model:
                self.current_modality = primary_modality
                if self.model is not None:
                    print(f"Modality changed to {primary_modality}, creating new model")

                if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                    print(f"Loading checkpoint from: {self.checkpoint_path}")
                    self.model = PPO.load(self.checkpoint_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
                    # Extract curriculum state from checkpoint filename if possible
                    self._restore_curriculum_state()
                else:
                    self.model = self._create_model(env)
            else:
                try:
                    self.model.set_env(env)
                except ValueError:
                    # If set_env fails (e.g., observation space mismatch), create new model
                    print("Observation space mismatch, creating new model")
                    self.model = self._create_model(env)

            # 3. Setup callbacks for this training segment
            callbacks = [WandbMetricsCallback()]
            ckpt_cb = CheckpointCallback(
                save_freq=10000,  # Save every 10k steps
                save_path=self.models_dir,
                name_prefix=f"adaptive_{self.current_modality}_{int(self.curriculum.difficulty_level*100)}"
            )
            callbacks.append(ckpt_cb)

            # 4. Train for evaluation interval
            steps_to_train = min(self.evaluation_interval, total_timesteps - timesteps_completed)
            self.model.learn(total_timesteps=steps_to_train, callback=CallbackList(callbacks))
            timesteps_completed += steps_to_train

            # 5. Evaluate performance
            performance = self._evaluate_performance(env, n_episodes=15)

            # 6. Update curriculum based on performance
            difficulty_changed = self.curriculum.update_difficulty(performance)

            # 7. Log progress
            progress = self.curriculum.get_progress_summary()
            self._log_progress(timesteps_completed, performance, progress)

            # 8. Save checkpoint model
            if difficulty_changed or timesteps_completed % 20000 == 0:
                self._save_checkpoint(timesteps_completed)

        # Final save
        self._save_checkpoint(timesteps_completed, final=True)

        if self.curriculum.difficulty_level >= target_difficulty:
            # Check if we achieved mastery (last few evaluations)
            recent_evals = list(self.curriculum.performance_window)[-5:] if len(self.curriculum.performance_window) >= 5 else self.curriculum.performance_window
            if recent_evals:
                avg_success = np.mean([p["success_rate"] for p in recent_evals])
                avg_crashes = np.mean([p["crash_rate"] for p in recent_evals])
                if avg_success >= 0.95 and avg_crashes <= 0.05:
                    print(f"MASTERY ACHIEVED: Difficulty {self.curriculum.difficulty_level:.2f} with {avg_success:.1%} success and {avg_crashes:.1%} crash rate!")
                    print("Adaptive curriculum training completed successfully!")
                else:
                    print(f"TARGET REACHED: Difficulty {self.curriculum.difficulty_level:.2f} achieved, but mastery not yet reached.")
                    print(f"Recent performance: {avg_success:.1%} success, {avg_crashes:.1%} crash rate")
                    print("Consider additional training for full mastery.")
            else:
                print(f"TARGET ACHIEVED: Difficulty {self.curriculum.difficulty_level:.2f} reached!")
                print("Adaptive curriculum training completed successfully!")
        else:
            print(f"TIME LIMIT: Reached {total_timesteps} timesteps with difficulty {self.curriculum.difficulty_level:.2f}")
            print("Consider increasing timesteps or adjusting thresholds to reach target difficulty.")

        return self.model

    def _create_adaptive_environment(self, config: Dict[str, Any]):
        """Create environment that mixes scenarios based on curriculum config."""
        from stable_baselines3.common.vec_env import DummyVecEnv

        def make_env():
            """
            Create a vectorized environment for training.

            Returns:
                DummyVecEnv: Vectorized environment wrapper
            """

            # Hardcoding merge lidar run
            scenario = "merge"
            sampled_modality = self.base_modality

            # Select scenario based on mixing weights
            #scenarios = list(config["scenario_mix"].keys())
            #weights = list(config["scenario_mix"].values())
            #scenario = np.random.choice(scenarios, p=weights)

            # Select modality based on mixing weights (if curriculum enabled)
            """
            if "modality_mix" in config:
                modalities = list(config["modality_mix"].keys())
                modality_weights = list(config["modality_mix"].values())
                sampled_modality = np.random.choice(modalities, p=modality_weights)
            else:
                sampled_modality = self.base_modality
            """

            # Get curriculum configuration
            env_config = get_curriculum_config(scenario, "easy", sampled_modality)

            # Override with adaptive parameters
            env_config["vehicles_count"] = config["vehicle_density"]
            env_config["collision_reward"] = config["collision_penalty"]
            env_config["arrived_reward"] = config["success_bonus"]

            # Adjust speed ranges based on traffic speed factor
            if "reward_speed_range" in env_config:
                base_range = env_config["reward_speed_range"]
                env_config["reward_speed_range"] = [
                    base_range[0] * config["traffic_speed_factor"],
                    base_range[1] * config["traffic_speed_factor"]
                ]

            # Create and configure environment BEFORE wrapping
            #env = gym.make(f"{scenario}-v0", render_mode=None)
            #env.unwrapped.configure(env_config)
            #env.reset()  # Initialize with new config

            env = UrbanJunctionEnv(
                config=env_config,
                scenario=scenario,
                modality=sampled_modality,
                render_mode=None,
            )       

            return env

        return DummyVecEnv([make_env])

    def _evaluate_performance(self, env, n_episodes=15) -> Dict[str, float]:
        """Evaluate current agent performance."""
        success_count = 0
        total_reward = 0.0
        crash_count = 0
        episode_lengths = []

        for episode in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0.0
            episode_crashes = 0
            steps = 0

            while not done and steps < 100:  # Max episode length
                action, _ = self.model.predict(obs, deterministic=True)

                # Handle different Gymnasium API versions
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                elif len(step_result) == 4:
                    obs, reward, done, info = step_result
                    terminated = done  # For compatibility
                    truncated = False
                else:
                    raise ValueError(f"Unexpected step result length: {len(step_result)}")

                episode_reward += reward
                steps += 1

                # Check for crashes (handle different env info formats)
                if hasattr(info, '__iter__'):  # Handle vec env
                    for info_item in info:
                        if isinstance(info_item, dict) and info_item.get("crashed", False):
                            episode_crashes += 1
                elif isinstance(info, dict) and info.get("crashed", False):
                    episode_crashes += 1

            # Episode complete
            total_reward += episode_reward
            crash_count += episode_crashes
            episode_lengths.append(steps)

            # Success criteria: positive reward and no crashes
            # Success criteria: survived reasonably long and didn't crash too much
            if episode_reward > 0 or (steps > 60 and episode_crashes == 0):
                success_count += 1


        return {
            "success_rate": float(success_count / n_episodes),
            "avg_reward": float(total_reward / n_episodes),
            "crash_rate": float(crash_count / n_episodes),
            "avg_episode_length": float(np.mean(episode_lengths)),
            "episodes": n_episodes
        }

    def _log_progress(self, timesteps, performance, progress):
        """Log training progress."""
        print(f"\nProgress @ {timesteps:,} timesteps:")
        print(f"   Difficulty: {progress['current_difficulty']:.2f}")
        print(f"   Success Rate: {performance['success_rate']:.2f}")
        print(f"   Crash Rate: {performance['crash_rate']:.2f}")
        print(f"   Avg Reward: {performance['avg_reward']:.2f}")
        print(f"   Scenario Mix: {progress['scenario_mix']}")
        if progress.get('modality_mix'):
            print(f"   Modality Mix: {progress['modality_mix']}")

        # Log to wandb if available
        if wandb.run is not None:
            log_data = {
                "timesteps": timesteps,
                "difficulty_level": progress['current_difficulty'],
                "success_rate": performance['success_rate'],
                "crash_rate": performance['crash_rate'],
                "avg_reward": performance['avg_reward'],
                "scenario_highway": progress['scenario_mix']['highway'],
                "scenario_merge": progress['scenario_mix']['merge'],
                "scenario_intersection": progress['scenario_mix']['intersection'],
            }
            if progress.get('modality_mix'):
                log_data.update({
                    "modality_lidar": progress['modality_mix'].get('lidar', 0),
                    "modality_grayscale": progress['modality_mix'].get('grayscale', 0),
                    "modality_both": progress['modality_mix'].get('both', 0),
                })
            wandb.log(log_data)

    def _save_checkpoint(self, timesteps, final=False):
        """Save model checkpoint."""
        suffix = "final" if final else f"{timesteps}"
        save_path = os.path.join(self.models_dir, f"adaptive_{self.current_modality}_{suffix}.zip")
        self.model.save(save_path)
        print(f"Model saved: {save_path}")

def run_adaptive_curriculum(modality="both", total_timesteps=100000, use_attention=True, use_wandb=True, checkpoint_path=None, target_difficulty=1.0, enable_modality_curriculum=True):
    """
    Run adaptive curriculum training with modality progression.

    Args:
        modality: Base modality ("lidar", "grayscale", or "both")
        total_timesteps: Total training steps
        use_attention: Whether to use attention mechanisms
        use_wandb: Whether to log to Weights & Biases
        checkpoint_path: Path to checkpoint to resume from
        target_difficulty: Target difficulty level to reach (0.0-1.0)
        enable_modality_curriculum: Whether to enable modality progression
    """
    if use_wandb:
        wandb.init(
            project="autonomous-driving-adaptive",
            name=f"adaptive-{modality}-{'modality' if enable_modality_curriculum else 'single'}-{'attention' if use_attention else 'baseline'}-to-{target_difficulty}",
            config={
                "modality": modality,
                "use_attention": use_attention,
                "total_timesteps": total_timesteps,
                "checkpoint_path": checkpoint_path,
                "target_difficulty": target_difficulty,
                "enable_modality_curriculum": enable_modality_curriculum,
            }
        )

    trainer = AdaptiveTrainer(modality=modality, use_attention=use_attention, checkpoint_path=checkpoint_path, enable_modality_curriculum=enable_modality_curriculum)
    final_model = trainer.train_adaptive_curriculum(total_timesteps=total_timesteps, target_difficulty=target_difficulty)

    if use_wandb:
        wandb.finish()

    return final_model
