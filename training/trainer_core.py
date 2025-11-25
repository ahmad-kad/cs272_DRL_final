import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import wandb
import os
from utils.config import get_curriculum_config
from utils.callbacks import WandbMetricsCallback, AdaptiveCurriculumCallback

class CurriculumTrainer:
    def __init__(self, modality="lidar", base_dir="outputs"):
        self.modality = modality
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "models")
        self.logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def train_phase(self, phase_name, env_name, timesteps, model=None, difficulty="easy", adaptive=True):
        """
        Trains a single curriculum phase.
        adaptive: If True, stops early if success threshold is met.
        """
        print(f"\n=== Starting Phase: {phase_name} ({env_name} - {difficulty}) ===")
        
        # 1. Setup Environment
        config = get_curriculum_config(env_name, difficulty, self.modality)
        env = gym.make(env_name, render_mode=None)
        env.unwrapped.configure(config)
        env.reset()
        
        # 2. Initialize or Load Model
        if model is None:
            policy = "MlpPolicy" if self.modality == "lidar" else "CnnPolicy"
            model = PPO(policy, env, verbose=1, tensorboard_log=self.logs_dir, learning_rate=3e-4)
        else:
            model.set_env(env)
        
        # 3. Callbacks
        callbacks = [WandbMetricsCallback()]
        
        # Checkpoint every 10k steps
        ckpt_cb = CheckpointCallback(
            save_freq=10000, 
            save_path=self.models_dir, 
            name_prefix=f"{self.modality}_{phase_name}"
        )
        callbacks.append(ckpt_cb)
        
        # Adaptive Progression
        if adaptive:
            # Thresholds: Easy=0.8, Medium=0.85, Hard=0.9
            threshold = 0.8 if difficulty == "easy" else (0.85 if difficulty == "medium" else 0.9)
            adaptive_cb = AdaptiveCurriculumCallback(success_threshold=threshold, min_steps=5000)
            callbacks.append(adaptive_cb)
        
        # 4. Train
        # If adaptive, we might stop early, but we set an upper limit with 'total_timesteps'
        model.learn(total_timesteps=timesteps, callback=CallbackList(callbacks))
        
        # 5. Check completion status
        if adaptive and adaptive_cb.goal_reached:
            print(f"Phase {phase_name} completed early due to high performance!")
        
        # 6. Save Final Model for this Phase
        save_path = os.path.join(self.models_dir, f"{self.modality}_{phase_name}_final")
        model.save(save_path)
        print(f"Phase {phase_name} complete. Model saved to {save_path}")
        
        return model

