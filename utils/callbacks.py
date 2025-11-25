from stable_baselines3.common.callbacks import BaseCallback
import wandb
import numpy as np
from collections import deque

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

