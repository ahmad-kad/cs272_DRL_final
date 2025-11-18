#!/usr/bin/env python3
"""
Simple Training Logger - Less is More

Minimal logging for RL training with essential insights only.
"""

import csv
import time
import numpy as np
from pathlib import Path
import os


class SimpleLogger:
    """Minimal logger focused on essential training insights."""

    def __init__(self, phase_name: str, outputs_dir: str = "outputs", use_wandb: bool = True):
        self.phase_name = phase_name
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(exist_ok=True)

        self.episodes = []
        self.start_time = time.time()
        self.use_wandb = use_wandb and os.getenv('WANDB_DISABLED') != 'true'

        # Initialize wandb if enabled
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project="urban-junction-rl",
                    name=f"{phase_name}_training",
                    config={
                        "phase": phase_name,
                        "test_mode": os.getenv('TEST_MODE') == 'true'
                    }
                )
                self.wandb = wandb
                print("Wandb logging enabled")
            except ImportError:
                print("Warning: Wandb not available, continuing without logging")
                self.use_wandb = False
        else:
            self.use_wandb = False
            print("Info: Wandb logging disabled")

    def log_episode(self, reward: float, success: bool = False):
        """Log episode result and show progress every 100 episodes."""
        self.episodes.append({'reward': reward, 'success': success})

        episode_num = len(self.episodes)
        # Print more frequently in TEST_MODE
        log_frequency = 10 if os.getenv('TEST_MODE') == 'true' else 100

        if episode_num % log_frequency == 0:
            window_size = min(episode_num, 100)  # Use available episodes for averaging
            recent_rewards = [e['reward'] for e in self.episodes[-window_size:]]
            avg_reward = np.mean(recent_rewards)
            success_rate = np.mean([e['success'] for e in self.episodes[-window_size:]])

            print(f"Episode {episode_num}: {avg_reward:.2f} avg reward, {success_rate:.1%} success")

            # Log to wandb
            if self.use_wandb:
                self.wandb.log({
                    "episode": episode_num,
                    "avg_reward": avg_reward,
                    "success_rate": success_rate,
                    "episode_reward": reward,
                    "episode_success": success
                })

    def save_results(self):
        """Save training results to CSV and finalize logging."""
        csv_path = self.outputs_dir / f"{self.phase_name}_results.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['episode', 'reward', 'success'])
            writer.writeheader()
            for i, ep in enumerate(self.episodes):
                writer.writerow({'episode': i+1, 'reward': ep['reward'], 'success': ep['success']})

        print(f"Results saved to: {csv_path}")

        # Final wandb logging
        if self.use_wandb:
            total_time = time.time() - self.start_time
            final_reward = np.mean([e['reward'] for e in self.episodes[-100:]])
            final_success = np.mean([e['success'] for e in self.episodes[-100:]])

            self.wandb.log({
                "final_avg_reward": final_reward,
                "final_success_rate": final_success,
                "total_episodes": len(self.episodes),
                "training_time_hours": total_time / 3600
            })
            self.wandb.finish()
            print("Wandb logging completed")

        return csv_path


# Convenience functions for each phase
def create_phase1_logger():
    """Create logger for Phase 1 training."""
    return SimpleLogger("phase1")

def create_phase2_logger():
    """Create logger for Phase 2 training."""
    return SimpleLogger("phase2")

def create_phase3_logger():
    """Create logger for Phase 3 training."""
    return SimpleLogger("phase3")

