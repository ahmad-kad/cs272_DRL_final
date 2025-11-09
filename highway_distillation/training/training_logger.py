#!/usr/bin/env python3
"""
Simple Training Logger - Less is More

Minimal logging for RL training with essential insights only.
"""

import csv
import time
import numpy as np
from pathlib import Path


class SimpleLogger:
    """Minimal logger focused on essential training insights."""

    def __init__(self, phase_name: str, outputs_dir: str = "outputs"):
        self.phase_name = phase_name
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(exist_ok=True)

        self.episodes = []
        self.start_time = time.time()

    def log_episode(self, reward: float, success: bool = False):
        """Log episode result and show progress every 100 episodes."""
        self.episodes.append({'reward': reward, 'success': success})

        episode_num = len(self.episodes)
        if episode_num % 100 == 0:
            recent_rewards = [e['reward'] for e in self.episodes[-100:]]
            avg_reward = np.mean(recent_rewards)
            success_rate = np.mean([e['success'] for e in self.episodes[-100:]])

            print(f"Episode {episode_num}: {avg_reward:.2f} avg reward, {success_rate:.1%} success")

    def save_results(self):
        """Save training results to CSV."""
        csv_path = self.outputs_dir / f"{self.phase_name}_results.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['episode', 'reward', 'success'])
            writer.writeheader()
            for i, ep in enumerate(self.episodes):
                writer.writerow({'episode': i+1, 'reward': ep['reward'], 'success': ep['success']})

        print(f"Results saved to: {csv_path}")
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

