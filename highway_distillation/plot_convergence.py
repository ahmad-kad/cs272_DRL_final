#!/usr/bin/env python3
"""
Convergence Plotting Script

Creates convergence plots from exported CSV data to visualize training progress.

Usage:
    python plot_convergence.py outputs/data/phase1_convergence_data.csv
    python plot_convergence.py outputs/data/phase2_convergence_data.csv --save
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use('default')  # Clean style


def plot_convergence(csv_path: str, save_path: str = None, show_plot: bool = True):
    """
    Create convergence plots from CSV data.

    Args:
        csv_path: Path to convergence CSV file
        save_path: Path to save plot (optional, defaults to outputs/plots/)
        show_plot: Whether to display plot
    """
    # Default save path in outputs/plots directory
    if save_path is None:
        csv_name = Path(csv_path).stem
        save_path = f"outputs/plots/{csv_name}_convergence.png"
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return

    # Load data
    df = pd.read_csv(csv_path)

    if df.empty:
        print("Error: CSV file is empty")
        return

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Convergence Analysis - {Path(csv_path).parent.name}', fontsize=16, fontweight='bold')

    # 1. Reward progression with moving averages
    ax1.plot(df['episode'], df['reward'], 'b-', alpha=0.6, linewidth=1, label='Episode Reward')
    if 'moving_avg_reward_50' in df.columns:
        ax1.plot(df['episode'], df['moving_avg_reward_50'], 'r-', linewidth=2,
                label='50-Episode Moving Avg')
    if 'moving_avg_reward_100' in df.columns:
        ax1.plot(df['episode'], df['moving_avg_reward_100'], 'g-', linewidth=2,
                label='100-Episode Moving Avg')
    ax1.set_title('Reward Progression')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Reward distribution (recent episodes)
    if len(df) > 50:
        recent_rewards = df['reward'].tail(100)  # Last 100 episodes
        ax2.hist(recent_rewards, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(recent_rewards.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {recent_rewards.mean():.2f}')
        ax2.axvline(recent_rewards.median(), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {recent_rewards.median():.2f}')
        ax2.set_title('Recent Reward Distribution (Last 100 Episodes)')
        ax2.set_xlabel('Reward')
        ax2.set_ylabel('Frequency')
        ax2.legend()
    else:
        # Fallback: simple histogram of all data
        ax2.hist(df['reward'], bins=max(10, len(df)//10), alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Reward Distribution')
        ax2.set_xlabel('Reward')
        ax2.set_ylabel('Frequency')

    # 3. Success rate progression
    if 'cumulative_success_rate' in df.columns:
        ax3.plot(df['episode'], df['cumulative_success_rate'] * 100, 'green', linewidth=2)
        ax3.set_title('Success Rate Progression')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Success Rate (%)')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)

        # Add final success rate annotation
        final_success = df['cumulative_success_rate'].iloc[-1] * 100
        ax3.annotate('.1f',
                    xy=(df['episode'].iloc[-1], final_success),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                    fontsize=10, ha='left')

    # 4. Episode length progression
    ax4.plot(df['episode'], df['length'], 'purple', alpha=0.7, linewidth=1)
    if len(df) > 20:
        # Add moving average for episode lengths
        ma_length = df['length'].rolling(window=min(50, len(df)//5), center=True).mean()
        ax4.plot(df['episode'], ma_length, 'orange', linewidth=2, label='Moving Avg')
        ax4.legend()
    ax4.set_title('Episode Length Progression')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Episode Length (steps)')
    ax4.grid(True, alpha=0.3)

    # Calculate and display convergence metrics
    if len(df) >= 50:
        # Rate of change over last 50 episodes
        recent_rewards = df['reward'].tail(50)
        reward_slope = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]

        # Convergence assessment
        recent_std = recent_rewards.std()
        if abs(reward_slope) < 0.01 and recent_std < 1.0:
            convergence_status = "CONVERGED"
            status_color = "green"
        elif reward_slope > 0.05:
            convergence_status = "IMPROVING"
            status_color = "blue"
        elif reward_slope < -0.05:
            convergence_status = "DECLINING"
            status_color = "red"
        else:
            convergence_status = "STABLE"
            status_color = "orange"

        # Add convergence status as text
        fig.text(0.02, 0.02,
                f'Convergence Status: {convergence_status}\n'
                f'Final Reward: {df["reward"].iloc[-1]:.2f}\n'
                f'Reward Slope: {reward_slope:+.4f}/episode\n'
                f'Std Dev (recent): {recent_std:.2f}',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=status_color, alpha=0.1),
                verticalalignment='bottom')

    plt.tight_layout()

    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Convergence plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_multiple_phases(csv_paths: list, labels: list = None, save_path: str = None):
    """
    Plot convergence comparison across multiple phases.

    Args:
        csv_paths: List of CSV file paths
        labels: List of labels for each CSV (optional)
        save_path: Path to save comparison plot
    """
    if not labels:
        labels = [f'Phase {i+1}' for i in range(len(csv_paths))]

    plt.figure(figsize=(12, 8))

    for i, (csv_path, label) in enumerate(zip(csv_paths, labels)):
        if not os.path.exists(csv_path):
            print(f"Warning: CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        color = plt.cm.tab10(i % 10)  # Cycle through colors
        plt.plot(df['episode'], df['reward'], '-', color=color, alpha=0.7,
                linewidth=2, label=f'{label} (reward)')
        if 'moving_avg_reward_50' in df.columns:
            plt.plot(df['episode'], df['moving_avg_reward_50'], '--',
                    color=color, linewidth=3, alpha=0.9)

    plt.title('Multi-Phase Convergence Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot RL convergence from CSV data')
    parser.add_argument('csv_files', nargs='+', help='CSV file(s) to plot')
    parser.add_argument('--save', '-s', help='Save plot to file')
    parser.add_argument('--no-show', action='store_true', help='Don\'t display plot')
    parser.add_argument('--compare', '-c', action='store_true',
                       help='Compare multiple phases in one plot')

    args = parser.parse_args()

    if args.compare and len(args.csv_files) > 1:
        # Multi-phase comparison
        labels = [Path(csv).parent.name for csv in args.csv_files]
        plot_multiple_phases(args.csv_files, labels, args.save)
    else:
        # Individual plots
        for csv_file in args.csv_files:
            plot_name = Path(csv_file).stem
            save_path = args.save or f'{plot_name}_convergence.png'
            plot_convergence(csv_file, save_path, not args.no_show)


if __name__ == '__main__':
    main()
