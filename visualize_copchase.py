# visualize_copchase.py

import argparse
import os
import time
import statistics

import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("[WARNING] matplotlib/seaborn not available, violin plots disabled")

# Try to import TQC
try:
    from sb3_contrib import TQC
    TQC_AVAILABLE = True
except ImportError:
    TQC = None
    TQC_AVAILABLE = False

# Import the custom environment
from team2_env.crazy_driver_enviornment import crazy_driver_env


def make_env(render_mode: str = "rgb_array"):
    """
    Create the CopChase environment for visualization.

    render_mode="rgb_array" → env.render() returns RGB frames (good for GIFs).
    render_mode="none" → No rendering, just evaluation.
    If you want a visible window instead, you can change to "human".
    """
    config = crazy_driver_env.default_config()

    # Visualization-specific config overrides
    actual_render_mode = None if render_mode == "none" else render_mode
    config.update({
        "offscreen_rendering": render_mode == "rgb_array",
        "render_agent": render_mode != "none",  # Only render agent if we're rendering
        "show_trajectories": False,  # Don't show trajectories for cleaner visualization
    })

    env = gym.make(
        "CopChase-v0",
        render_mode=actual_render_mode,
        config=config,
    )
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the CopChase environment (crazy driver scenario) with TQC models."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to a trained TQC model .zip file (optional - if not provided, uses random actions)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to visualize (default: 3)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Delay between steps in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--save_gif",
        action="store_true",
        help="If set, save each episode as a GIF in outputs/visualization",
    )
    parser.add_argument(
        "--gif_dir",
        type=str,
        default="outputs/visualization",
        help="Directory to save GIFs (default: outputs/visualization)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Max steps per episode during visualization/GIF (default: 200)",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        choices=["human", "rgb_array", "none"],
        default="rgb_array",
        help="Render mode: 'human' for live window, 'rgb_array' for GIFs, 'none' for no rendering (default: rgb_array)",
    )
    parser.add_argument(
        "--evaluate_stats",
        action="store_true",
        help="Run evaluation and output statistics (success rate, reward, length) for analysis",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create environment
    env = make_env(render_mode=args.render_mode)

    # Load VecNormalize for TQC models (they need normalization for proper behavior)
    if args.model:
        vec_normalize_path = args.model.replace('.zip', '_vecnormalize.pkl')
        if os.path.exists(vec_normalize_path):
            print("[INFO] Loading VecNormalize stats for TQC model (required for proper behavior)")
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = False  # Set to evaluation mode
            env.norm_reward = False  # Don't normalize rewards during visualization

    # Load model if provided, otherwise use random actions
    if args.model:
        if not TQC_AVAILABLE:
            raise ImportError("TQC not available - install sb3-contrib")

        # Load TQC model
        model = TQC.load(args.model)
        print(f"[LOAD] TQC model loaded from {args.model}")
        use_model = True
    else:
        model = None
        print("[INFO] No model provided - using random actions")
        use_model = False

    print(f"[ENV] CopChase-v0 environment")
    print(f"[ENV] Action space: {env.action_space}")
    print(f"[ENV] Observation space: {env.observation_space}")
    config = env.unwrapped.config
    print(f"[ENV] Duration: {config['duration']} seconds")
    print(f"[ENV] Vehicles: {config['vehicles_count']} NPCs + {config['cop_count']} cops")

    if args.save_gif:
        os.makedirs(args.gif_dir, exist_ok=True)

    # Statistics collection for evaluation
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    crash_count = 0

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        steps = 0
        frames = []

        print(f"\n[EPISODE {ep}] Starting...")

        while not (done or truncated) and steps < args.max_steps:
            if use_model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Random action: sample from action space
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1

            if args.render_mode == "rgb_array":
                frame = env.render()  # numpy array
                frames.append(frame)
            elif args.render_mode == "human":
                env.render()
                time.sleep(args.delay)
            # For "none" mode, skip rendering entirely

        crashed = info.get('crashed', False)
        print(f"Episode {ep}: steps={steps}, reward={ep_reward:.2f}, crashed={crashed}")

        # Collect statistics
        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)
        if crashed:
            crash_count += 1
        else:
            success_count += 1

        if args.save_gif and args.render_mode == "rgb_array" and frames:
            if use_model:
                model_name = args.model.split('/')[-1].split('.')[0]  # Extract model name
                gif_name = f"copchase_{model_name}_ep{ep}.gif"
            else:
                gif_name = f"copchase_random_ep{ep}.gif"

            gif_path = os.path.join(args.gif_dir, gif_name)
            duration = int(1000 * args.delay) if args.delay > 0 else 50  # duration in ms
            imageio.mimsave(gif_path, frames, duration=duration)
            print(f"[GIF] Saved {gif_path}")

    env.close()

    # Output evaluation statistics
    if args.evaluate_stats and args.episodes > 1:
        print("\n" + "="*60)
        print("EVALUATION STATISTICS")
        print("="*60)

        success_rate = success_count / args.episodes
        crash_rate = crash_count / args.episodes

        avg_reward = statistics.mean(episode_rewards)
        reward_std = statistics.stdev(episode_rewards) if len(episode_rewards) > 1 else 0

        avg_length = statistics.mean(episode_lengths)
        length_std = statistics.stdev(episode_lengths) if len(episode_lengths) > 1 else 0

        print(f"Model: {args.model.split('/')[-1] if args.model else 'Random'}")
        print(f"Episodes evaluated: {args.episodes}")
        print()
        print("SUCCESS METRICS:")
        print(f"  Success rate: {success_rate:.3f} ({success_count}/{args.episodes})")
        print(f"  Crash rate: {crash_rate:.3f} ({crash_count}/{args.episodes})")
        print()
        print("REWARD STATISTICS:")
        print(f"  Average reward: {avg_reward:.2f} ± {reward_std:.2f}")
        print(f"  Reward range: {min(episode_rewards):.2f} to {max(episode_rewards):.2f}")
        print()
        print("EPISODE LENGTH STATISTICS:")
        print(f"  Average length: {avg_length:.1f} ± {length_std:.1f} steps")
        print(f"  Length range: {min(episode_lengths)} to {max(episode_lengths)} steps")

        # Generate violin plots if plotting libraries are available
        if PLOTTING_AVAILABLE and args.save_gif:
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                # Reward violin plot
                sns.violinplot(y=episode_rewards, ax=ax1, color='skyblue')
                ax1.set_title(f'Reward Distribution\n{args.episodes} Episodes')
                ax1.set_ylabel('Episode Reward')
                ax1.grid(True, alpha=0.3)

                # Episode length violin plot
                sns.violinplot(y=episode_lengths, ax=ax2, color='lightgreen')
                ax2.set_title(f'Episode Length Distribution\n{args.episodes} Episodes')
                ax2.set_ylabel('Episode Length (steps)')
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()

                # Save the plot
                model_name = args.model.split('/')[-1].split('.')[0] if args.model else 'random'
                plot_path = os.path.join(args.gif_dir, f'{model_name}_violin_plots.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"[PLOT] Violin plots saved to {plot_path}")

                plt.close()

            except Exception as e:
                print(f"[WARNING] Failed to generate violin plots: {e}")

        print("="*60)


if __name__ == "__main__":
    main()
