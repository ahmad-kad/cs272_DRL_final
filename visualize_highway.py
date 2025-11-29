#!/usr/bin/env python3
"""
Visualize trained autonomous driving model across all scenarios and modalities.

This script loads a trained model and runs inference visualization
on highway, merge, and intersection scenarios with lidar or grayscale modalities.
Supports both real-time pygame visualization and GIF recording.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠️ PIL not available - GIF saving disabled")

from environments.enhanced_urban_env import EnhancedUrbanJunctionEnv


def visualize_model(model_path: str, scenario: str = "highway", modality: str = "lidar", episodes: int = 5, max_steps: int = 1000, save_gif: bool = False, gif_fps: int = 10, output_filename: str = None):
    """
    Visualize a trained model on the specified scenario.

    Args:
        model_path: Path to the saved model (.zip file)
        scenario: Scenario to visualize ("highway", "merge", "intersection")
        episodes: Number of episodes to run
        max_steps: Maximum steps per episode
    """
    print(f"🎮 Visualizing model: {model_path}")
    print(f"🛣️ Scenario: {scenario}")
    print(f"📡 Modality: {modality}")
    print(f"🎯 Episodes: {episodes}")
    if save_gif:
        print(f"🎬 GIF recording enabled (FPS: {gif_fps})")
    print("=" * 60)

    # Load the model
    try:
        model = PPO.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Modality is now passed as parameter, no need to auto-detect

    # Create environment with visualization and longer duration
    config = {
        "duration": max_steps,  # Set duration to match max_steps for visualization
        "simulation_frequency": 10,  # 10 Hz for smoother visualization
        "policy_frequency": 10,
    }

    # Use rgb_array mode for GIF saving, human mode for display
    render_mode = "rgb_array" if save_gif else "human"

    env = EnhancedUrbanJunctionEnv(
        config=config,
        scenario=scenario,
        modality=modality,
        render_mode=render_mode
    )
    env = Monitor(env)

    try:
        # Storage for GIF frames if saving
        gif_frames = [] if save_gif else None

        for episode in range(episodes):
            print(f"\n🎬 Episode {episode + 1}/{episodes}")
            print("-" * 30)

            obs, info = env.reset()
            episode_reward = 0.0
            steps = 0
            terminated = False
            truncated = False
            crash_detected = False
            action_counts = {"lane_left": 0, "idle": 0, "lane_right": 0, "faster": 0, "slower": 0}

            while not (terminated or truncated) and steps < max_steps:
                # Get model prediction
                action, _ = model.predict(obs, deterministic=True)

                # Step environment
                step_result = env.step(action)

                if len(step_result) == 5:
                    next_obs, reward, terminated, truncated, info = step_result
                else:
                    next_obs, reward, terminated, truncated = step_result[:4]

                episode_reward += reward
                obs = next_obs
                steps += 1

                # Track actions
                action_names = ["lane_left", "idle", "lane_right", "faster", "slower"]
                if 0 <= action < len(action_names):
                    action_counts[action_names[action]] += 1

                # Print model actions and state (for debugging when pygame not visible)
                if steps % 50 == 0:  # Print every 50 steps
                    action_name = action_names[action] if 0 <= action < len(action_names) else f"unknown_{action}"
                    print(f"Step {steps}: Action={action_name}, Reward={reward:.2f}, Total Reward={episode_reward:.2f}")

                # Render the environment
                frame = env.render()

                # Capture frame for GIF if enabled
                if save_gif and frame is not None:
                    # Convert to PIL Image and resize if too large
                    if isinstance(frame, np.ndarray):
                        pil_frame = Image.fromarray(frame)

                        # Resize large frames for GIF efficiency (max 640px width)
                        if pil_frame.width > 640:
                            aspect_ratio = pil_frame.height / pil_frame.width
                            new_width = 640
                            new_height = int(new_width * aspect_ratio)
                            pil_frame = pil_frame.resize((new_width, new_height), Image.Resampling.LANCZOS)

                        gif_frames.append(pil_frame)

                # Small delay for visualization (only when not saving GIF)
                if not save_gif:
                    time.sleep(0.02)

                # Check for crash
                if hasattr(env, 'vehicle') and env.vehicle.crashed:
                    crash_detected = True
                    print(f"💥 Crash detected at step {steps}")
                    break

            # Episode summary
            print("📊 Episode Results:")
            print(f"   Steps: {steps}")
            print(".2f")
            print(f"   Terminated: {terminated} (success)")
            print(f"   Truncated: {truncated} (timeout)")
            print(f"   Crashed: {crash_detected}")

            # Action distribution
            print("🎮 Action Distribution:")
            total_actions = sum(action_counts.values())
            if total_actions > 0:
                for action_name, count in action_counts.items():
                    percentage = (count / total_actions) * 100
                    print(".1f")

            # Save GIF for this episode if enabled
            if save_gif and gif_frames and HAS_PIL:
                if output_filename:
                    gif_filename = output_filename
                else:
                    gif_filename = f"highway_episode_{episode + 1}.gif"
                print(f"💾 Saving GIF: {gif_filename}")

                # Calculate frame duration based on desired FPS
                frame_duration = int(1000 / gif_fps)  # milliseconds

                # Save as GIF
                gif_frames[0].save(
                    gif_filename,
                    save_all=True,
                    append_images=gif_frames[1:],
                    duration=frame_duration,
                    loop=0  # Infinite loop
                )
                print(f"✅ GIF saved: {gif_filename} ({len(gif_frames)} frames)")

                # Clear frames for next episode
                gif_frames = [] if save_gif else None

            # Wait a bit between episodes
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n⏹️ Visualization stopped by user")

    except Exception as e:
        print(f"❌ Visualization error: {e}")

    finally:
        env.close()
        print("\n🏁 Visualization completed")


def main():
    parser = argparse.ArgumentParser(description="Visualize trained RL model on highway scenarios")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/models/generalized_lidar_phase_3/phase_3_152880_steps.zip",
        help="Path to trained model (.zip file)"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="highway",
        choices=["highway", "merge", "intersection"],
        help="Driving scenario to visualize (ignored in batch mode)"
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="lidar",
        choices=["lidar", "grayscale"],
        help="Observation modality to use (ignored in batch mode)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--save-gif",
        action="store_true",
        help="Save episodes as GIF animations"
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=10,
        help="Frames per second for GIF (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output GIF filename (default: highway_episode_X.gif)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run visualization on all scenarios and modalities"
    )

    args = parser.parse_args()

    # Check if model file exists
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")

        # Try to find available models
        model_dir = Path("outputs/models")
        if model_dir.exists():
            print("\n📁 Available models:")
            for model_file in model_dir.rglob("*.zip"):
                print(f"  {model_file}")

        sys.exit(1)

    if args.batch:
        # Run visualization on all scenario/modality combinations
        scenarios = ["highway", "merge", "intersection"]
        modalities = ["lidar", "grayscale"]

        print("🔄 Running batch visualization across all scenarios and modalities...")
        print(f"Testing {len(scenarios)} scenarios × {len(modalities)} modalities = {len(scenarios) * len(modalities)} combinations")
        print("=" * 80)

        for scenario in scenarios:
            for modality in modalities:
                print(f"\n🎯 Testing {scenario} scenario with {modality} modality")
                gif_name = f"{scenario}_{modality}_demo.gif" if args.save_gif else None
                visualize_model(args.model, scenario, modality, 1, args.max_steps,
                              args.save_gif, args.gif_fps, gif_name)
    else:
        # Single scenario/modality visualization
        visualize_model(args.model, args.scenario, args.modality, args.episodes,
                       args.max_steps, args.save_gif, args.gif_fps, args.output)


if __name__ == "__main__":
    main()
