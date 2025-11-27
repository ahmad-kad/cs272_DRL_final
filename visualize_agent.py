#!/usr/bin/env python3
"""
Visualize Trained Agent Driving

This script loads a trained model and shows the agent driving in the environment
with a persistent pygame window so you can actually see the visualization.

Usage:
    python visualize_agent.py --model outputs/models/adaptive_grayscale_final.zip --scenario intersection
    python visualize_agent.py --model outputs/models/adaptive_grayscale_final.zip --scenario highway
"""

import pygame
import time
import argparse
import numpy as np
from stable_baselines3 import PPO
from environments.urban_junction_env import UrbanJunctionEnv
from utils.config import get_curriculum_config

def visualize_agent(model_path, scenario="intersection", modality="grayscale", max_steps=200, delay=0.1):
    """
    Visualize a trained agent driving in the specified scenario.

    Args:
        model_path: Path to the trained model
        scenario: Scenario to visualize ("highway", "merge", "intersection")
        modality: Observation modality ("grayscale", "lidar", "both")
        max_steps: Maximum steps per episode
        delay: Delay between frames (seconds)
    """
    print(f"Loading model: {model_path}")
    print(f"Scenario: {scenario}")
    print(f"Modality: {modality}")
    print("-" * 50)

    # Get configuration
    config = get_curriculum_config(scenario, "hard", modality)

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    pygame.display.set_caption(f"Autonomous Driving Agent - {scenario.upper()}")
    print("[OK] Pygame window created (1280x720)")
    print("[OK] Window should be visible now - if not, check if it's behind other windows")
    print("[OK] Press SPACE for new episode, ESC to quit")
    print("[OK] Waiting 3 seconds for you to see the window...")
    time.sleep(3)  # Give user time to see the window

    # Create environment
    env = UrbanJunctionEnv(config=config, scenario=scenario, modality=modality, render_mode='rgb_array')
    print(f"Environment created with obs space: {env.observation_space}")

    # Load model
    model = PPO.load(model_path, env=env)
    print(f"Model loaded successfully")

    # Reset environment
    obs, info = env.reset()
    print("Environment reset - starting visualization...")

    # Episode tracking
    episode_reward = 0
    steps = 0
    running = True
    episode_count = 0

    # Font for displaying info
    font = pygame.font.Font(None, 20)

    try:
        while running and episode_count < 5:  # Run up to 5 episodes
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # Reset for new episode
                        obs, info = env.reset()
                        episode_reward = 0
                        steps = 0
                        episode_count += 1
                        print(f"Starting episode {episode_count + 1}")

            if not running:
                break

            # Get model action
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
            else:
                next_obs, reward, done, info = step_result
                terminated = done
                truncated = False

            episode_reward += reward
            steps += 1

            # Render environment
            frame = env.render()
            if frame is not None and hasattr(frame, 'shape') and len(frame.shape) == 3:
                # Convert RGB frame to pygame surface
                surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                screen.blit(surface, (0, 0))

                # Add text overlay with episode info
                info_lines = [
                    f"Episode: {episode_count + 1}",
                    f"Steps: {steps}",
                    f"Reward: {episode_reward:.2f}",
                    f"Scenario: {scenario.upper()}",
                    f"Press SPACE for new episode",
                    f"Press ESC to quit"
                ]

                y_offset = 10
                for line in info_lines:
                    text_surface = font.render(line, True, (255, 255, 255))
                    screen.blit(text_surface, (10, y_offset))
                    y_offset += 25

                # Check for crash
                if isinstance(info, dict) and info.get("crashed", False):
                    crash_text = font.render("CRASH!", True, (255, 0, 0))
                    screen.blit(crash_text, (600, 350))

                pygame.display.flip()

                # Debug: print frame info occasionally
                if steps % 50 == 0:
                    print(f"Frame {steps}: Agent driving...")
            else:
                # Fallback: just show text if rendering fails
                screen.fill((50, 50, 50))
                error_text = font.render("Rendering not available", True, (255, 255, 255))
                screen.blit(error_text, (400, 300))
                pygame.display.flip()

            # Check for episode end
            done = terminated or truncated or steps >= max_steps
            if done:
                print(f"Episode {episode_count + 1} ended:")
                print(f"  Steps: {steps}")
                print(f"  Reward: {episode_reward:.2f}")
                print(f"  Crashed: {isinstance(info, dict) and info.get('crashed', False)}")

                # Wait a bit before starting new episode
                time.sleep(2)

                # Reset for new episode
                obs, info = env.reset()
                episode_reward = 0
                steps = 0
                episode_count += 1

            # Control frame rate
            time.sleep(delay)

    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")

    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        env.close()
        pygame.quit()
        print("\nVisualization ended")

def main():
    parser = argparse.ArgumentParser(description="Visualize trained agent driving")
    parser.add_argument("--model", required=True, help="Path to trained model (.zip)")
    parser.add_argument("--scenario", default="intersection",
                       choices=["highway", "merge", "intersection"],
                       help="Scenario to visualize")
    parser.add_argument("--modality", default="grayscale",
                       choices=["grayscale", "lidar", "both"],
                       help="Observation modality")
    parser.add_argument("--max-steps", type=int, default=200,
                       help="Maximum steps per episode")
    parser.add_argument("--delay", type=float, default=0.1,
                       help="Delay between frames (seconds)")

    args = parser.parse_args()

    visualize_agent(
        model_path=args.model,
        scenario=args.scenario,
        modality=args.modality,
        max_steps=args.max_steps,
        delay=args.delay
    )

if __name__ == "__main__":
    main()
