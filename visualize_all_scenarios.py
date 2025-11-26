#!/usr/bin/env python3
"""
Visualize Agent in All Driving Scenarios

This script runs the trained agent through all available driving scenarios:
- Highway: Multi-lane highway with traffic
- Merge: Highway with on-ramp merging
- Intersection: Urban intersection with traffic lights
- Custom: User-defined scenario

Usage:
    python visualize_all_scenarios.py --model outputs/models/adaptive_grayscale_final.zip
    python visualize_all_scenarios.py --model outputs/models/adaptive_grayscale_final.zip --cycle
"""

import pygame
import time
import argparse
import numpy as np
from stable_baselines3 import PPO
from environments.urban_junction_env import UrbanJunctionEnv
from utils.config import get_curriculum_config

def visualize_scenario(model, scenario, modality="grayscale", max_steps=100, delay=0.1, single_episode=True):
    """
    Visualize agent in a specific scenario.

    Args:
        model: Loaded PPO model
        scenario: Scenario name
        modality: Observation modality
        max_steps: Maximum steps per episode
        delay: Delay between frames

    Returns:
        dict: Episode results
    """
    print(f"\n{'='*60}")
    print(f"[SCENARIO] VISUALIZING: {scenario.upper()} SCENARIO")
    print(f"{'='*60}")

    # Get configuration
    config = get_curriculum_config(scenario, "hard", modality)

    # Create environment
    env = UrbanJunctionEnv(config=config, scenario=scenario, modality=modality, render_mode='rgb_array')

    # Font for displaying info
    font = pygame.font.Font(None, 24)

    # Reset environment
    obs, info = env.reset()
    episode_reward = 0
    steps = 0
    crashes = 0

    print(f"Starting {scenario} scenario...")
    print("Controls: SPACE=new episode, ESC=quit, X=close window")

    try:
        running = True
        episode_count = 0
        max_episodes = 1 if single_episode else 5

        while running and episode_count < max_episodes:
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
                        crashes = 0
                        episode_count += 1
                        print(f"New episode started in {scenario} (episode {episode_count})")

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

            # Check for crashes
            if isinstance(info, dict) and info.get("crashed", False):
                crashes += 1

            # Render environment
            frame = env.render()
            if frame is not None and hasattr(frame, 'shape') and len(frame.shape) == 3:
                # Convert RGB frame to pygame surface
                surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                screen.blit(surface, (0, 0))

                # Add text overlay with episode info
                info_lines = [
                    f"SCENARIO: {scenario.upper()}",
                    f"Episode Reward: {episode_reward:.2f}",
                    f"Steps: {steps}",
                    f"Crashes: {crashes}",
                    f"Modality: {modality}",
                    f"Controls: SPACE=new episode, ESC=quit"
                ]

                y_offset = 10
                for line in info_lines:
                    text_surface = font.render(line, True, (255, 255, 255))
                    screen.blit(text_surface, (10, y_offset))
                    y_offset += 25

                # Show crash indicator
                if crashes > 0:
                    crash_text = font.render("CRASH DETECTED!", True, (255, 0, 0))
                    screen.blit(crash_text, (600, 350))

                pygame.display.flip()
            else:
                # Fallback: just show text if rendering fails
                screen.fill((50, 50, 50))
                error_text = font.render(f"Rendering failed - {scenario}", True, (255, 255, 255))
                screen.blit(error_text, (400, 300))
                pygame.display.flip()

            # Check for episode end
            done = terminated or truncated or steps >= max_steps
            if done:
                success = episode_reward > 0 and crashes == 0
                status = "SUCCESS" if success else "FAILED"

                print(f"Episode completed in {scenario}:")
                print(".2f")
                print(f"  Steps: {steps}")
                print(f"  Crashes: {crashes}")
                print(f"  Status: {status}")

                # Wait before starting new episode
                time.sleep(2)

                # Reset for new episode
                obs, info = env.reset()
                episode_reward = 0
                steps = 0
                crashes = 0

            # Control frame rate
            time.sleep(delay)

    except KeyboardInterrupt:
        print(f"\n{scenario} visualization interrupted by user")

    except Exception as e:
        print(f"\nError in {scenario} visualization: {e}")

    finally:
        env.close()
        # Don't quit pygame here - let the main function handle it

    return {
        "scenario": scenario,
        "episodes_completed": 1,
        "total_reward": episode_reward,
        "total_steps": steps,
        "total_crashes": crashes
    }

def run_all_scenarios(model_path, modality="grayscale", cycle=False, max_steps=100, delay=0.1):
    """
    Run visualization for all scenarios.

    Args:
        model_path: Path to trained model
        modality: Observation modality
        cycle: Whether to cycle through scenarios automatically
        max_steps: Maximum steps per episode
        delay: Delay between frames
    """
    # Define all scenarios
    scenarios = ["highway", "merge", "intersection"]

    print("=" * 80)
    print("[CAR] AUTONOMOUS DRIVING VISUALIZATION - ALL SCENARIOS")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Modality: {modality}")
    print(f"Mode: {'Cycling' if cycle else 'Sequential'}")
    print()

    # Initialize pygame
    pygame.init()
    global screen
    screen = pygame.display.set_mode((1280, 720))
    pygame.display.set_caption("Autonomous Driving - All Scenarios")

    # Load model once
    print("Loading model...")
    dummy_env = UrbanJunctionEnv(scenario="highway", modality=modality)
    model = PPO.load(model_path, env=dummy_env)
    dummy_env.close()
    print("Model loaded successfully!")

    try:
        if cycle:
            # Cycle through scenarios automatically
            print("Starting cycling mode - press ESC to quit")
            scenario_idx = 0

            while True:
                current_scenario = scenarios[scenario_idx % len(scenarios)]

                # Update window title
                pygame.display.set_caption(f"Autonomous Driving - {current_scenario.upper()}")

                # Run scenario
                result = visualize_scenario(model, current_scenario, modality, max_steps, delay)

                scenario_idx += 1

                # Check for quit event during scenario transition
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        raise KeyboardInterrupt

        else:
            # Run scenarios sequentially
            results = {}
            for scenario in scenarios:
                print(f"\nPreparing {scenario.upper()} scenario...")
                time.sleep(1)  # Brief pause between scenarios

                # Update window title
                pygame.display.set_caption(f"Autonomous Driving - {scenario.upper()}")

                # Run scenario (single episode per scenario)
                result = visualize_scenario(model, scenario, modality, max_steps, delay, single_episode=True)
                results[scenario] = result

                # Check if user wants to quit (only if pygame is still initialized)
                try:
                    quit_requested = False
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            quit_requested = True
                    if quit_requested:
                        break
                except pygame.error:
                    # Pygame not initialized, continue
                    break

            # Print summary
            print("\n" + "="*80)
            print("VISUALIZATION SUMMARY")
            print("="*80)
            for scenario, result in results.items():
                success_rate = "SUCCESS" if result['total_reward'] > 0 and result['total_crashes'] == 0 else "FAILED"
                print(f"{scenario.upper()}: {result['total_steps']} steps, {result['total_reward']:.2f} reward, {result['total_crashes']} crashes - {success_rate}")

    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")

    finally:
        pygame.quit()
        print("\nVisualization ended")

def main():
    parser = argparse.ArgumentParser(description="Visualize agent in all driving scenarios")
    parser.add_argument("--model", required=True, help="Path to trained model (.zip)")
    parser.add_argument("--modality", default="grayscale",
                       choices=["grayscale", "lidar", "both"],
                       help="Observation modality")
    parser.add_argument("--cycle", action="store_true",
                       help="Cycle through scenarios automatically")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="Maximum steps per episode")
    parser.add_argument("--delay", type=float, default=0.1,
                       help="Delay between frames (seconds)")

    args = parser.parse_args()

    run_all_scenarios(
        model_path=args.model,
        modality=args.modality,
        cycle=args.cycle,
        max_steps=args.max_steps,
        delay=args.delay
    )

if __name__ == "__main__":
    main()
