#!/usr/bin/env python3
"""
Interactive Autonomous Driving Visualization

This script provides an interactive visualization where you can switch between
different driving scenarios in real-time using keyboard controls.

Controls:
- 1: Highway scenario
- 2: Merge scenario
- 3: Intersection scenario
- SPACE: New episode in current scenario
- ESC: Quit visualization

Usage:
    python interactive_visualization.py --model outputs/models/adaptive_grayscale_final.zip
"""

import pygame
import time
import argparse
import numpy as np
from stable_baselines3 import PPO
from environments.urban_junction_env import UrbanJunctionEnv
from utils.config import get_curriculum_config

class InteractiveVisualizer:
    def __init__(self, model_path, modality="grayscale", max_steps=200, delay=0.1):
        self.model_path = model_path
        self.modality = modality
        self.max_steps = max_steps
        self.delay = delay

        # Scenario configurations
        self.scenarios = {
            'highway': {'name': 'Highway', 'key': pygame.K_1},
            'merge': {'name': 'Merge', 'key': pygame.K_2},
            'intersection': {'name': 'Intersection', 'key': pygame.K_3}
        }

        self.current_scenario = 'highway'
        self.model = None
        self.env = None
        self.screen = None
        self.font = None

        # Episode tracking
        self.episode_count = 0
        self.current_reward = 0
        self.current_steps = 0
        self.current_crashes = 0

    def initialize(self):
        """Initialize pygame and load model"""
        print("Initializing interactive visualization...")

        # Initialize pygame
        pygame.init()
        print("Pygame initialized")
        self.screen = pygame.display.set_mode((1400, 800))
        print("Display created (1400x800)")
        pygame.display.set_caption("Interactive Autonomous Driving Visualization")
        self.font = pygame.font.Font(None, 28)
        print("Font loaded")

        # Test rendering
        self.screen.fill((100, 150, 200))
        test_text = self.font.render("Loading...", True, (255, 255, 255))
        self.screen.blit(test_text, (600, 350))
        pygame.display.flip()
        print("Test render successful")

        # Load model once
        print(f"Loading model: {self.model_path}")
        dummy_env = UrbanJunctionEnv(scenario="highway", modality=self.modality)
        self.model = PPO.load(self.model_path, env=dummy_env)
        dummy_env.close()
        print("Model loaded successfully!")

        # Initialize first scenario
        self.switch_scenario(self.current_scenario)

        print("\n" + "="*70)
        print("INTERACTIVE AUTONOMOUS DRIVING VISUALIZATION")
        print("="*70)
        print("Controls:")
        print("  1: Switch to Highway scenario")
        print("  2: Switch to Merge scenario")
        print("  3: Switch to Intersection scenario")
        print("  SPACE: Start new episode")
        print("  ESC: Quit")
        print("="*70)
        print("Starting with Highway scenario...")
        time.sleep(2)

    def switch_scenario(self, scenario_name):
        """Switch to a different scenario"""
        if scenario_name not in self.scenarios:
            return

        print(f"\nSwitching to {scenario_name.upper()} scenario...")

        # Close current environment
        if self.env:
            self.env.close()

        # Create new environment
        config = get_curriculum_config(scenario_name, "hard", self.modality)
        self.env = UrbanJunctionEnv(config=config, scenario=scenario_name, modality=self.modality, render_mode='rgb_array')

        self.current_scenario = scenario_name

        # Reset episode tracking
        self.episode_count = 0
        self.start_new_episode()

        # Update window title
        scenario_info = self.scenarios[scenario_name]
        pygame.display.set_caption(f"Autonomous Driving - {scenario_info['name']} Scenario")

    def start_new_episode(self):
        """Start a new episode in current scenario"""
        if not self.env:
            return

        self.episode_count += 1
        obs, info = self.env.reset()
        self.current_reward = 0
        self.current_steps = 0
        self.current_crashes = 0
        self.obs = obs

        scenario_info = self.scenarios[self.current_scenario]
        print(f"Started Episode {self.episode_count} in {scenario_info['name']} scenario")

    def update_visualization(self):
        """Update the visualization display"""
        if not self.env or not self.screen:
            return

        # Render environment
        frame = self.env.render()
        if frame is not None and hasattr(frame, 'shape') and len(frame.shape) == 3:
            # Convert RGB frame to pygame surface and scale to fit screen
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

            # Scale to fit left side of screen (leaving room for info panel)
            frame_width, frame_height = 1000, 600
            scaled_surface = pygame.transform.scale(surface, (frame_width, frame_height))

            # Draw frame
            self.screen.blit(scaled_surface, (0, 0))

            # Draw info panel on the right
            self.draw_info_panel(frame_width)

        pygame.display.flip()

    def draw_info_panel(self, frame_width):
        """Draw the information panel"""
        panel_x = frame_width + 20
        panel_width = 350
        panel_height = 600

        # Background
        pygame.draw.rect(self.screen, (50, 50, 70), (panel_x, 0, panel_width, panel_height))
        pygame.draw.rect(self.screen, (100, 100, 120), (panel_x, 0, panel_width, panel_height), 2)

        # Title
        title_text = self.font.render("AUTONOMOUS DRIVING", True, (255, 255, 255))
        self.screen.blit(title_text, (panel_x + 20, 20))

        # Current scenario
        scenario_info = self.scenarios[self.current_scenario]
        scenario_text = self.font.render(f"Scenario: {scenario_info['name']}", True, (200, 255, 200))
        self.screen.blit(scenario_text, (panel_x + 20, 60))

        # Episode info
        episode_text = self.font.render(f"Episode: {self.episode_count}", True, (255, 255, 255))
        self.screen.blit(episode_text, (panel_x + 20, 100))

        steps_text = self.font.render(f"Steps: {self.current_steps}", True, (255, 255, 255))
        self.screen.blit(steps_text, (panel_x + 20, 130))

        reward_text = self.font.render(f"Reward: {self.current_reward:.2f}", True, (255, 255, 255))
        self.screen.blit(reward_text, (panel_x + 20, 160))

        crashes_text = self.font.render(f"Crashes: {self.current_crashes}", True, (255, 0, 0) if self.current_crashes > 0 else (255, 255, 255))
        self.screen.blit(crashes_text, (panel_x + 20, 190))

        # Status
        status_color = (0, 255, 0) if self.current_reward > 0 and self.current_crashes == 0 else (255, 165, 0) if self.current_steps > 0 else (255, 255, 255)
        status_text = "SUCCESS" if (self.current_reward > 0 and self.current_crashes == 0) else "DRIVING" if self.current_steps > 0 else "READY"
        status_surface = self.font.render(f"Status: {status_text}", True, status_color)
        self.screen.blit(status_surface, (panel_x + 20, 230))

        # Controls
        controls_title = self.font.render("CONTROLS:", True, (255, 255, 0))
        self.screen.blit(controls_title, (panel_x + 20, 280))

        controls = [
            "1: Highway",
            "2: Merge",
            "3: Intersection",
            "SPACE: New Episode",
            "ESC: Quit"
        ]

        y_offset = 310
        for control in controls:
            control_text = self.font.render(control, True, (200, 200, 200))
            self.screen.blit(control_text, (panel_x + 40, y_offset))
            y_offset += 25

        # Performance summary
        summary_title = self.font.render("PERFORMANCE:", True, (255, 255, 0))
        self.screen.blit(summary_title, (panel_x + 20, 430))

        # Calculate performance metrics
        if self.episode_count > 0:
            avg_reward = self.current_reward / max(1, self.episode_count)
            crash_rate = self.current_crashes / max(1, self.episode_count)
            summary_text = [
                f"Avg Reward: {avg_reward:.2f}",
                f"Crash Rate: {crash_rate:.2f}",
                f"Total Episodes: {self.episode_count}"
            ]

            y_offset = 460
            for text in summary_text:
                summary_surface = self.font.render(text, True, (200, 200, 200))
                self.screen.blit(summary_surface, (panel_x + 40, y_offset))
                y_offset += 25

    def run_episode(self):
        """Run one episode step"""
        if not self.env or not self.obs is not None:
            return False

        # Get model action
        action, _ = self.model.predict(self.obs, deterministic=True)

        # Step environment
        step_result = self.env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
        else:
            next_obs, reward, done, info = step_result
            terminated = done
            truncated = False

        self.current_reward += reward
        self.current_steps += 1

        # Check for crashes
        if isinstance(info, dict) and info.get("crashed", False):
            self.current_crashes += 1

        self.obs = next_obs

        # Check for episode end
        done = terminated or truncated or self.current_steps >= self.max_steps
        if done:
            success = self.current_reward > 0 and self.current_crashes == 0
            status = "SUCCESS!" if success else "CRASHED"
            print(f"Episode {self.episode_count} ended: {status} "
                  f"(Reward: {self.current_reward:.2f}, Steps: {self.current_steps}, Crashes: {self.current_crashes})")
            return True  # Episode ended

        return False  # Episode continuing

    def run(self):
        """Main visualization loop"""
        self.initialize()

        running = True
        episode_active = False
        last_step_time = time.time()

        try:
            print("Starting main visualization loop...")
            while running:
                current_time = time.time()

                # Handle events
                for event in pygame.event.get():
                    print(f"Event: {event.type}")  # Debug: show events
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_SPACE:
                            # Start new episode
                            if not episode_active:
                                self.start_new_episode()
                                episode_active = True
                        elif event.key in [s['key'] for s in self.scenarios.values()]:
                            # Switch scenario
                            for scenario_name, scenario_info in self.scenarios.items():
                                if event.key == scenario_info['key']:
                                    self.switch_scenario(scenario_name)
                                    episode_active = False
                                    break

                # Run episode step if active and enough time has passed
                if episode_active and (current_time - last_step_time) >= self.delay:
                    episode_ended = self.run_episode()
                    if episode_ended:
                        episode_active = False
                    last_step_time = current_time

                # Update visualization
                self.update_visualization()

                # Small delay to prevent 100% CPU usage
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nVisualization interrupted by user")

        except Exception as e:
            print(f"\nError during visualization: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Clean up
            if self.env:
                self.env.close()
            pygame.quit()
            print("\nInteractive visualization ended")

def main():
    parser = argparse.ArgumentParser(description="Interactive autonomous driving visualization")
    parser.add_argument("--model", required=True, help="Path to trained model (.zip)")
    parser.add_argument("--modality", default="grayscale",
                       choices=["grayscale", "lidar", "both"],
                       help="Observation modality")
    parser.add_argument("--max-steps", type=int, default=200,
                       help="Maximum steps per episode")
    parser.add_argument("--delay", type=float, default=0.1,
                       help="Delay between frames (seconds)")

    args = parser.parse_args()

    visualizer = InteractiveVisualizer(
        model_path=args.model,
        modality=args.modality,
        max_steps=args.max_steps,
        delay=args.delay
    )

    visualizer.run()

if __name__ == "__main__":
    main()
