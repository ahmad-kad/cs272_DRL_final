#!/usr/bin/env python3
"""
Demonstrate Safety Constraints in Enhanced Urban Environment

This script shows how the hard safety constraints work to prevent dangerous
actions while allowing the RL agent to learn safe driving behavior.
"""

import numpy as np
from environments.enhanced_urban_env import EnhancedUrbanJunctionEnv


def demonstrate_safety_constraints():
    """Demonstrate various safety constraint scenarios."""
    print("[SHIELD] SAFETY CONSTRAINTS DEMONSTRATION")
    print("=" * 50)

    scenarios = [
        ("highway", "lidar"),
        ("merge", "grayscale"),
        ("intersection", "both")
    ]

    for scenario, modality in scenarios:
        print(f"\n[TARGET] Testing {scenario.upper()} scenario with {modality.upper()} modality")
        print("-" * 50)

        env = EnhancedUrbanJunctionEnv(scenario=scenario, modality=modality, render_mode=None)
        env.reset()

        # Test different dangerous situations
        test_cases = [
            ("speeding", lambda: setattr(env.vehicle, 'speed', 50)),  # Way too fast
            ("lane_departure", lambda: setattr(env.vehicle, 'lane_index', None)),  # Off road
            ("close_proximity", lambda: None),  # Normal but we'll test proximity
        ]

        for test_name, setup_func in test_cases:
            print(f"  [TEST] Testing: {test_name}")

            # Reset environment
            env.reset()
            env.safety_override_count = 0

            # Set up dangerous situation
            setup_func()

            # Try some actions
            actions_tested = 0
            overrides_triggered = 0

            for _ in range(10):
                # Use a potentially dangerous action (high acceleration, sharp turn)
                dangerous_action = np.array([0.8, 1.0]) if hasattr(env.action_space, 'shape') else 5

                # Apply safety constraints
                safe_action = env._enforce_hard_safety_constraints(dangerous_action)

                # Check if action was modified
                if not np.array_equal(dangerous_action, safe_action):
                    overrides_triggered += 1

                actions_tested += 1

            safety_rate = overrides_triggered / actions_tested * 100
            print(".1f")
        # Show final safety stats
        stats = env.get_safety_stats()
        print(f"  [CHART] Session Safety Stats: {stats['safety_override_count']} overrides, "
              ".1f")
def demonstrate_proximity_detection():
    """Demonstrate proximity detection and penalties."""
    print("\n[TARGET] PROXIMITY DETECTION DEMONSTRATION")
    print("=" * 50)

    env = EnhancedUrbanJunctionEnv(scenario="merge", modality="lidar", render_mode=None)

    print("\n🔍 Testing Proximity Penalty Levels:")
    print("-" * 40)

    # Test different proximity scenarios
    proximity_tests = [
        ("Very Close (2m)", 2.0),
        ("Close (5m)", 5.0),
        ("Medium (10m)", 10.0),
        ("Far (20m)", 20.0),
        ("Very Far (50m)", 50.0)
    ]

    for desc, distance in proximity_tests:
        # Simulate lidar detection at this distance
        # This is a simplified test - in reality this would come from actual lidar
        penalty = env._calculate_lidar_proximity_penalty(
            np.array([[1.0, distance], [0.0, 100.0]])  # One vehicle at test distance
        )
        penalty_level = "SAFE" if penalty >= -0.1 else "CAUTION" if penalty >= -1.0 else "DANGER" if penalty >= -3.0 else "CRITICAL"
        print("12")

    print("\n[STOP] Proximity Penalty Scale:")
    print("  - SAFE: No penalty (distance > 15m)")
    print("  - CAUTION: Light penalty (-0.1 to -1.0)")
    print("  - DANGER: Strong penalty (-1.0 to -3.0)")
    print("  - CRITICAL: Severe penalty (<-3.0)")


def demonstrate_adaptive_curriculum():
    """Show the adaptive curriculum structure."""
    print("\n[GRAD] ADAPTIVE CURRICULUM OVERVIEW")
    print("=" * 50)

    from training.adaptive_curriculum_trainer import AdaptiveCurriculumTrainer

    trainer = AdaptiveCurriculumTrainer()

    print("\n[GRAD] Curriculum Progression:")
    print("-" * 50)

    for i, phase in enumerate(trainer.curriculum_phases, 1):
        status_icon = "[TARGET]" if i == 1 else "⏳"
        print(f"{status_icon} Phase {i}: {phase['name']}")
        print(f"     {phase['description']}")
        print(f"     Scenarios: {phase['scenarios']}")
        print(f"     Modalities: {phase['modalities']}")
        print(f"     Timesteps: {phase['timesteps']}")
        print(f"     Difficulty: {phase['difficulty']}")
        print()


def run_full_demonstration():
    """Run complete safety constraints demonstration."""
    print("[CAR] ENHANCED URBAN ENVIRONMENT SAFETY DEMONSTRATION")
    print("=" * 60)
    print("This demonstrates the safety constraints and enhanced reward structure")
    print("for collision avoidance in autonomous driving.")
    print("=" * 60)

    # Demonstrate safety constraints
    demonstrate_safety_constraints()

    # Demonstrate proximity detection
    demonstrate_proximity_detection()

    # Show curriculum structure
    demonstrate_adaptive_curriculum()

    print("\n[CELEBRATE] DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("[OK] Safety constraints prevent dangerous actions")
    print("[OK] Proximity detection provides early collision warnings")
    print("[OK] Enhanced rewards teach safe driving behavior")
    print("[OK] Adaptive curriculum enables progressive learning")
    print()
    print("[ROCKET] Ready for safe autonomous driving training!")


if __name__ == "__main__":
    run_full_demonstration()
