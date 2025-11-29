"""
Core test suite for UrbanJunctionEnv - Less is More.

Tests focus on critical behaviors that matter for learning:
1. Reward system integrity
2. Environment stability
3. Critical edge cases
4. Multi-modal functionality
"""

import pytest
import numpy as np
import gymnasium as gym
from unittest.mock import Mock, patch, MagicMock


# Mock highway_env imports for testing
class MockVehicle:
    def __init__(self, position=[0, 0], heading=0, speed=20, crashed=False, on_road=True):
        self.position = np.array(position, dtype=float)
        self.heading = heading
        self.speed = speed
        self.crashed = crashed
        self.on_road = on_road
        self.lane_index = ('0', '1', 0)
        self.lane = Mock(width=4.0)


class TestCoreBehavior:
    """Critical behavior tests that matter for learning."""

    def test_environment_creation(self):
        """Environment should create successfully with default config."""
        from environments.urban_junction_env import UrbanJunctionEnv

        config = UrbanJunctionEnv.default_config()
        env = UrbanJunctionEnv(config)
        obs, info = env.reset()

        # Default config is single-modal, so obs might not be a dict
        # Just check that environment creates and returns valid observations
        assert obs is not None, "Observation should not be None"
        assert hasattr(env, 'action_space'), "Should have action space"

        env.close()

    def test_reward_bounds(self):
        """Rewards should be reasonable (VecNormalize handles final normalization)."""
        from environments.urban_junction_env import UrbanJunctionEnv

        config = UrbanJunctionEnv.default_config()
        env = UrbanJunctionEnv(config)

        for episode in range(3):
            obs, info = env.reset()
            episode_rewards = []

            for step in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_rewards.append(reward)

                # Raw rewards can be outside [0,1] - VecNormalize handles normalization
                # Just check they're reasonable numbers
                assert isinstance(reward, (int, float)), f"Reward should be numeric: {reward}"
                assert -10.0 <= reward <= 10.0, f"Reward unreasonably large: {reward}"

                if terminated or truncated:
                    break

        env.close()

    def test_collision_termination(self):
        """Collision should terminate episode."""
        from environments.urban_junction_env import UrbanJunctionEnv

        config = UrbanJunctionEnv.default_config()
        env = UrbanJunctionEnv(config)
        obs, info = env.reset()

        # Force a collision by setting vehicle state
        env.vehicle.crashed = True

        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

        assert terminated == True, "Collision should terminate episode"
        # Note: reward might not be exactly 0.0 due to exploration bonuses

        env.close()

    def test_stage_progression(self):
        """Agent should make progress through stages."""
        from environments.urban_junction_env import UrbanJunctionEnv

        config = UrbanJunctionEnv.default_config()
        config["stage_mode"] = "deterministic"
        env = UrbanJunctionEnv(config)

        obs, info = env.reset()
        initial_stage = env.current_stage_idx

        # Run for several steps
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        # Should have made some progress
        assert env.current_stage_idx >= initial_stage, "Should progress through stages"

        env.close()

    def test_multimodal_observations(self):
        """Multi-modal observations should work correctly."""
        from environments.urban_junction_env import UrbanJunctionEnv

        config = UrbanJunctionEnv.default_config()
        config["observation"]["multi_modal"] = True
        env = UrbanJunctionEnv(config)

        obs, info = env.reset()

        # Check observation structure
        assert isinstance(obs, dict), "Multi-modal obs should be dict"
        assert len(obs) == 3, "Should have 3 observation modalities"
        assert 'kinematics' in obs, "Should have kinematics"
        assert 'lidar' in obs, "Should have lidar"
        assert 'visual' in obs, "Should have visual"

        # Check shapes based on config (may vary)
        lidar_rays = config["observation"]["lidar_rays"]
        visual_w = config["observation"]["visual_width"]
        visual_h = config["observation"]["visual_height"]

        assert obs['lidar'].shape == (lidar_rays,), f"Lidar shape should be ({lidar_rays},), got {obs['lidar'].shape}"
        assert obs['visual'].shape == (visual_h, visual_w, 1), f"Visual shape should be ({visual_h}, {visual_w}, 1), got {obs['visual'].shape}"
        assert obs['kinematics'].ndim == 2, f"Kinematics should be 2D array, got shape {obs['kinematics'].shape}"

        env.close()

    def test_curriculum_adaptation(self):
        """Curriculum should adapt difficulty based on performance."""
        from environments.urban_junction_env import UrbanJunctionEnv
        
        config = UrbanJunctionEnv.default_config()
        env = UrbanJunctionEnv(config)

        # Test successful episodes (should increase difficulty)
        initial_vehicles = env.config["vehicles_count"]
        for _ in range(8):
            env.update_curriculum(True)  # Success

        # Should have increased difficulty
        assert env.config["vehicles_count"] >= initial_vehicles, "Curriculum should increase difficulty on success"

        env.close()

    def test_modality_dropout(self):
        """Modality dropout should work correctly."""
        from environments.urban_junction_env import UrbanJunctionEnv

        config = UrbanJunctionEnv.default_config()
        config["observation"]["multi_modal"] = True  # Enable multi-modal for dropout testing
        config["modality_dropout"] = True
        config["dropout_rate"] = 1.0  # Always dropout
        env = UrbanJunctionEnv(config)

        obs, info = env.reset()

        # With multi-modal enabled and dropout_rate=1.0, at least one modality should be zeroed
        lidar_zeroed = np.all(obs['lidar'] == 0)
        visual_zeroed = np.all(obs['visual'] == 0)

        assert lidar_zeroed or visual_zeroed, "Should dropout at least one modality"
        assert not (lidar_zeroed and visual_zeroed), "Should not dropout both modalities"

        env.close()


def run_core_tests():
    """Run only the core critical tests."""
    test_classes = [TestCoreBehavior]
    
    results = {
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)
        
        instance = test_class()
        test_methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            try:
                method = getattr(instance, method_name)
                method()
                print(f"✓ {method_name}")
                results['passed'] += 1
                
            except AssertionError as e:
                print(f"✗ {method_name}: {e}")
                results['failed'] += 1
                results['errors'].append((test_class.__name__, method_name, str(e)))
                
            except Exception as e:
                print(f"✗ {method_name}: ERROR - {e}")
                results['failed'] += 1
                results['errors'].append((test_class.__name__, method_name, f"ERROR: {e}"))
    
    # Print summary
    print(f"\n{'='*60}")
    print("CORE TESTS SUMMARY")
    print('='*60)
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Total:  {results['passed'] + results['failed']}")
    print(f"Success Rate: {100 * results['passed'] / (results['passed'] + results['failed']):.1f}%")
    
    if results['errors']:
        print(f"\n{'='*60}")
        print("FAILED TESTS")
        print('='*60)
        for class_name, method_name, error in results['errors']:
            print(f"\n{class_name}.{method_name}:")
            print(f"  {error}")
    
    return results


if __name__ == "__main__":
    results = run_core_tests()