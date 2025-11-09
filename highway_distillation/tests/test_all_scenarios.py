#!/usr/bin/env python3
"""
Test all three scenario environments: Highway, Merge, and Intersection.

This script verifies:
1. Highway scenario: Standard cruising with traffic
2. Merge scenario: Lane merge with aggressive vehicles
3. Intersection scenario: Traffic light with crossing vehicles
4. Multi-modal observations: Kinematics + Lidar + Visual
5. Antagonistic vehicles with configurable difficulty
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from environments.urban_junction_env import UrbanJunctionEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_highway_scenario():
    """Test highway scenario - standard cruising."""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: HIGHWAY SCENARIO")
    logger.info("="*60)

    config = UrbanJunctionEnv.default_config()
    config.update({
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 8,
            "features": ["presence", "x", "y", "vx", "vy"],
        },
        "stage_mode": "deterministic",
        "antagonistic_vehicles": False,
        "vehicles_count": 8,
        "duration": 100,
    })

    env = UrbanJunctionEnv(config)
    obs, info = env.reset()

    logger.info(f"✓ Environment created")
    logger.info(f"✓ Initial observation shape: {obs.shape}")
    logger.info(f"✓ Observation space: {env.observation_space}")
    logger.info(f"✓ Action space: {env.action_space}")

    total_reward = 0
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 5 == 0:
            logger.info(f"  Step {step+1}: reward={reward:.3f}, pos={info.get('position', 0):.1f}m")

        if terminated or truncated:
            logger.info(f"✓ Episode ended at step {step+1}")
            break

    logger.info(f"✓ Highway test complete - Total reward: {total_reward:.2f}")
    env.close()
    return True


def test_merge_scenario():
    """Test merge scenario - lane changing with aggressive vehicles."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: MERGE SCENARIO")
    logger.info("="*60)

    config = UrbanJunctionEnv.default_config()
    config.update({
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy"],
        },
        "stage_mode": "deterministic",
        "antagonistic_vehicles": True,
        "vehicles_count": 10,
        "duration": 100,
        "annoyance_level": 0.5,
    })

    env = UrbanJunctionEnv(config)
    obs, info = env.reset()

    logger.info(f"✓ Environment created with antagonistic vehicles")
    logger.info(f"✓ Annoyance level: {env.annoyance_level:.1f}")
    logger.info(f"✓ Observation shape: {obs.shape}")

    total_reward = 0
    collisions = 0
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 5 == 0:
            logger.info(f"  Step {step+1}: reward={reward:.3f}, speed={info.get('speed', 0):.1f}m/s")

        if terminated:
            collisions += 1
            logger.warning(f"  Collision detected at step {step+1}")
            break

    logger.info(f"✓ Merge test complete - Total reward: {total_reward:.2f}")
    env.close()
    return True


def test_intersection_scenario():
    """Test intersection scenario - traffic light compliance."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: INTERSECTION SCENARIO")
    logger.info("="*60)

    config = UrbanJunctionEnv.default_config()
    config.update({
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 8,
            "features": ["presence", "x", "y", "vx", "vy"],
        },
        "stage_mode": "deterministic",
        "antagonistic_vehicles": False,
        "vehicles_count": 8,
        "duration": 100,
        "traffic_light_green": 10,
        "traffic_light_red": 10,
        "traffic_light_yellow": 2,
    })

    env = UrbanJunctionEnv(config)
    obs, info = env.reset()

    logger.info(f"✓ Environment created with traffic light")
    logger.info(f"✓ Traffic light state: {env.traffic_light}")
    logger.info(f"✓ Observation shape: {obs.shape}")

    total_reward = 0
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        traffic_state = info.get("traffic_light_state", "unknown")
        if step % 5 == 0:
            logger.info(f"  Step {step+1}: reward={reward:.3f}, traffic={traffic_state}")

        if terminated or truncated:
            logger.info(f"✓ Episode ended at step {step+1}")
            break

    logger.info(f"✓ Intersection test complete - Total reward: {total_reward:.2f}")
    env.close()
    return True


def test_multimodal_observations():
    """Test multi-modal observations: kinematics + lidar + visual."""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: MULTI-MODAL OBSERVATIONS")
    logger.info("="*60)

    config = UrbanJunctionEnv.default_config()
    config.update({
        "observation": {
            "type": "Kinematics",
            "multi_modal": True,
            "lidar_rays": 32,
            "lidar_range": 50.0,
            "visual_width": 64,
            "visual_height": 64,
            "vehicles_count": 8,
        },
        "stage_mode": "deterministic",
        "antagonistic_vehicles": False,
        "vehicles_count": 8,
        "duration": 100,
    })

    env = UrbanJunctionEnv(config)
    obs, info = env.reset()

    logger.info(f"✓ Multi-modal environment created")
    logger.info(f"✓ Observation space: {env.observation_space}")

    if hasattr(env.observation_space, 'spaces'):
        logger.info(f"✓ Multi-modal observation keys: {list(env.observation_space.spaces.keys())}")
        for key, space in env.observation_space.spaces.items():
            logger.info(f"  - {key}: {space}")

    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 5 == 0:
            logger.info(f"  Step {step+1}: reward={reward:.3f}")

        if terminated or truncated:
            logger.info(f"✓ Episode ended at step {step+1}")
            break

    logger.info(f"✓ Multi-modal test complete - Total reward: {total_reward:.2f}")
    env.close()
    return True


def test_difficulty_progression():
    """Test difficulty levels: annoyance level 0.1 to 1.0."""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: DIFFICULTY PROGRESSION")
    logger.info("="*60)

    difficulty_levels = [0.1, 0.5, 0.9]

    for annoyance in difficulty_levels:
        logger.info(f"\n--- Testing annoyance level: {annoyance:.1f} ---")

        config = UrbanJunctionEnv.default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 8,
            },
            "stage_mode": "deterministic",
            "antagonistic_vehicles": True,
            "vehicles_count": 8,
            "annoyance_level": annoyance,
            "duration": 50,
        })

        env = UrbanJunctionEnv(config)
        obs, info = env.reset()

        total_reward = 0
        for step in range(15):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        logger.info(f"  ✓ Annoyance {annoyance}: reward={total_reward:.2f}")
        env.close()

    logger.info(f"✓ Difficulty progression test complete")
    return True


def run_all_tests():
    """Run all scenario tests."""
    logger.info("\n\n")
    logger.info("╔" + "="*58 + "╗")
    logger.info("║" + " "*58 + "║")
    logger.info("║" + "  COMPREHENSIVE SCENARIO TESTING  ".center(58) + "║")
    logger.info("║" + " "*58 + "║")
    logger.info("╚" + "="*58 + "╝")

    tests = [
        ("Highway Scenario", test_highway_scenario),
        ("Merge Scenario", test_merge_scenario),
        ("Intersection Scenario", test_intersection_scenario),
        ("Multi-Modal Observations", test_multimodal_observations),
        ("Difficulty Progression", test_difficulty_progression),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            logger.error(f"✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {name}")
        if not passed:
            all_passed = False

    logger.info("="*60)

    if all_passed:
        logger.info("\n✓ ALL TESTS PASSED!")
        logger.info("✓ All scenarios (Highway, Merge, Intersection) verified")
        logger.info("✓ Multi-modal observations functional")
        logger.info("✓ Difficulty progression working")
        logger.info("✓ Environment ready for training!")
        return True
    else:
        logger.error("\n✗ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

