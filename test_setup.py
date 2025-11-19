#!/usr/bin/env python3
"""
Quick Validation Test - Verifies setup before production training.
"""

import sys
import os
import importlib.util

sys.path.insert(0, os.path.dirname(__file__))

def import_training_module():
    """Import training module directly."""
    module_path = os.path.join(os.path.dirname(__file__), "highway_distillation", "training.py")
    spec = importlib.util.spec_from_file_location("training_module", module_path)
    training_module = importlib.util.module_from_spec(spec)
    sys.modules['training_module'] = training_module
    spec.loader.exec_module(training_module)
    return training_module

def quick_test(agent_type: str, test_steps: int = 2000):
    """Quick validation for agent type."""
    print(f"\n{'='*60}")
    print(f"VALIDATION TEST: {agent_type.upper()} Agent")
    print(f"{'='*60}")
    print(f"Test steps: {test_steps}")
    
    try:
        training_module = import_training_module()
        create_environment = training_module.create_environment
        create_model = training_module.create_model
        
        use_lidar = (agent_type == 'lidar')
        use_gray = (agent_type == 'grayscale')
        
        print("[1/3] Creating environment...")
        env = create_environment(
            use_grayscale_only=use_gray,
            use_lidar_only=use_lidar,
            num_envs=2,
            use_subprocess=False
        )
        print("  Environment OK")
        
        print("[2/3] Creating model...")
        model = create_model(env, use_grayscale_only=use_gray, use_lidar_only=use_lidar)
        print("  Model OK")
        
        print(f"[3/3] Training {test_steps} steps...")
        model.learn(total_timesteps=test_steps, progress_bar=True)
        print("  Training OK")
        
        env.close()
        
        print(f"\n{'='*60}")
        print(f"VALIDATION PASSED: {agent_type.upper()}")
        print(f"{'='*60}")
        return True
        
    except Exception as e:
        print(f"\nVALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, choices=['lidar', 'grayscale', 'all'], default='lidar')
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("QUICK VALIDATION TEST")
    print("="*60)
    
    if args.agent == 'all':
        lidar_ok = quick_test('lidar', args.steps)
        gray_ok = quick_test('grayscale', args.steps)
        success = lidar_ok and gray_ok
    else:
        success = quick_test(args.agent, args.steps)
    
    if success:
        print("\nREADY FOR PRODUCTION TRAINING")
        print("Run: python run_training.py")
    else:
        print("\nFIX ISSUES BEFORE TRAINING")
    
    sys.exit(0 if success else 1)

