#!/usr/bin/env python3
"""
Quick Validation Test - Verifies setup before production training.
"""

import sys
import os
import importlib.util
import time
import wandb

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

    test_start_time = time.time()

    try:
        training_module = import_training_module()
        create_environment = training_module.create_environment
        create_model = training_module.create_model

        use_lidar = (agent_type == 'lidar')
        use_gray = (agent_type == 'grayscale')

        print("[1/3] Creating environment...")
        env_creation_start = time.time()
        env = create_environment(
            use_grayscale_only=use_gray,
            use_lidar_only=use_lidar,
            num_envs=2,
            use_subprocess=False
        )
        env_creation_time = time.time() - env_creation_start
        print(f"  Environment OK (created in {env_creation_time:.2f}s)")

        print("[2/3] Creating model...")
        model_creation_start = time.time()
        model = create_model(env, use_grayscale_only=use_gray, use_lidar_only=use_lidar)
        model_creation_time = time.time() - model_creation_start
        print(f"  Model OK (created in {model_creation_time:.2f}s)")

        print(f"[3/3] Training {test_steps} steps...")
        training_start = time.time()
        model.learn(total_timesteps=test_steps, progress_bar=True)
        training_time = time.time() - training_start
        print(f"  Training OK (completed in {training_time:.2f}s)")

        env.close()

        total_test_time = time.time() - test_start_time

        # Log to WandB
        wandb.log({
            f"validation_{agent_type}_completed": True,
            f"validation_{agent_type}_test_steps": test_steps,
            f"validation_{agent_type}_env_creation_time": env_creation_time,
            f"validation_{agent_type}_model_creation_time": model_creation_time,
            f"validation_{agent_type}_training_time": training_time,
            f"validation_{agent_type}_total_time": total_test_time,
            f"validation_{agent_type}_steps_per_second": test_steps / training_time if training_time > 0 else 0,
            f"validation_{agent_type}_timestamp": time.time()
        })

        print(f"\n{'='*60}")
        print(f"VALIDATION PASSED: {agent_type.upper()}")
        print(f"{'='*60}")
        return True

    except Exception as e:
        test_time = time.time() - test_start_time
        print(f"\nVALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()

        # Log failure to WandB
        wandb.log({
            f"validation_{agent_type}_failed": True,
            f"validation_{agent_type}_error": str(e),
            f"validation_{agent_type}_test_time": test_time,
            f"validation_{agent_type}_timestamp": time.time()
        })

        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, choices=['lidar', 'grayscale', 'all'], default='lidar')
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--wandb-project", type=str, default="highway-distillation",
                      help="WandB project name")
    args = parser.parse_args()

    # Initialize WandB for validation testing
    wandb_config = {
        "test_type": "validation_test",
        "agents_tested": args.agent,
        "test_steps": args.steps,
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S")
    }

    wandb.init(
        project=args.wandb_project,
        name=f"validation_test_{args.agent}_{int(time.time())}",
        config=wandb_config,
        notes=f"Validation testing for {args.agent} agent(s) with {args.steps} steps"
    )

    print("\n" + "="*60)
    print("QUICK VALIDATION TEST")
    print("="*60)
    print(f"WandB Project: {args.wandb_project}")
    print("="*60)

    validation_start_time = time.time()

    if args.agent == 'all':
        lidar_ok = quick_test('lidar', args.steps)
        gray_ok = quick_test('grayscale', args.steps)
        success = lidar_ok and gray_ok
    else:
        success = quick_test(args.agent, args.steps)

    validation_total_time = time.time() - validation_start_time

    # Log final validation results
    final_metrics = {
        "validation_completed": True,
        "validation_success": success,
        "validation_total_time": validation_total_time,
        "agents_tested": args.agent,
        "lidar_test_passed": lidar_ok if args.agent == 'all' else (success if args.agent == 'lidar' else None),
        "grayscale_test_passed": gray_ok if args.agent == 'all' else (success if args.agent == 'grayscale' else None),
        "validation_timestamp": time.time()
    }

    wandb.log(final_metrics)

    if success:
        print("\nREADY FOR PRODUCTION TRAINING")
        print("Run: python run_training.py")
        print(f"WandB Dashboard: https://wandb.ai/{wandb.run.entity}/{args.wandb_project}")
    else:
        print("\nFIX ISSUES BEFORE TRAINING")

    sys.exit(0 if success else 1)

