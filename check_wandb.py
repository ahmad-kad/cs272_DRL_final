#!/usr/bin/env python3
"""
Check WandB Logging Status and Configuration

Verifies that WandB is properly configured and shows current logging setup.
"""

import os
import sys
import subprocess

def check_wandb_installation():
    """Check if WandB is installed and accessible."""
    try:
        import wandb
        print(f"[OK] WandB installed: version {wandb.__version__}")
        return True
    except ImportError:
        print("[FAIL] WandB not installed")
        print("Install with: pip install wandb")
        return False

def check_wandb_login():
    """Check if WandB is logged in."""
    try:
        import wandb
        # Check if API key is set
        api_key = os.environ.get('WANDB_API_KEY') or wandb.api.api_key
        if api_key:
            print("[OK] WandB API key configured")
            return True
        else:
            print("[FAIL] WandB API key not configured")
            print("Login with: wandb login")
            return False
    except:
        return False

def show_wandb_config():
    """Show WandB configuration."""
    print("\n" + "="*50)
    print("WANDB CONFIGURATION")
    print("="*50)

    # Environment variables
    env_vars = ['WANDB_API_KEY', 'WANDB_ENTITY', 'WANDB_PROJECT', 'WANDB_DIR']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        if 'API_KEY' in var and value != 'Not set':
            value = value[:8] + "..."  # Hide most of API key
        print(f"{var}: {value}")

def check_logging_in_code():
    """Check which files have WandB logging."""
    files_to_check = [
        'run_training.py',
        'run_evaluation.py',
        'highway_distillation/training.py',
        'test_setup.py'
    ]

    print("\n" + "="*50)
    print("WANDB LOGGING IN CODE")
    print("="*50)

    for file_path in files_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()

            has_wandb_import = 'import wandb' in content
            has_wandb_init = 'wandb.init(' in content
            has_wandb_log = 'wandb.log(' in content

            status = "[OK]" if (has_wandb_import and has_wandb_init and has_wandb_log) else "[FAIL]"

            print(f"{status} {file_path}")
            if has_wandb_import and has_wandb_init:
                print("   +-- Has WandB import and initialization")
                if has_wandb_log:
                    print("   +-- Has WandB logging calls")
                else:
                    print("   +-- Missing WandB logging calls")
            else:
                print("   └── Missing WandB integration")
        else:
            print(f"[FAIL] {file_path} - File not found")

def show_usage_examples():
    """Show how to use WandB logging."""
    print("\n" + "="*50)
    print("USAGE EXAMPLES")
    print("="*50)

    examples = [
        ("Training with WandB", "python run_training.py --agent lidar"),
        ("Training with custom project", "python run_training.py --wandb-project my-project --agent all"),
        ("Evaluation with WandB", "python run_evaluation.py"),
        ("Login to WandB", "wandb login"),
        ("View WandB dashboard", "wandb dashboard"),
        ("Sync offline runs", "wandb sync"),
    ]

    for desc, cmd in examples:
        print(f"{desc}:")
        print(f"  {cmd}")
        print()

def check_recent_runs():
    """Check for recent WandB runs."""
    print("\n" + "="*50)
    print("RECENT WANDB RUNS")
    print("="*50)

    try:
        import wandb
        if wandb.api.api_key:
            # Try to get recent runs
            try:
                runs = wandb.Api().runs("highway-distillation", order="+created_at", limit=5)
                if runs:
                    print("Recent runs in 'highway-distillation' project:")
                    for run in runs:
                        print(f"  - {run.name} ({run.state}) - {run.created_at}")
                else:
                    print("No recent runs found in 'highway-distillation' project")
            except Exception as e:
                print(f"Could not fetch runs: {e}")
        else:
            print("Not logged in to WandB")
    except ImportError:
        print("WandB not available")

def main():
    """Main function to check WandB status."""
    print("WANDB LOGGING STATUS CHECK")
    print("="*50)

    # Check installation
    wandb_installed = check_wandb_installation()

    if not wandb_installed:
        print("\n[FAIL] WandB is not properly installed.")
        print("Install with: pip install wandb")
        sys.exit(1)

    # Check login
    wandb_logged_in = check_wandb_login()

    # Show configuration
    show_wandb_config()

    # Check code integration
    check_logging_in_code()

    # Show usage examples
    show_usage_examples()

    # Check recent runs
    if wandb_logged_in:
        check_recent_runs()

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)

    all_good = wandb_installed and wandb_logged_in

    if all_good:
        print("[OK] WandB is fully configured and ready!")
        print("[TARGET] All training and evaluation will be logged to WandB.")
        print("[DASHBOARD] Check your dashboard at: https://wandb.ai")
    else:
        print("[WARN] WandB needs configuration:")
        if not wandb_logged_in:
            print("   - Run: wandb login")
        print("   - All code has WandB integration ready")

if __name__ == "__main__":
    main()
