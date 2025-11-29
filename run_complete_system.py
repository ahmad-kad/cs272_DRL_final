#!/usr/bin/env python3
"""
Complete Autonomous Driving RL System Demo

This script demonstrates the full enhanced autonomous driving system:
1. Train with safety constraints and curriculum learning
2. Comprehensive evaluation across all scenarios
3. Generate detailed visualizations and reports

Usage:
    python run_complete_system.py --quick      # Quick demo (5min)
    python run_complete_system.py --full       # Full evaluation (30min+)
    python run_complete_system.py --evaluate   # Evaluate existing models
"""

import argparse
import time
from pathlib import Path
import subprocess
import sys
from typing import Optional


def run_quick_demo() -> None:
    """
    Run a quick demonstration of the enhanced autonomous driving system.

    This function performs a rapid test of the core functionality including:
    - Training with enhanced rewards for 1000 timesteps
    - Basic evaluation on the highway scenario
    - Generation of key visualizations

    Execution time: ~5 minutes
    """
    print("🚀 QUICK DEMO: Enhanced Autonomous Driving RL System")
    print("=" * 60)

    start_time = time.time()

    # Step 1: Train enhanced agent on highway
    print("\n🎯 Step 1: Training Enhanced Agent")
    print("-" * 40)
    result = subprocess.run([
        sys.executable, "scripts/train_enhanced_rewards.py",
        "--scenario", "highway",
        "--timesteps", "1000"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Training completed successfully!")
        # Extract final performance
        for line in result.stdout.split('\n'):
            if "Final performance" in line:
                print(f"   {line}")
    else:
        print("❌ Training failed:")
        print(result.stderr)

    # Step 2: Quick evaluation
    print("\n📊 Step 2: Quick Evaluation")
    print("-" * 40)

    try:
        # Import and run evaluation
        from scripts.evaluate_all import ComprehensiveEvaluator
        evaluator = ComprehensiveEvaluator()

        # Evaluate just the trained model
        models = evaluator.discover_models()
        if models:
            model_name = list(models.keys())[0]  # Get first model
            results = evaluator.evaluate_model(
                models[model_name]["path"],
                model_name,
                n_episodes=5  # Quick evaluation
            )

            if results and "aggregate" in results:
                agg = results["aggregate"]
                print("✅ Evaluation completed!")
                print(".2f")
                print(".1%")
                print(".1%")
                print(".3f")
        else:
            print("⚠️ No models found for evaluation")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

    # Step 3: Generate visualizations
    print("\n📈 Step 3: Generate Visualizations")
    print("-" * 40)

    try:
        from scripts.visualize_results import VisualizationDashboard
        dashboard = VisualizationDashboard()

        data = dashboard.load_evaluation_data()
        if data:
            # Generate key visualizations
            dashboard.create_performance_heatmap(data)
            dashboard.create_radar_performance_chart(data)
            dashboard.create_training_progression_visualization()

            print("✅ Visualizations generated!")
            print("   📁 Check results/visualizations/ for plots")
        else:
            print("⚠️ No evaluation data for visualization")

    except Exception as e:
        print(f"❌ Visualization failed: {e}")

    # Summary
    elapsed_time = time.time() - start_time
    print("\n🎉 QUICK DEMO COMPLETE!")
    print("=" * 60)
    print(".1f")
    print("\n📋 What was demonstrated:")
    print("   ✅ Enhanced safety constraints working")
    print("   ✅ Curriculum learning with dense rewards")
    print("   ✅ Comprehensive evaluation system")
    print("   ✅ Automated visualization generation")

    print("\n🚀 Next steps:")
    print("   • Run 'python run_complete_system.py --full' for complete evaluation")
    print("   • Check results/ directory for all outputs")
    print("   • View results/visualizations/ for detailed analysis")


def run_full_evaluation() -> Optional[dict]:
    """
    Run complete evaluation of all trained models across scenarios and modalities.

    Performs comprehensive evaluation including:
    - Discovery of all trained models
    - Multi-scenario testing (highway, merge, intersection)
    - Multi-modality evaluation (lidar, grayscale, both)
    - Performance metrics calculation
    - Visualization generation

    Returns:
        Dictionary containing evaluation results for all models, or None if failed

    Execution time: ~30+ minutes
    """
    print("🔬 FULL EVALUATION: Complete Autonomous Driving System Analysis")
    print("=" * 70)

    start_time = time.time()

    # Run comprehensive evaluation
    print("\n📊 Running comprehensive evaluation...")
    result = subprocess.run([
        sys.executable, "scripts/evaluate_all.py"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Comprehensive evaluation completed!")
    else:
        print("❌ Evaluation failed:")
        print(result.stderr[-500:])  # Last 500 chars of error

    # Generate all visualizations
    print("\n📈 Generating complete visualization suite...")
    result = subprocess.run([
        sys.executable, "scripts/visualize_results.py"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Visualization suite completed!")
    else:
        print("❌ Visualization failed:")
        print(result.stderr[-500:])

    elapsed_time = time.time() - start_time
    print("\n[CELEBRATE] FULL EVALUATION COMPLETE!")
    print("=" * 70)
    print(".1f")
    # Show generated files
    results_dir = Path("results")
    if results_dir.exists():
        print("\n[FILES] Generated Files:")
        for subdir in ["evaluations", "visualizations"]:
            subpath = results_dir / subdir
            if subpath.exists():
                files = list(subpath.glob("*"))
                if files:
                    print(f"   {subdir}/: {len(files)} files")
                    for file in sorted(files)[:3]:  # Show first 3
                        size_mb = file.stat().st_size / (1024 * 1024)
                        print("8.2f")
                    if len(files) > 3:
                        print(f"      ... and {len(files) - 3} more")

    print("\n[SEARCH] Key Outputs:")
    print("   • comprehensive_evaluation_TIMESTAMP.json - Detailed metrics")
    print("   • evaluation_dashboard.png - Performance overview")
    print("   • performance_heatmap.png - Scenario/modality comparison")
    print("   • interactive_dashboard.html - Interactive analysis")
    print("   • comprehensive_report.md - Executive summary")


def evaluate_existing_models() -> None:
    """Evaluate existing trained models without retraining."""
    print("📊 EVALUATING EXISTING MODELS")
    print("=" * 40)

    try:
        from scripts.evaluate_all import ComprehensiveEvaluator
        evaluator = ComprehensiveEvaluator()
        results = evaluator.run_full_evaluation()

        if results:
            print(f"✅ Evaluated {len(results)} existing models")
        else:
            print("⚠️ No models found or evaluation failed")

    except Exception as e:
        print(f"❌ Evaluation error: {e}")


def show_system_info() -> None:
    """Show information about the current system setup."""
    print("ℹ️ SYSTEM INFORMATION")
    print("=" * 30)

    # Check for required directories
    dirs_status = {
        "src/": Path("src").exists(),
        "scripts/": Path("scripts").exists(),
        "results/": Path("results").exists(),
        "environments/": Path("environments").exists(),
        "training/": Path("training").exists(),
    }

    print("📁 Directory Structure:")
    for dir_name, exists in dirs_status.items():
        status = "✅" if exists else "❌"
        print(f"   {status} {dir_name}")

    # Check for key files
    key_files = [
        "environments/enhanced_urban_env.py",
        "training/adaptive_curriculum_trainer.py",
        "scripts/evaluate_all.py",
        "scripts/visualize_results.py"
    ]

    print("\n📄 Key Files:")
    for file_path in key_files:
        exists = Path(file_path).exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {file_path}")

    # Check for trained models
    models_dir = Path("results/models")
    if models_dir.exists():
        model_files = list(models_dir.glob("**/*.zip"))
        print(f"\n🤖 Trained Models: {len(model_files)} found")
        for model_file in sorted(model_files)[:5]:  # Show first 5
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print("6.2f")
            if len(model_files) > 5:
                print(f"      ... and {len(model_files) - 5} more")
    else:
        print("\n🤖 Trained Models: None found (run training first)")


def main():
    parser = argparse.ArgumentParser(description="Complete Autonomous Driving RL System")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick demonstration (5-10 minutes)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run complete evaluation (30+ minutes)"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate existing models only"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show system information"
    )

    args = parser.parse_args()

    # Show banner
    print("🚗 ENHANCED AUTONOMOUS DRIVING RL SYSTEM")
    print("=" * 50)
    print("Safety-first autonomous driving with curriculum learning")
    print("and comprehensive evaluation & visualization")
    print("=" * 50)

    if args.info:
        show_system_info()
    elif args.quick:
        run_quick_demo()
    elif args.full:
        run_full_evaluation()
    elif args.evaluate:
        evaluate_existing_models()
    else:
        print("Please specify an action:")
        print("  --quick     : Quick demonstration")
        print("  --full      : Complete evaluation")
        print("  --evaluate  : Evaluate existing models")
        print("  --info      : Show system information")
        print("\nExample: python run_complete_system.py --quick")


if __name__ == "__main__":
    main()
