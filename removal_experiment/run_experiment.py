"""
Removal Experiment - Main Entry Point
======================================
Investigates the impact of limiting first-layer information flow
on CNN classification performance.

Usage:
    python run_experiment.py              # Interactive single run
    python run_experiment.py --batch      # Batch mode with predefined configs
"""

import os
import sys
import argparse
import numpy as np

from config import (
    DEFAULT_LAYERS, DEFAULT_FILTERS, DEFAULT_N_PASS,
    TRAINING_SAMPLES, TEST_SAMPLES, EPOCHS, LEARNING_RATE,
    EXPERIMENT_BASE_DIR, EXAMPLES_PER_DIGIT
)
from cnn import SparseCNN
from data_loader import load_mnist, get_examples_per_digit
from training import train_model, evaluate_model
from visualization import (
    plot_training_history, 
    plot_per_digit_accuracy,
    save_digit_examples
)


# =============================================================================
# DIRECTORY MANAGEMENT
# =============================================================================

def get_experiment_dir(num_layers, filters_per_layer, n_pass):
    """
    Generate the directory name for a specific experiment configuration.
    
    Format: {N}_layers_{F}_filters_{P}_pass/
    Where F can be a single number or hyphen-separated list.
    """
    if isinstance(filters_per_layer, int):
        filters_str = str(filters_per_layer)
    else:
        filters_str = "-".join(map(str, filters_per_layer))
    
    n_pass_str = str(n_pass) if n_pass is not None else "all"
    
    dir_name = f"{num_layers}_layers_{filters_str}_filters_{n_pass_str}_pass"
    
    # Navigate up from removal_experiment to project root
    base_path = os.path.dirname(__file__)
    return os.path.join(base_path, dir_name)


def experiment_exists(experiment_dir):
    """Check if an experiment has already been run."""
    model_path = os.path.join(experiment_dir, "model.npz")
    return os.path.exists(model_path)


# =============================================================================
# USER INTERACTION
# =============================================================================

def prompt_for_configuration():
    """
    Interactively prompt the user for experiment configuration.
    
    Returns:
        (num_layers, filters_per_layer, n_pass)
    """
    print("\n" + "=" * 60)
    print("REMOVAL EXPERIMENT - Configuration")
    print("=" * 60)
    
    # Number of layers
    while True:
        try:
            num_layers = input(f"\n1) How many convolutional layers? [{DEFAULT_LAYERS}]: ").strip()
            if num_layers == "":
                num_layers = DEFAULT_LAYERS
            else:
                num_layers = int(num_layers)
            
            if num_layers < 1:
                print("   Must have at least 1 layer.")
                continue
            break
        except ValueError:
            print("   Please enter a valid integer.")
    
    # Filters per layer
    while True:
        try:
            filters_input = input(f"\n2) How many filters per layer? [{DEFAULT_FILTERS}]: ").strip()
            if filters_input == "":
                filters_per_layer = DEFAULT_FILTERS
            else:
                filters_per_layer = int(filters_input)
            
            if filters_per_layer < 1:
                print("   Must have at least 1 filter.")
                continue
            break
        except ValueError:
            print("   Please enter a valid integer.")
    
    # N_PASS
    while True:
        try:
            n_pass_input = input(f"\n3) Activations to pass from layer 1 (or 'all') [all]: ").strip()
            if n_pass_input == "" or n_pass_input.lower() == "all":
                n_pass = None
            else:
                n_pass = int(n_pass_input)
                
                if n_pass < 1:
                    print("   Must pass at least 1 activation (or use 'all').")
                    continue
                
                if n_pass >= filters_per_layer:
                    print(f"   Note: {n_pass} >= {filters_per_layer} filters, so all will pass.")
                    n_pass = None
            break
        except ValueError:
            print("   Please enter a valid integer or 'all'.")
    
    return num_layers, filters_per_layer, n_pass


def prompt_for_reuse(experiment_dir):
    """
    Ask user whether to reuse existing model or retrain.
    
    Returns:
        'reuse', 'retrain', or 'cancel'
    """
    print(f"\n⚠️  Existing experiment found: {os.path.basename(experiment_dir)}")
    
    while True:
        choice = input("\nOptions:\n"
                      "  [r] Reuse existing model (regenerate plots only)\n"
                      "  [t] Retrain from scratch\n"
                      "  [c] Cancel\n"
                      "Choice [r]: ").strip().lower()
        
        if choice == "" or choice == "r":
            return "reuse"
        elif choice == "t":
            return "retrain"
        elif choice == "c":
            return "cancel"
        else:
            print("   Please enter 'r', 't', or 'c'.")


# =============================================================================
# MAIN EXPERIMENT LOGIC
# =============================================================================

def run_single_experiment(num_layers, filters_per_layer, n_pass, force_retrain=False,
                          use_custom_filters=False, activation='sigmoid'):
    """
    Run a single experiment with the given configuration.
    
    Args:
        num_layers: Number of conv layers
        filters_per_layer: Filters per layer (int or list)
        n_pass: Activations to pass from layer 1 (None = all)
        force_retrain: If True, retrain even if model exists
        use_custom_filters: If True, use hand-crafted edge filters for layer 1
        activation: Activation function for conv layers ('sigmoid', 'relu', 'abs')
    
    Returns:
        Dict with results
    """
    experiment_dir = get_experiment_dir(num_layers, filters_per_layer, n_pass)
    model_path = os.path.join(experiment_dir, "model.npz")
    
    # Check for existing experiment
    if experiment_exists(experiment_dir) and not force_retrain:
        choice = prompt_for_reuse(experiment_dir)
        if choice == "cancel":
            print("Experiment cancelled.")
            return None
        elif choice == "reuse":
            print("\nLoading existing model...")
            cnn = SparseCNN.load(model_path)
            cnn.summary()
            
            # Still need data for evaluation
            print("\nLoading data for evaluation...")
            _, _, test_images, test_labels = load_mnist(
                max_train=100,  # Minimal train load
                max_test=TEST_SAMPLES
            )
            
            # Evaluate
            overall_accuracy, per_digit_accuracy, confusion = evaluate_model(
                cnn, test_images, test_labels
            )
            
            # Regenerate plots
            _generate_all_visualizations(
                cnn, experiment_dir, 
                history=None,  # No training history for reused model
                overall_accuracy=overall_accuracy,
                per_digit_accuracy=per_digit_accuracy,
                test_images=test_images,
                test_labels=test_labels,
                config_name=os.path.basename(experiment_dir)
            )
            
            return {
                'overall_accuracy': overall_accuracy,
                'per_digit_accuracy': per_digit_accuracy,
                'reused': True
            }
    
    # Create experiment directory
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Initialize model
    print("\n" + "=" * 60)
    print("INITIALIZING MODEL")
    print("=" * 60)
    
    # Load custom filters if requested
    custom_filters = None
    if use_custom_filters:
        from custom_filters import get_edge_filters, FILTER_NAMES
        custom_filters = get_edge_filters()
        print(f"Using custom edge detection filters:")
        for i, name in enumerate(FILTER_NAMES):
            print(f"  Filter {i}: {name}")
    
    print(f"Activation function: {activation}")
    
    cnn = SparseCNN(
        num_layers=num_layers,
        filters_per_layer=filters_per_layer,
        n_pass=n_pass,
        custom_filters=custom_filters,
        activation=activation
    )
    cnn.summary()
    
    # Load data
    print("\nLoading MNIST data...")
    train_images, train_labels, test_images, test_labels = load_mnist(
        max_train=TRAINING_SAMPLES,
        max_test=TEST_SAMPLES
    )
    
    # Train
    history = train_model(
        cnn, train_images, train_labels,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    overall_accuracy, per_digit_accuracy, confusion = evaluate_model(
        cnn, test_images, test_labels
    )
    
    # Save model
    cnn.save(model_path)
    
    # Save confusion matrix
    confusion_path = os.path.join(experiment_dir, "confusion_matrix.txt")
    _save_confusion_matrix(confusion, confusion_path)
    
    # Generate all visualizations
    _generate_all_visualizations(
        cnn, experiment_dir,
        history=history,
        overall_accuracy=overall_accuracy,
        per_digit_accuracy=per_digit_accuracy,
        test_images=test_images,
        test_labels=test_labels,
        config_name=os.path.basename(experiment_dir)
    )
    
    print("\n" + "=" * 60)
    print(f"EXPERIMENT COMPLETE")
    print(f"Results saved to: {experiment_dir}")
    print("=" * 60)
    
    return {
        'overall_accuracy': overall_accuracy,
        'per_digit_accuracy': per_digit_accuracy,
        'history': history,
        'reused': False
    }


def _generate_all_visualizations(cnn, experiment_dir, history, 
                                  overall_accuracy, per_digit_accuracy,
                                  test_images, test_labels, config_name):
    """Generate all plots and save example activation maps."""
    
    plots_dir = os.path.join(experiment_dir, "plots")
    examples_dir = os.path.join(experiment_dir, "digit_examples")
    
    print("\nGenerating visualizations...")
    
    # Training history plots (if available)
    if history is not None:
        plot_training_history(history, plots_dir, config_name)
        print(f"  ✓ Training curves saved to {plots_dir}")
    
    # Per-digit accuracy
    plot_per_digit_accuracy(per_digit_accuracy, plots_dir, config_name)
    print(f"  ✓ Per-digit accuracy saved to {plots_dir}")
    
    # Summary file
    _save_summary(
        experiment_dir, config_name,
        cnn, overall_accuracy, per_digit_accuracy, history
    )
    print(f"  ✓ Summary saved")
    
    # Example activation maps for each digit
    examples = get_examples_per_digit(test_images, test_labels, EXAMPLES_PER_DIGIT)
    save_digit_examples(cnn, examples, examples_dir)
    print(f"  ✓ Digit examples saved to {examples_dir}")


def _save_summary(experiment_dir, config_name, cnn, 
                  overall_accuracy, per_digit_accuracy, history):
    """Save experiment summary to text file."""
    summary_path = os.path.join(experiment_dir, "summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write(f"Experiment Summary: {config_name}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Layers: {cnn.num_layers}\n")
        f.write(f"Filters per layer: {cnn.filters_per_layer}\n")
        f.write(f"N_PASS (first layer): {cnn.n_pass if cnn.n_pass else 'ALL'}\n")
        f.write(f"Filter size: {cnn.filter_size}x{cnn.filter_size}\n")
        f.write(f"Pool size: {cnn.pool_size}x{cnn.pool_size}\n\n")
        
        f.write("RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Overall Test Accuracy: {overall_accuracy * 100:.2f}%\n\n")
        
        f.write("Per-Digit Accuracy:\n")
        for digit in range(10):
            acc = per_digit_accuracy[digit] * 100
            bar = "█" * int(acc / 5)
            f.write(f"  {digit}: {acc:5.1f}% {bar}\n")
        
        if history is not None:
            f.write(f"\nTRAINING\n")
            f.write("-" * 40 + "\n")
            f.write(f"Epochs: {len(history['epoch_losses'])}\n")
            f.write(f"Final training loss: {history['epoch_losses'][-1]:.4f}\n")
            f.write(f"Final training accuracy: {history['epoch_accuracies'][-1] * 100:.2f}%\n")
            f.write(f"Total training time: {history['total_time'] / 60:.1f} minutes\n")


def _save_confusion_matrix(confusion, filepath):
    """Save confusion matrix to text file."""
    with open(filepath, 'w') as f:
        f.write("Confusion Matrix\n")
        f.write("(rows = actual, columns = predicted)\n\n")
        
        # Header
        f.write("     ")
        for i in range(10):
            f.write(f"{i:5d}")
        f.write("\n")
        f.write("     " + "-" * 50 + "\n")
        
        # Matrix
        for i in range(10):
            f.write(f"{i:3d} |")
            for j in range(10):
                f.write(f"{confusion[i, j]:5d}")
            f.write("\n")


# =============================================================================
# BATCH MODE
# =============================================================================

def run_batch_experiments(configs):
    """
    Run multiple experiments in batch.
    
    Args:
        configs: List of (num_layers, filters_per_layer, n_pass) tuples
    """
    print("\n" + "=" * 60)
    print("BATCH MODE - Running multiple experiments")
    print("=" * 60)
    print(f"Configurations to run: {len(configs)}")
    
    results = []
    
    for i, (num_layers, filters_per_layer, n_pass) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running: "
              f"{num_layers} layers, {filters_per_layer} filters, "
              f"n_pass={n_pass if n_pass else 'all'}")
        
        result = run_single_experiment(
            num_layers, filters_per_layer, n_pass,
            force_retrain=False  # Will prompt if exists
        )
        
        if result is not None:
            results.append({
                'config': (num_layers, filters_per_layer, n_pass),
                **result
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    
    for result in results:
        layers, filters, n_pass = result['config']
        acc = result['overall_accuracy'] * 100
        status = "(reused)" if result.get('reused') else "(trained)"
        print(f"  {layers}L, {filters}F, n_pass={n_pass if n_pass else 'all'}: "
              f"{acc:.2f}% {status}")
    
    return results


def get_default_batch_configs():
    """
    Generate default batch configurations for comparison.
    
    Tests various n_pass values with a standard architecture.
    """
    base_layers = 3
    base_filters = 12
    
    # Test n_pass from 1 to all
    n_pass_values = [1, 2, 3, 4, 6, 8, 10, None]  # None = all (12)
    
    return [(base_layers, base_filters, n_pass) for n_pass in n_pass_values]


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the removal experiment."""
    parser = argparse.ArgumentParser(
        description="Removal Experiment: Test impact of first-layer sparsification"
    )
    parser.add_argument(
        '--batch', action='store_true',
        help='Run batch mode with predefined configurations'
    )
    parser.add_argument(
        '--layers', type=int, default=None,
        help='Number of conv layers (skips prompt)'
    )
    parser.add_argument(
        '--filters', type=int, default=None,
        help='Filters per layer (skips prompt)'
    )
    parser.add_argument(
        '--n-pass', type=str, default=None,
        help='Activations to pass from layer 1 (int or "all", skips prompt)'
    )
    parser.add_argument(
        '--retrain', action='store_true',
        help='Force retraining even if model exists'
    )
    parser.add_argument(
        '--custom-filters', action='store_true',
        help='Use custom edge detection filters (8 filters: H/V/diagonal edges)'
    )
    parser.add_argument(
        '--activation', type=str, default='sigmoid',
        choices=['sigmoid', 'relu', 'abs'],
        help='Activation function for conv layers (default: sigmoid)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█   REMOVAL EXPERIMENT: First-Layer Sparsification Study   █")
    print("█" + " " * 58 + "█")
    print("█" * 60)
    
    if args.batch:
        # Batch mode
        configs = get_default_batch_configs()
        run_batch_experiments(configs)
    else:
        # Single experiment mode
        if args.layers is not None and args.filters is not None and args.n_pass is not None:
            # All params provided via CLI
            num_layers = args.layers
            filters_per_layer = args.filters
            n_pass = None if args.n_pass.lower() == 'all' else int(args.n_pass)
        else:
            # Interactive prompt
            num_layers, filters_per_layer, n_pass = prompt_for_configuration()
        
        run_single_experiment(
            num_layers, filters_per_layer, n_pass,
            force_retrain=args.retrain,
            use_custom_filters=args.custom_filters,
            activation=args.activation
        )


if __name__ == "__main__":
    main()
