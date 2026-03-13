"""
Performance Comparison Experiment Runner
========================================

Compares:
- 3 encodings (fixed_rank, fixed_vector, learned_rank) × 4 n_pass × 3 kernel sizes = 36 experiments
- 1 conventional CNN × 3 kernel sizes = 3 experiments
- Total: 39 experiments
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(__file__))

from model import EncodedCNN, ConventionalCNN
from data_loader import load_mnist, get_data_loaders
from train import train_model, evaluate_model, get_per_digit_accuracy, get_confusion_matrix
from config import (
    TRAINING_SAMPLES, TEST_SAMPLES, EPOCHS,
    KERNEL_SIZES, N_PASS_VALUES, ENCODING_TYPES, NUM_FILTERS_CONV2
)


def get_experiment_dir(model_type, kernel_size, encoding_type=None, n_pass=None):
    """Get experiment output directory."""
    base_dir = os.path.dirname(__file__)
    suffix = f"_c{NUM_FILTERS_CONV2}"  # Include conv2 filter count
    
    if model_type == 'conventional':
        return os.path.join(base_dir, f"conventional_k{kernel_size}{suffix}")
    else:
        return os.path.join(base_dir, f"{encoding_type}_k{kernel_size}_n{n_pass}{suffix}")


def save_summary(experiment_dir, model_type, kernel_size, history, 
                 overall_acc, per_digit_acc, encoding_type=None, n_pass=None):
    """Save experiment summary."""
    filepath = os.path.join(experiment_dir, "summary.txt")
    
    with open(filepath, 'w') as f:
        if model_type == 'conventional':
            f.write(f"Experiment: Conventional CNN (kernel={kernel_size})\n")
        else:
            f.write(f"Experiment: {encoding_type} (kernel={kernel_size}, n_pass={n_pass})\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model type: {model_type}\n")
        f.write(f"Conv1 kernel size: {kernel_size}x{kernel_size}\n")
        if model_type == 'encoded':
            f.write(f"Encoding: {encoding_type}\n")
            f.write(f"N_pass: {n_pass}\n")
        f.write(f"Conv2: 16 filters (3x3)\n\n")
        
        f.write("RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Final Test Accuracy: {overall_acc:.2f}%\n")
        f.write(f"Best Test Accuracy: {max(history['test_acc']):.2f}%\n")
        f.write(f"Training Time: {history['total_time']:.1f}s\n\n")
        
        f.write("PER-DIGIT ACCURACY\n")
        f.write("-" * 40 + "\n")
        for digit, acc in enumerate(per_digit_acc):
            f.write(f"  Digit {digit}: {acc:.2f}%\n")
        
        f.write("\nTRAINING HISTORY\n")
        f.write("-" * 40 + "\n")
        for epoch in range(len(history['train_acc'])):
            f.write(f"  Epoch {epoch+1}: "
                   f"Loss={history['train_loss'][epoch]:.4f}, "
                   f"Train={history['train_acc'][epoch]:.1f}%, "
                   f"Test={history['test_acc'][epoch]:.1f}%\n")


def save_confusion_matrix(confusion, filepath):
    """Save confusion matrix to file."""
    with open(filepath, 'w') as f:
        f.write("Confusion Matrix (rows=true, cols=predicted)\n")
        f.write("=" * 60 + "\n\n")
        f.write("      " + "".join(f"{i:>5}" for i in range(10)) + "\n")
        f.write("-" * 60 + "\n")
        for i in range(10):
            row_sum = confusion[i].sum()
            f.write(f"{i:>3} | " + "".join(f"{confusion[i, j]:>5}" for j in range(10)))
            f.write(f" | {100 * confusion[i, i] / row_sum:.1f}%\n")


def run_encoded_experiment(encoding_type, n_pass, kernel_size, device='cpu'):
    """Run a single encoded model experiment."""
    experiment_dir = get_experiment_dir('encoded', kernel_size, encoding_type, n_pass)
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {encoding_type} / kernel={kernel_size} / n_pass={n_pass}")
    print(f"{'='*60}")
    
    # Create model
    model = EncodedCNN(encoding_type, n_pass, conv1_kernel_size=kernel_size)
    model.summary()
    
    # Load data
    train_dataset, test_dataset = load_mnist(max_train=TRAINING_SAMPLES, max_test=TEST_SAMPLES)
    train_loader, test_loader = get_data_loaders(train_dataset, test_dataset)
    
    # Train
    history = train_model(model, train_loader, test_loader, epochs=EPOCHS, device=device)
    
    # Evaluate
    overall_acc = evaluate_model(model, test_loader, device)
    per_digit_acc = get_per_digit_accuracy(model, test_loader, device)
    confusion = get_confusion_matrix(model, test_loader, device)
    
    print(f"\nFinal Test Accuracy: {overall_acc:.2f}%")
    
    # Save results
    save_confusion_matrix(confusion, os.path.join(experiment_dir, "confusion_matrix.txt"))
    save_summary(experiment_dir, 'encoded', kernel_size, history, 
                 overall_acc, per_digit_acc, encoding_type, n_pass)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(experiment_dir, "model.pt"))
    
    return overall_acc


def run_conventional_experiment(kernel_size, device='cpu'):
    """Run a conventional CNN experiment."""
    experiment_dir = get_experiment_dir('conventional', kernel_size)
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Experiment: Conventional CNN / kernel={kernel_size}")
    print(f"{'='*60}")
    
    # Create model
    model = ConventionalCNN(conv1_kernel_size=kernel_size)
    model.summary()
    
    # Load data
    train_dataset, test_dataset = load_mnist(max_train=TRAINING_SAMPLES, max_test=TEST_SAMPLES)
    train_loader, test_loader = get_data_loaders(train_dataset, test_dataset)
    
    # Train
    history = train_model(model, train_loader, test_loader, epochs=EPOCHS, device=device)
    
    # Evaluate
    overall_acc = evaluate_model(model, test_loader, device)
    per_digit_acc = get_per_digit_accuracy(model, test_loader, device)
    confusion = get_confusion_matrix(model, test_loader, device)
    
    print(f"\nFinal Test Accuracy: {overall_acc:.2f}%")
    
    # Save results
    save_confusion_matrix(confusion, os.path.join(experiment_dir, "confusion_matrix.txt"))
    save_summary(experiment_dir, 'conventional', kernel_size, history, 
                 overall_acc, per_digit_acc)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(experiment_dir, "model.pt"))
    
    return overall_acc


def run_dense_layer_experiments():
    """Run experiments with varying dense layer sizes."""
    for dense_size in DENSE_LAYER_SIZES:
        for kernel_size in KERNEL_SIZES:
            for encoding_type in ENCODING_TYPES:
                for n_pass in N_PASS_VALUES:
                    experiment_dir = get_experiment_dir('encoded', kernel_size, encoding_type, n_pass)
                    os.makedirs(experiment_dir, exist_ok=True)

                    model = EncodedCNN(kernel_size=kernel_size, encoding_type=encoding_type, n_pass=n_pass, dense_hidden_size=dense_size)
                    train_loader, test_loader = get_data_loaders(MNIST_DATA_DIR, BATCH_SIZE, TRAINING_SAMPLES, TEST_SAMPLES)

                    history = train_model(model, train_loader, test_loader, EPOCHS, LEARNING_RATE, LR_DECAY)
                    overall_acc, per_digit_acc = evaluate_model(model, test_loader)

                    save_summary(experiment_dir, 'encoded', kernel_size, history, overall_acc, per_digit_acc, encoding_type, n_pass)

            # Run conventional CNN experiments
            experiment_dir = get_experiment_dir('conventional', kernel_size)
            os.makedirs(experiment_dir, exist_ok=True)

            model = ConventionalCNN(kernel_size=kernel_size, dense_hidden_size=dense_size)
            train_loader, test_loader = get_data_loaders(MNIST_DATA_DIR, BATCH_SIZE, TRAINING_SAMPLES, TEST_SAMPLES)

            history = train_model(model, train_loader, test_loader, EPOCHS, LEARNING_RATE, LR_DECAY)
            overall_acc, per_digit_acc = evaluate_model(model, test_loader)

            save_summary(experiment_dir, 'conventional', kernel_size, history, overall_acc, per_digit_acc)


def run_all_experiments():
    """Run all 39 experiments."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Starting experiments at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    total_experiments = len(KERNEL_SIZES) * len(ENCODING_TYPES) * len(N_PASS_VALUES) + len(KERNEL_SIZES)
    current = 0
    
    # Encoded experiments
    for kernel_size in KERNEL_SIZES:
        for encoding_type in ENCODING_TYPES:
            for n_pass in N_PASS_VALUES:
                current += 1
                print(f"\n{'#'*60}")
                print(f"# Experiment {current}/{total_experiments}")
                print(f"{'#'*60}")
                
                acc = run_encoded_experiment(encoding_type, n_pass, kernel_size, device)
                results.append({
                    'model': 'encoded',
                    'encoding': encoding_type,
                    'kernel': kernel_size,
                    'n_pass': n_pass,
                    'accuracy': acc
                })
    
    # Conventional experiments
    for kernel_size in KERNEL_SIZES:
        current += 1
        print(f"\n{'#'*60}")
        print(f"# Experiment {current}/{total_experiments}")
        print(f"{'#'*60}")
        
        acc = run_conventional_experiment(kernel_size, device)
        results.append({
            'model': 'conventional',
            'encoding': '-',
            'kernel': kernel_size,
            'n_pass': '-',
            'accuracy': acc
        })
    
    # Save all results summary
    base_dir = os.path.dirname(__file__)
    results_file = f"all_results_c{NUM_FILTERS_CONV2}.txt"
    with open(os.path.join(base_dir, results_file), 'w') as f:
        f.write(f"Performance Comparison Experiment Results (Conv2={NUM_FILTERS_CONV2} filters)\n")
        f.write(f"Run completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        # Table header
        f.write(f"{'Model':<12} {'Encoding':<14} {'Kernel':<8} {'N_Pass':<8} {'Accuracy':<10}\n")
        f.write("-" * 52 + "\n")
        
        for r in results:
            f.write(f"{r['model']:<12} {r['encoding']:<14} {r['kernel']:<8} "
                   f"{str(r['n_pass']):<8} {r['accuracy']:.2f}%\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("\nSummary by Kernel Size:\n")
        f.write("-" * 40 + "\n")
        
        for kernel_size in KERNEL_SIZES:
            f.write(f"\nKernel {kernel_size}x{kernel_size}:\n")
            kernel_results = [r for r in results if r['kernel'] == kernel_size]
            for r in kernel_results:
                if r['model'] == 'conventional':
                    f.write(f"  Conventional: {r['accuracy']:.2f}%\n")
                else:
                    f.write(f"  {r['encoding']} n={r['n_pass']}: {r['accuracy']:.2f}%\n")
    
    print(f"\n{'='*60}")
    print(f"All {total_experiments} experiments completed!")
    print(f"Results saved to {os.path.join(base_dir, results_file)}")
    print(f"{'='*60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Performance Comparison Experiments")
    parser.add_argument('--all', action='store_true', help='Run all 39 experiments')
    parser.add_argument('--encoding', type=str, choices=['fixed_rank', 'fixed_vector', 'learned_rank', 'conventional'])
    parser.add_argument('--kernel', type=int, choices=[3, 4, 5])
    parser.add_argument('--n-pass', type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--dense", action="store_true", help="Run dense layer size experiments.")
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.all:
        run_all_experiments()
    elif args.encoding == 'conventional' and args.kernel:
        run_conventional_experiment(args.kernel, device)
    elif args.encoding and args.kernel and args.n_pass:
        run_encoded_experiment(args.encoding, args.n_pass, args.kernel, device)
    else:
        print("Usage:")
        print("  --all                        Run all 39 experiments")
        print("  --encoding X --kernel K --n-pass N   Run single encoded experiment")
        print("  --encoding conventional --kernel K    Run single conventional experiment")
        sys.exit(1)


if __name__ == "__main__":
    main()
