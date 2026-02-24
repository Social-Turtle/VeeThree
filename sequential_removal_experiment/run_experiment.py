"""
Run Sequential Removal Experiments
===================================
"""

import os
import sys
import argparse
import numpy as np
import torch

from config import TRAINING_SAMPLES, TEST_SAMPLES, EPOCHS, LEARNING_RATE
from model import SequentialCNN
from data_loader import load_mnist, get_data_loaders
from train import train_model, get_confusion_matrix


ENCODING_DIRS = {
    'binary': 'Binary',
    'fixed_rank': 'FixedRank',
    'fixed_vector': 'FixedVector',
    'learned_rank': 'LearnedRank',
}


def get_experiment_dir(encoding_type, n_pass, num_filters=8):
    """Get experiment output directory."""
    base_dir = os.path.dirname(__file__)
    encoding_dir = ENCODING_DIRS[encoding_type]
    filters_suffix = f"_{num_filters}f" if num_filters != 8 else ""
    return os.path.join(base_dir, encoding_dir + filters_suffix, f"n_pass_{n_pass}")


def save_confusion_matrix(confusion, filepath):
    """Save confusion matrix to file."""
    with open(filepath, 'w') as f:
        f.write("Confusion Matrix\n")
        f.write("=" * 60 + "\n")
        f.write("Rows = True label, Columns = Predicted label\n\n")
        
        # Header
        f.write("     ")
        for j in range(10):
            f.write(f"  {j:3d}")
        f.write("\n")
        f.write("     " + "-" * 45 + "\n")
        
        # Rows
        for i in range(10):
            f.write(f"  {i} |")
            for j in range(10):
                f.write(f"  {confusion[i][j]:3d}")
            f.write("\n")


def save_summary(experiment_dir, encoding_type, n_pass, history, 
                 overall_acc, per_digit_acc, num_filters=8):
    """Save experiment summary."""
    filepath = os.path.join(experiment_dir, "summary.txt")
    
    with open(filepath, 'w') as f:
        f.write(f"Experiment Summary: {ENCODING_DIRS[encoding_type]}/n_pass_{n_pass}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Encoding: {encoding_type}\n")
        f.write(f"N_PASS: {n_pass}\n")
        f.write(f"Layers: 2\n")
        f.write(f"Filters: {num_filters} (fixed edge detectors)\n\n")
        
        f.write("RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Overall Test Accuracy: {overall_acc:.2f}%\n\n")
        
        f.write("Per-Digit Accuracy:\n")
        for digit in range(10):
            acc = per_digit_acc[digit]
            bar = "█" * int(acc / 5)
            f.write(f"  {digit}: {acc:5.1f}% {bar}\n")
        
        f.write("\nTRAINING\n")
        f.write("-" * 40 + "\n")
        f.write(f"Epochs: {len(history['train_loss'])}\n")
        f.write(f"Final training loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"Final training accuracy: {history['train_acc'][-1]:.2f}%\n")
        f.write(f"Total training time: {history['total_time']/60:.1f} minutes\n")


def run_single_experiment(encoding_type, n_pass, num_filters=8, device='cpu'):
    """Run a single experiment."""
    experiment_dir = get_experiment_dir(encoding_type, n_pass, num_filters)
    os.makedirs(experiment_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print(f"Experiment: {ENCODING_DIRS[encoding_type]} / n_pass={n_pass} / {num_filters} filters")
    print("=" * 60)
    
    # Create model
    model = SequentialCNN(encoding_type, n_pass, num_filters)
    model.summary()
    
    # Load data
    train_dataset, test_dataset = load_mnist(
        max_train=TRAINING_SAMPLES,
        max_test=TEST_SAMPLES
    )
    train_loader, test_loader = get_data_loaders(train_dataset, test_dataset)
    
    # Train
    history = train_model(model, train_loader, test_loader, device=device)
    
    # Evaluate
    confusion, per_digit_acc = get_confusion_matrix(model, test_loader, device)
    overall_acc = sum(per_digit_acc.values()) / 10
    
    # For overall accuracy, recalculate properly
    total_correct = sum(confusion[i][i] for i in range(10))
    total_samples = confusion.sum()
    overall_acc = 100.0 * total_correct / total_samples
    
    print(f"\nFinal Test Accuracy: {overall_acc:.2f}%")
    
    # Save results
    save_confusion_matrix(confusion, os.path.join(experiment_dir, "confusion_matrix.txt"))
    save_summary(experiment_dir, encoding_type, n_pass, history, overall_acc, per_digit_acc, num_filters)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(experiment_dir, "model.pt"))
    
    return overall_acc


def run_all_experiments(num_filters=8, max_n_pass=None):
    """Run experiments for all encodings."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Filters: {num_filters}")
    
    if max_n_pass is None:
        max_n_pass = num_filters
    
    results = []
    total_experiments = 4 * max_n_pass
    current = 0
    
    for encoding_type in ['binary', 'fixed_rank', 'fixed_vector', 'learned_rank']:
        for n_pass in range(1, max_n_pass + 1):
            current += 1
            print(f"\n{'#'*60}")
            print(f"# Experiment {current}/{total_experiments}")
            print(f"{'#'*60}")
            
            acc = run_single_experiment(encoding_type, n_pass, num_filters, device)
            results.append({
                'encoding': encoding_type,
                'n_pass': n_pass,
                'accuracy': acc
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"\n{'Encoding':<15} {'N_Pass':<8} {'Accuracy':<10}")
    print("-" * 35)
    for r in results:
        print(f"{r['encoding']:<15} {r['n_pass']:<8} {r['accuracy']:.2f}%")
    
    # Save results summary
    base_dir = os.path.dirname(__file__)
    suffix = f"_{num_filters}f" if num_filters != 8 else ""
    with open(os.path.join(base_dir, f"all_results{suffix}.txt"), 'w') as f:
        f.write(f"Sequential Removal Experiment Results ({num_filters} filters)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Encoding':<15} {'N_Pass':<8} {'Accuracy':<10}\n")
        f.write("-" * 35 + "\n")
        for r in results:
            f.write(f"{r['encoding']:<15} {r['n_pass']:<8} {r['accuracy']:.2f}%\n")


def main():
    parser = argparse.ArgumentParser(description="Sequential Removal Experiments")
    parser.add_argument('--encoding', type=str, default=None,
                        choices=['binary', 'fixed_rank', 'fixed_vector', 'learned_rank'],
                        help='Encoding type (if not specified, runs all)')
    parser.add_argument('--n-pass', type=int, default=None,
                        help='N_pass value (if not specified, runs all)')
    parser.add_argument('--filters', type=int, default=8, choices=[4, 8],
                        help='Number of filters (4 or 8)')
    parser.add_argument('--all', action='store_true',
                        help='Run all experiments')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.all or (args.encoding is None and args.n_pass is None):
        run_all_experiments(args.filters)
    elif args.encoding and args.n_pass:
        run_single_experiment(args.encoding, args.n_pass, args.filters, device)
    else:
        print("Please specify both --encoding and --n-pass, or use --all")
        sys.exit(1)


if __name__ == "__main__":
    main()
