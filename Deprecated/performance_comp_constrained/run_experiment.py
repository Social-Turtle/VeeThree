import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../performance_comp')))

from model import EncodedCNN, ConventionalCNN
from data_loader import load_mnist, get_data_loaders
from train import train_model, evaluate_model, get_per_digit_accuracy, get_confusion_matrix
from rank_encodings import *
from custom_pooling import *

from config import (
    TRAINING_SAMPLES, TEST_SAMPLES, EPOCHS,
    KERNEL_SIZES, N_PASS_VALUES, ENCODING_TYPES, NUM_FILTERS_CONV2, DENSE_LAYER_SIZES, MNIST_DATA_DIR, BATCH_SIZE, LEARNING_RATE, LR_DECAY
)

def get_experiment_dir(model_type, kernel_size, dense_size, encoding_type=None, n_pass=None):
    base_dir = os.path.dirname(__file__)
    suffix = f"_c{NUM_FILTERS_CONV2}_d{dense_size}"
    if model_type == 'conventional':
        return os.path.join(base_dir, f"conventional_k{kernel_size}{suffix}")
    else:
        return os.path.join(base_dir, f"{encoding_type}_k{kernel_size}_n{n_pass}{suffix}")

def save_summary(experiment_dir, model_type, kernel_size, dense_size, history, overall_acc, per_digit_acc, encoding_type=None, n_pass=None):
    filepath = os.path.join(experiment_dir, "summary.txt")
    with open(filepath, 'w') as f:
        if model_type == 'conventional':
            f.write(f"Experiment: Conventional CNN (kernel={kernel_size}, dense={dense_size})\n")
        else:
            f.write(f"Experiment: {encoding_type} (kernel={kernel_size}, n_pass={n_pass}, dense={dense_size})\n")
        f.write("=" * 60 + "\n\n")
        f.write("CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model type: {model_type}\n")
        f.write(f"Conv1 kernel size: {kernel_size}x{kernel_size}\n")
        if model_type == 'encoded':
            f.write(f"Encoding: {encoding_type}\n")
            f.write(f"N_pass: {n_pass}\n")
        f.write(f"Dense layer size: {dense_size}\n")
        f.write(f"Conv2: {NUM_FILTERS_CONV2} filters (3x3)\n\n")
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

def run_dense_layer_experiments():
    for dense_size in DENSE_LAYER_SIZES:
        for kernel_size in KERNEL_SIZES:
            for encoding_type in ENCODING_TYPES:
                for n_pass in N_PASS_VALUES:
                    experiment_dir = get_experiment_dir('encoded', kernel_size, dense_size, encoding_type, n_pass)
                    os.makedirs(experiment_dir, exist_ok=True)
                    model = EncodedCNN(encoding_type=encoding_type, n_pass=n_pass, conv1_kernel_size=kernel_size, dense_hidden_size=dense_size)
                    train_loader, test_loader = get_data_loaders(MNIST_DATA_DIR, BATCH_SIZE, TRAINING_SAMPLES, TEST_SAMPLES)
                    history = train_model(model, train_loader, test_loader, EPOCHS, LEARNING_RATE, LR_DECAY)
                    overall_acc, per_digit_acc = evaluate_model(model, test_loader)
                    save_summary(experiment_dir, 'encoded', kernel_size, dense_size, history, overall_acc, per_digit_acc, encoding_type, n_pass)
            experiment_dir = get_experiment_dir('conventional', kernel_size, dense_size)
            os.makedirs(experiment_dir, exist_ok=True)
            model = ConventionalCNN(conv1_kernel_size=kernel_size, dense_hidden_size=dense_size)
            train_loader, test_loader = get_data_loaders(MNIST_DATA_DIR, BATCH_SIZE, TRAINING_SAMPLES, TEST_SAMPLES)
            history = train_model(model, train_loader, test_loader, EPOCHS, LEARNING_RATE, LR_DECAY)
            overall_acc, per_digit_acc = evaluate_model(model, test_loader)
            save_summary(experiment_dir, 'conventional', kernel_size, dense_size, history, overall_acc, per_digit_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run constrained dense layer size experiments.")
    parser.add_argument("--dense", action="store_true", help="Run dense layer size experiments.")
    args = parser.parse_args()
    if args.dense:
        run_dense_layer_experiments()
