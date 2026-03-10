"""Train CNN across architecture configs and record accuracy vs. active bits.

Output: results/cnn_pareto.csv

Usage:
    python benchmark/cnn_sweep.py           # full sweep (~hours)
    python benchmark/cnn_sweep.py --quick   # 3 configs, 1000 train / 200 test
"""

import os
import sys
import csv
import time
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

_BENCH_DIR  = os.path.dirname(os.path.abspath(__file__))
_VEETHREE   = os.path.dirname(_BENCH_DIR)
_RESULTS    = os.path.join(_VEETHREE, "results")
sys.path.insert(0, _VEETHREE)

from conventional_cnn import ConventionalCNN
from benchmark.metrics import cnn_active_bits


# ------------------------------------------------------------------ #
# Sweep configurations
# ------------------------------------------------------------------ #

FULL_CONFIGS = [
    # (arch, n_filters, n_filters2, hidden_size)
    ("linear", None,  None,  None),
    ("small",  4,     None,  None),
    ("small",  8,     None,  None),
    ("small",  16,    None,  None),
    ("small",  32,    None,  None),
    ("lenet",  4,     8,     64),
    ("lenet",  4,     8,     128),
    ("lenet",  8,     16,    64),
    ("lenet",  8,     16,    128),
    ("lenet",  8,     16,    256),
    ("lenet",  16,    32,    128),
    ("lenet",  16,    32,    256),
    ("lenet",  32,    64,    256),
]

QUICK_CONFIGS = [
    ("linear", None,  None,  None),
    ("small",  8,     None,  None),
    ("lenet",  8,     16,    128),
]


# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #

def load_mnist(n_train: int | None, n_test: int | None):
    """Return PyTorch DataLoaders for MNIST train and test splits."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = torchvision.datasets.MNIST(root='/tmp/mnist', train=True,  download=True, transform=transform)
    test_ds  = torchvision.datasets.MNIST(root='/tmp/mnist', train=False, download=True, transform=transform)

    if n_train is not None:
        train_ds = Subset(train_ds, list(range(min(n_train, len(train_ds)))))
    if n_test is not None:
        test_ds  = Subset(test_ds,  list(range(min(n_test,  len(test_ds)))))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)
    return train_loader, test_loader


# ------------------------------------------------------------------ #
# Training and evaluation
# ------------------------------------------------------------------ #

def train_model(model: ConventionalCNN, loader: DataLoader, epochs: int, device: torch.device) -> float:
    """Train with cross-entropy + Adam. Returns total elapsed seconds."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    t0 = time.time()
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return time.time() - t0


def evaluate_accuracy(model: ConventionalCNN, loader: DataLoader, device: torch.device) -> float:
    """Return test accuracy as a percentage."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total


def measure_active_bits(model: ConventionalCNN, loader: DataLoader, device: torch.device) -> float:
    """Return mean active bits per inference across the test set."""
    model.register_cost_hooks()
    model.eval()
    total_bits = 0
    total_images = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            for i in range(x.size(0)):
                model.reset_active_bits()
                model(x[i:i+1])
                total_bits += model.get_active_bits()
                total_images += 1
    model.remove_cost_hooks()
    return total_bits / max(total_images, 1)


# ------------------------------------------------------------------ #
# Main sweep
# ------------------------------------------------------------------ #

def run_sweep(configs, n_train, n_test, epochs, device):
    train_loader, test_loader = load_mnist(n_train, n_test)
    rows = []

    for arch, n_filters, n_filters2, hidden_size in configs:
        kwargs = {"arch": arch}
        if n_filters    is not None: kwargs["n_filters"]   = n_filters
        if n_filters2   is not None: kwargs["n_filters2"]  = n_filters2
        if hidden_size  is not None: kwargs["hidden_size"] = hidden_size

        model = ConventionalCNN(**kwargs).to(device)
        config_str = json.dumps(model.config_dict())
        print(f"\n[CNN] {config_str}")

        train_time = train_model(model, train_loader, epochs, device)
        accuracy   = evaluate_accuracy(model, test_loader, device)
        mean_bits  = measure_active_bits(model, test_loader, device)

        print(f"  acc={accuracy:.2f}%  active_bits={mean_bits:.0f}  time={train_time:.1f}s")
        rows.append({
            "model":            "cnn",
            "config":           config_str,
            "accuracy":         round(accuracy, 4),
            "active_bits_mean": round(mean_bits, 1),
            "train_time_s":     round(train_time, 2),
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description="CNN Pareto sweep")
    parser.add_argument("--quick", action="store_true",
                        help="3 configs, 1000 train / 200 test samples")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs")
    args = parser.parse_args()

    if args.quick:
        configs = QUICK_CONFIGS
        n_train, n_test, epochs = 1000, 200, 3
    else:
        configs = FULL_CONFIGS
        n_train, n_test, epochs = None, None, 10

    if args.epochs is not None:
        epochs = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  configs: {len(configs)}  |  epochs: {epochs}")

    rows = run_sweep(configs, n_train, n_test, epochs, device)

    os.makedirs(_RESULTS, exist_ok=True)
    out_path = os.path.join(_RESULTS, "cnn_pareto.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "config", "accuracy", "active_bits_mean", "train_time_s"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
