"""Train LUT model across N_T × EMBED_DIM × N_GRID configs and record Pareto data.

Output: results/lut_pareto.csv

Usage:
    python benchmark/lut_sweep.py           # full sweep (~hours)
    python benchmark/lut_sweep.py --quick   # 3 configs, 1000 train / 200 test
"""

import os
import sys
import csv
import time
import argparse
import json

import numpy as np

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_VEETHREE  = os.path.dirname(_BENCH_DIR)
_RESULTS   = os.path.join(_VEETHREE, "results")
sys.path.insert(0, _VEETHREE)

import lut_model as lm
from main import load_mnist
from benchmark.metrics import lut_active_bits


BIT_WIDTH = 32


# ------------------------------------------------------------------ #
# Sweep configurations  (N_T, EMBED_DIM, N_GRID)
# ------------------------------------------------------------------ #

FULL_CONFIGS = [
    # (4,  16, 2),
    # (4,  16, 4),
    # (4,  32, 2),
    # (4,  32, 4),
    # (8,  16, 2),
    # (8,  16, 4),
    # (8,  32, 2),
    # (8,  32, 4),
    # (8,  64, 4),
    # (16, 32, 4),
    # (16, 32, 7),
    # (16, 64, 4),
    # (32, 32, 4),
    # (32, 64, 4),
    # (32, 64, 7),
    #(16, 8,  7),   # N_T=16, EMBED_DIM=8, N_GRID=7: 49
    #regions → 8 timing spikes each
   #(16, 4,  7),   # extreme neckdown: 49 → 4
    #(8,  8,  7),
    (16,  6,  4),

]

QUICK_CONFIGS = [
    (4,  16, 2),
    (8,  32, 4),
    (16, 64, 4),
]


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _set_globals(N_T, EMBED_DIM, N_GRID):
    """Monkey-patch lut_model module constants for the current config run."""
    lm.N_T       = N_T
    lm.EMBED_DIM = EMBED_DIM
    lm.N_GRID    = N_GRID


def train(model, images, labels, epochs):
    """Minimal training loop (no printing). Returns elapsed seconds."""
    n = len(images)
    t0 = time.time()
    global_step = 0
    for _ in range(epochs):
        indices = np.random.permutation(n)
        for idx in indices:
            global_step += 1
            model.step(images[idx], labels[idx], t=global_step)
    return time.time() - t0


def evaluate_accuracy(model, images, labels):
    """Test accuracy as a float in [0, 100]."""
    correct = 0
    for image, label in zip(images, labels):
        logits, _, _, _ = model.forward(image)
        if int(np.argmax(logits)) == label:
            correct += 1
    return 100.0 * correct / len(images)


def measure_costs(model, images):
    """Mean active signals, seq2s, adds, and multiplies per inference."""
    n_t = lm.N_T
    n_regions = lm.N_GRID * lm.N_GRID

    # Each LUT layer computes y as a sum over N_T selected rows:
    # y[k] = sum_{i=0..N_T-1} S[i, j[i], k]
    # That is (N_T - 1) scalar adds per output element, no multiplies.
    local_adds = sum(
        n_regions * layer.y_dim * max(n_t - 1, 0)
        for layer in model.local_layers
    )
    global_adds = sum(
        layer.y_dim * max(n_t - 1, 0)
        for layer in model.global_layers
    )
    output_adds = lm.N_CLASSES * max(n_t - 1, 0)
    adds_per_image = (local_adds + global_adds + output_adds) * BIT_WIDTH

    total_active = 0
    total_seq2s = 0
    total_adds = 0
    total_multiplies = 0
    for image in images:
        logits, caches, spike_counts, seq2s = model.forward(image)
        total_active += lut_active_bits(spike_counts, caches['y_outputs'])
        total_seq2s += int(seq2s['total'])
        total_adds += int(adds_per_image)
        total_multiplies += 0

    n = len(images)
    return {
        "active_signals": total_active / n,
        "seq2s": total_seq2s / n,
        "adds": total_adds / n,
        "multiplies": total_multiplies / n,
    }


# ------------------------------------------------------------------ #
# Main sweep
# ------------------------------------------------------------------ #

def run_sweep(configs, n_train, n_test, epochs):
    data_dir = os.path.join(_VEETHREE, "mnist", "mnist")
    print("Loading MNIST...")
    train_imgs, train_lbls, test_imgs, test_lbls = load_mnist(
        data_dir=data_dir, max_samples=n_train
    )
    # load_mnist caps test at max_samples//6; re-cap explicitly for quick mode
    if n_test is not None:
        test_imgs  = test_imgs[:n_test]
        test_lbls  = test_lbls[:n_test]

    rows = []
    for N_T, EMBED_DIM, N_GRID in configs:
        _set_globals(N_T, EMBED_DIM, N_GRID)
        config_str = json.dumps({"N_T": N_T, "EMBED_DIM": EMBED_DIM, "N_GRID": N_GRID})
        print(f"\n[LUT] {config_str}")

        model = lm.LUTModel()

        train_time = train(model, train_imgs, train_lbls, epochs)
        accuracy   = evaluate_accuracy(model, test_imgs, test_lbls)
        mean_costs = measure_costs(model, test_imgs)
        mean_bits = mean_costs["active_signals"]

        print(
            f"  acc={accuracy:.2f}%  active_signals={mean_costs['active_signals']:.0f} "
            f"seq2s={mean_costs['seq2s']:.0f} adds={mean_costs['adds']:.0f} "
            f"multiplies={mean_costs['multiplies']:.0f} time={train_time:.1f}s"
        )
        rows.append({
            "model":            "lut",
            "config":           config_str,
            "accuracy":         round(accuracy, 4),
            "active_bits_mean": round(mean_bits, 1),
            "train_time_s":     round(train_time, 2),
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description="LUT Pareto sweep")
    parser.add_argument("--quick", action="store_true",
                        help="3 configs, 1000 train / 200 test samples")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.quick:
        configs = QUICK_CONFIGS
        n_train, n_test, epochs = 1000, 200, 2
    else:
        configs = FULL_CONFIGS
        n_train, n_test, epochs = None, None, 5

    if args.epochs is not None:
        epochs = args.epochs

    print(f"Configs: {len(configs)}  |  epochs: {epochs}")
    rows = run_sweep(configs, n_train, n_test, epochs)

    os.makedirs(_RESULTS, exist_ok=True)
    out_path = os.path.join(_RESULTS, "lut_pareto.csv")

    # Load existing rows and skip duplicate configs
    existing_configs = set()
    file_exists = os.path.exists(out_path)
    if file_exists:
        with open(out_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_configs.add(row["config"])

    # Filter out rows whose config already exists
    rows_to_write = [r for r in rows if r["config"] not in existing_configs]

    # Append new rows to CSV
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "config", "accuracy", "active_bits_mean", "train_time_s"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows_to_write)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
