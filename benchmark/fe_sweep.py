"""Run feature-engineering pipeline across pool_size configs.

Output: results/fe_pareto.csv

Usage:
    python benchmark/fe_sweep.py            # full sweep
    python benchmark/fe_sweep.py --quick    # 3 configs
"""

import os
import sys
import csv
import time
import argparse
import json

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_VEETHREE  = os.path.dirname(_BENCH_DIR)
_FE_DIR    = os.path.join(_VEETHREE, "feature_engineering")
_RESULTS   = os.path.join(_VEETHREE, "results")

# The FE pipeline must be imported from inside feature_engineering/
sys.path.insert(0, _VEETHREE)
sys.path.insert(0, _FE_DIR)
sys.path.insert(0, os.path.join(_FE_DIR, "experiments"))

from mnist_pipeline import evaluate_with_config


# ------------------------------------------------------------------ #
# Sweep configurations  (pool_size only — active_dirs sweep deferred
# until more edge directions are implemented in edge_detection.py)
# ------------------------------------------------------------------ #

FULL_CONFIGS  = [1, 2, 4]
QUICK_CONFIGS = [1, 2, 4]   # same — only 3 meaningful pool_size values


# ------------------------------------------------------------------ #
# Main sweep
# ------------------------------------------------------------------ #

def run_sweep(configs, n_per_class):
    rows = []
    for pool_size in configs:
        config_str = json.dumps({"pool_size": pool_size})
        print(f"\n[FE] {config_str}")

        t0 = time.time()
        accuracy, mean_active_bits = evaluate_with_config(
            n_per_class=n_per_class,
            pool_size=pool_size,
        )
        elapsed = time.time() - t0

        print(f"  acc={accuracy*100:.2f}%  active_bits={mean_active_bits:.1f}  time={elapsed:.1f}s")
        rows.append({
            "model":            "fe",
            "config":           config_str,
            "accuracy":         round(accuracy * 100, 4),
            "active_bits_mean": round(mean_active_bits, 1),
            "train_time_s":     0.0,
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description="FE Pareto sweep")
    parser.add_argument("--quick", action="store_true",
                        help="50 examples per class")
    parser.add_argument("--n-per-class", type=int, default=None,
                        help="Override samples per class (None = full test set)")
    args = parser.parse_args()

    if args.quick:
        configs     = QUICK_CONFIGS
        n_per_class = 50
    else:
        configs     = FULL_CONFIGS
        n_per_class = None

    if args.n_per_class is not None:
        n_per_class = args.n_per_class

    print(f"Configs: {len(configs)}  |  n_per_class: {n_per_class or 'all'}")
    rows = run_sweep(configs, n_per_class)

    os.makedirs(_RESULTS, exist_ok=True)
    out_path = os.path.join(_RESULTS, "fe_pareto.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "config", "accuracy", "active_bits_mean", "train_time_s"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
