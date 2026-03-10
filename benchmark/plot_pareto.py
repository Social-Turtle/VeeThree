"""Plot accuracy vs. active bits Pareto curves for all three model families.

Reads results/cnn_pareto.csv, results/lut_pareto.csv, results/fe_pareto.csv
and produces results/pareto_active_bits.png.

Usage:
    python benchmark/plot_pareto.py
    python benchmark/plot_pareto.py --no-frontier   # skip Pareto frontier line
"""

import os
import sys
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_VEETHREE  = os.path.dirname(_BENCH_DIR)
_RESULTS   = os.path.join(_VEETHREE, "results")


MODEL_STYLE = {
    "cnn": {"color": "#2196F3", "marker": "o", "label": "CNN"},
    "lut": {"color": "#FF5722", "marker": "s", "label": "LUT"},
    "fe":  {"color": "#4CAF50", "marker": "^", "label": "Feature Eng."},
}


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return points on the accuracy-maximising Pareto frontier.

    Given (active_bits, accuracy) pairs, find the subset where no point is
    dominated (lower bits AND higher accuracy available).
    """
    sorted_pts = sorted(points, key=lambda p: p[0])  # ascending bits
    frontier = []
    best_acc = -np.inf
    for bits, acc in sorted_pts:
        if acc > best_acc:
            frontier.append((bits, acc))
            best_acc = acc
    return frontier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-frontier", action="store_true",
                        help="Do not draw Pareto frontier lines")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load all results
    # ------------------------------------------------------------------ #
    all_rows: dict[str, list[dict]] = {}
    for model_name in ("cnn", "lut", "fe"):
        path = os.path.join(_RESULTS, f"{model_name}_pareto.csv")
        rows = load_csv(path)
        if rows:
            all_rows[model_name] = rows
            print(f"Loaded {len(rows)} rows from {os.path.basename(path)}")
        else:
            print(f"[warn] {os.path.basename(path)} not found or empty — skipping")

    if not all_rows:
        print("No data found. Run sweep scripts first.")
        return

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 7))

    legend_handles = []
    all_bits_vals = []
    all_acc_vals  = []

    for model_name, rows in all_rows.items():
        style = MODEL_STYLE[model_name]
        bits  = [float(r["active_bits_mean"]) for r in rows]
        accs  = [float(r["accuracy"])         for r in rows]

        # Scatter: all configs
        ax.scatter(
            bits, accs,
            color=style["color"], marker=style["marker"],
            s=60, alpha=0.75, zorder=3,
        )

        # Pareto frontier line
        if not args.no_frontier and len(bits) > 1:
            frontier = pareto_frontier(list(zip(bits, accs)))
            fx, fy = zip(*frontier)
            ax.plot(fx, fy, color=style["color"], linewidth=2, zorder=4, alpha=0.9)

        all_bits_vals.extend(bits)
        all_acc_vals.extend(accs)

        patch = mpatches.Patch(color=style["color"], label=style["label"])
        legend_handles.append(patch)

    # ------------------------------------------------------------------ #
    # Axes and labels
    # ------------------------------------------------------------------ #
    ax.set_xlabel("Mean active bits per inference", fontsize=13)
    ax.set_ylabel("Test accuracy (%)", fontsize=13)
    ax.set_title("MNIST: Accuracy vs. Compute Cost (active bits)", fontsize=14)
    ax.legend(handles=legend_handles, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Log scale if dynamic range is large
    if all_bits_vals:
        ratio = max(all_bits_vals) / (min(all_bits_vals) + 1e-9)
        if ratio > 100:
            ax.set_xscale("log")
            ax.set_xlabel("Mean active bits per inference (log scale)", fontsize=13)

    plt.tight_layout()
    out_path = os.path.join(_RESULTS, "pareto_active_bits.png")
    os.makedirs(_RESULTS, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
