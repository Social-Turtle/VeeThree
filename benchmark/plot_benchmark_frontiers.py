"""Plot frontier-only Pareto curves from results/benchmark.csv.

Creates two plots, each with one frontier curve per model type:
1) generous: x = active_signals + seq2s + adds + multiplies
2) honest:   x = active_signals + seq2s/2 + adds + (multiplies * 4)

Usage:
    python benchmark/plot_benchmark_frontiers.py
    python benchmark/plot_benchmark_frontiers.py --csv results/benchmark.csv --show
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


MODEL_COLORS = {
    "cnn": "#1f77b4",
    "lut": "#d62728",
    "fe_lut": "#2ca02c",
    "edge_lut": "#ff7f0e",
}


def to_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def compute_x(row: dict, mode: str) -> float:
    seq2s = to_float(row, "seq2s_per_img")
    active = to_float(row, "active_signals_per_img")
    adds = to_float(row, "adds_per_img")
    multiplies = to_float(row, "multiplies_per_img")

    if mode == "generous":
        return active + seq2s + adds + multiplies
    if mode == "honest":
        return active + (seq2s / 2.0) + adds + (multiplies * 4.0)
    if mode == "signals":
        return active
    if mode == "adds":
        return adds
    raise ValueError(f"Unknown mode: {mode}")


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return upper-left frontier (min x, max y).

    A point is kept if its y is strictly better than all previously seen points
    when scanning from smallest x to largest x.
    """
    points_sorted = sorted(points, key=lambda p: (p[0], -p[1]))
    frontier: list[tuple[float, float]] = []
    best_y_so_far = float("-inf")

    for x_val, y_val in points_sorted:
        if y_val > best_y_so_far:
            frontier.append((x_val, y_val))
            best_y_so_far = y_val

    return frontier


def read_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_mode(rows: list[dict], mode: str, out_path: str) -> None:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for row in rows:
        model = (row.get("model") or "").strip()
        if not model:
            continue
        x_val = compute_x(row, mode)
        y_val = to_float(row, "test_acc_pct")
        grouped[model].append((x_val, y_val))

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for model in sorted(grouped.keys()):
        frontier = pareto_frontier(grouped[model])
        if not frontier:
            continue
        xs = [p[0] for p in frontier]
        ys = [p[1] for p in frontier]
        color = MODEL_COLORS.get(model, "#7f7f7f")

        ax.plot(xs, ys, marker="o", linewidth=2, markersize=5, color=color, label=model)

    ax.set_title(f"Benchmark Pareto Frontiers ({mode.capitalize()})")
    ax.set_xlabel("Computed Cost per Image")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xscale("log")
    ax.set_xlim(left=1, right=1e7)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="Model")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot generous/honest/signals/adds frontier curves from benchmark.csv")
    parser.add_argument("--csv", default="results/benchmark.csv", help="Path to benchmark CSV")
    parser.add_argument(
        "--outdir",
        default="results",
        help="Directory where output images are written",
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    rows = read_rows(args.csv)
    if not rows:
        raise RuntimeError(f"No rows found in {args.csv}")

    generous_path = os.path.join(args.outdir, "benchmark_pareto_generous.png")
    honest_path = os.path.join(args.outdir, "benchmark_pareto_honest.png")
    signal_path = os.path.join(args.outdir, "benchmark_pareto_signals.png")
    add_path = os.path.join(args.outdir, "benchmark_pareto_adds.png")

    plot_mode(rows, mode="generous", out_path=generous_path)
    plot_mode(rows, mode="honest", out_path=honest_path)
    plot_mode(rows, mode="signals", out_path=signal_path)
    plot_mode(rows, mode="adds", out_path=add_path)

    print(f"Saved: {generous_path}")
    print(f"Saved: {honest_path}")
    print(f"Saved: {signal_path}")
    print(f"Saved: {add_path}")

    if args.show:
        # Re-open as interactive windows if requested.
        for mode in ("generous", "honest", "signals", "adds"):
            fig, ax = plt.subplots(figsize=(10, 6.5))
            grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for row in rows:
                model = (row.get("model") or "").strip()
                if not model:
                    continue
                grouped[model].append((compute_x(row, mode), to_float(row, "test_acc_pct")))

            for model in sorted(grouped.keys()):
                frontier = pareto_frontier(grouped[model])
                if not frontier:
                    continue
                xs = [p[0] for p in frontier]
                ys = [p[1] for p in frontier]
                color = MODEL_COLORS.get(model, "#7f7f7f")
                ax.plot(xs, ys, marker="o", linewidth=2, markersize=5, color=color, label=model)

            ax.set_title(f"Benchmark Pareto Frontiers ({mode.capitalize()})")
            ax.set_xlabel("Computed Cost per Image")
            ax.set_ylabel("Test Accuracy (%)")
            ax.set_xscale("log")
            ax.set_xlim(left=1e4, right=1e7)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(title="Model")
            plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    main()
