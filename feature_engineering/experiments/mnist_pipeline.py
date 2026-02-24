"""
MNIST Ordering-Primitive Pipeline
Stages 1-3 with visualization.
Run from feature_engineering/ directory:
  python experiments/mnist_pipeline.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torchvision
import torchvision.transforms as transforms

from stages.edge_detection import edge_detection
from stages.winner_take_all import winner_take_all
from stages.spatial_pooling import spatial_pooling
from visualization.stage_viz import visualize_all_digits


def load_one_per_class():
    """Load the first example of each digit class 0-9 from the MNIST test set.

    Returns a list of 10 (28, 28) float64 numpy arrays with values in [0, 255],
    indexed by digit class.
    """
    dataset = torchvision.datasets.MNIST(
        root='/tmp/mnist',
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    found = {}  # digit_class -> (28, 28) float64 numpy array
    for image_tensor, label in dataset:
        cls = int(label)
        if cls not in found:
            # image_tensor shape: (1, 28, 28), values in [0.0, 1.0]
            arr = image_tensor.squeeze(0).numpy().astype(np.float64) * 255.0
            found[cls] = arr
        if len(found) == 10:
            break

    # Return as a list indexed 0-9
    return [found[cls] for cls in range(10)]


def print_summary(stage_num: int, stage_name: str, all_values: list) -> None:
    """Print a summary table for a stage across all 10 digit classes.

    all_values: list of 10 (28, 28, 8) float64 arrays.
    """
    print(f"\nStage {stage_num} — {stage_name}")
    print(f"{'Digit':<6}| {'Active signals':<16}| {'Min value':<11}| {'Mean value'}")
    print(f"{'------'}+{'----------------'}+{'-----------'}+-----------")
    for cls in range(10):
        vals = all_values[cls]
        finite_mask = np.isfinite(vals)
        active_count = int(finite_mask.sum())
        if active_count > 0:
            finite_vals = vals[finite_mask]
            min_val = float(finite_vals.min())
            mean_val = float(finite_vals.mean())
            print(f"{cls:<6}| {active_count:<16}| {min_val:<11.1f}| {mean_val:.1f}")
        else:
            print(f"{cls:<6}| {active_count:<16}| {'N/A':<11}| N/A")


def main():
    # ------------------------------------------------------------------ #
    # Load MNIST — one example per class
    # ------------------------------------------------------------------ #
    print("Loading MNIST test set...")
    images = load_one_per_class()
    print("Loaded one example per digit class (0-9).")

    # Output directory base, relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_base = os.path.join(script_dir, "output")

    # ------------------------------------------------------------------ #
    # Stage 1 — Edge Detection
    # ------------------------------------------------------------------ #
    stage1_values = []
    stage1_dir_ids = []
    for cls in range(10):
        vals, dids = edge_detection(images[cls])
        stage1_values.append(vals)
        stage1_dir_ids.append(dids)

    print_summary(1, "Edge Detection", stage1_values)

    out_dir1 = os.path.join(out_base, "stage1")
    visualize_all_digits("stage1", stage1_values, stage1_dir_ids, out_dir1)
    print(f"Stage 1 visualizations saved to {out_dir1}/")

    # ------------------------------------------------------------------ #
    # Stage 2 — Winner-Take-All
    # ------------------------------------------------------------------ #
    stage2_values = []
    stage2_dir_ids = []
    for cls in range(10):
        vals, dids = winner_take_all(stage1_values[cls], stage1_dir_ids[cls])
        stage2_values.append(vals)
        stage2_dir_ids.append(dids)

    print_summary(2, "Winner-Take-All", stage2_values)

    out_dir2 = os.path.join(out_base, "stage2")
    visualize_all_digits("stage2", stage2_values, stage2_dir_ids, out_dir2)
    print(f"Stage 2 visualizations saved to {out_dir2}/")

    # ------------------------------------------------------------------ #
    # Stage 3 — Spatial Pooling
    # ------------------------------------------------------------------ #
    stage3_values = []
    stage3_dir_ids = []
    for cls in range(10):
        vals, dids = spatial_pooling(stage2_values[cls], stage2_dir_ids[cls])
        stage3_values.append(vals)
        stage3_dir_ids.append(dids)

    print_summary(3, "Spatial Pooling", stage3_values)

    out_dir3 = os.path.join(out_base, "stage3")
    visualize_all_digits("stage3", stage3_values, stage3_dir_ids, out_dir3)
    print(f"Stage 3 visualizations saved to {out_dir3}/")

    # ------------------------------------------------------------------ #
    # Done
    # ------------------------------------------------------------------ #
    print("\nPipeline complete. Visualizations saved to experiments/output/")


if __name__ == "__main__":
    main()
