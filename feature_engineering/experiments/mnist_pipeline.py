"""
MNIST Ordering-Primitive Pipeline
Stages 1-6 with visualization and bulk evaluation.

Run from feature_engineering/ directory:
  python experiments/mnist_pipeline.py            # viz only (10 images)
  python experiments/mnist_pipeline.py --eval 100 # viz + evaluate 100/class
  python experiments/mnist_pipeline.py --eval all  # viz + full test set
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torchvision
import torchvision.transforms as transforms

from stages.edge_detection import edge_detection
from stages.winner_take_all import winner_take_all
from stages.spatial_pooling import spatial_pooling
from stages.sweep_detection import (
    SweepDetector, sweep_horizontal, sweep_vertical,
    tighten_columns, tighten_rows,
    STAGE3_CHANNEL_LABELS,
)
from stages.cassian_pooling import cassian_pool_horizontal, cassian_pool_vertical
from stages.digit_classifier import classify, DIGIT_TEMPLATES
from visualization.stage_viz import visualize_all_digits
from visualization.sweep_viz import visualize_sweep


# ------------------------------------------------------------------ #
# Detectors — defined at module level so both main() and evaluate()
# use the same configuration.
# ------------------------------------------------------------------ #
H_DETECTORS = [
    SweepDetector(["h", "h"], output_color=(0, 0, 255), name="h_line"),
]
V_DETECTORS = [
    SweepDetector(["v", "v"], output_color=(255, 0, 0), name="v_line"),
]


# ------------------------------------------------------------------ #
# Single-image pipeline (stages 1-5, no visualization)
# ------------------------------------------------------------------ #
def run_pipeline(image: np.ndarray) -> np.ndarray:
    """Run stages 1-5 on one (28, 28) float64 image.

    Returns the combined Stage-5 map: (H, W, 4) float64
      ch 0-1 = h_pool (horizontal Cassians on h_line / v_line)
      ch 2-3 = v_pool (vertical  Cassians on h_line / v_line)
    """
    v1, d1 = edge_detection(image)
    v2, d2 = winner_take_all(v1, d1)
    v3, _  = spatial_pooling(v2, d2, 2, 2)

    hm, _ = sweep_horizontal(v3, STAGE3_CHANNEL_LABELS, H_DETECTORS, view=2, scan_view=2)
    vm, _ = sweep_vertical(  v3, STAGE3_CHANNEL_LABELS, V_DETECTORS, view=2, scan_view=2)
    combined4 = np.concatenate([hm, vm], axis=2)

    h_pool, _ = cassian_pool_horizontal(combined4, width=3, threshold=2)
    v_pool, _ = cassian_pool_vertical(  combined4, width=3, threshold=2)

    Hh, Wh, Dh = h_pool.shape
    Hv, Wv, Dv = v_pool.shape
    H_max, W_max = max(Hh, Hv), max(Wh, Wv)
    h_pad = np.full((H_max, W_max, Dh), np.inf)
    h_pad[:Hh, :Wh, :] = h_pool
    v_pad = np.full((H_max, W_max, Dv), np.inf)
    v_pad[:Hv, :Wv, :] = v_pool
    return np.concatenate([h_pad, v_pad], axis=2)


# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #
def load_one_per_class(train: bool = False) -> list[np.ndarray]:
    """Return the first example of each digit class as a list indexed 0-9."""
    dataset = torchvision.datasets.MNIST(
        root='/tmp/mnist', train=train, download=True,
        transform=transforms.ToTensor(),
    )
    found: dict[int, np.ndarray] = {}
    for img_tensor, label in dataset:
        cls = int(label)
        if cls not in found:
            found[cls] = img_tensor.squeeze(0).numpy().astype(np.float64) * 255.0
        if len(found) == 10:
            break
    return [found[c] for c in range(10)]


def load_dataset(n_per_class: int | None = None, train: bool = False) -> list[tuple[np.ndarray, int]]:
    """Load up to n_per_class examples per digit class.

    Parameters
    ----------
    n_per_class : max images per class; None = load all available
    train       : use training split if True, else test split

    Returns list of (image_array, label) tuples.
    """
    dataset = torchvision.datasets.MNIST(
        root='/tmp/mnist', train=train, download=True,
        transform=transforms.ToTensor(),
    )
    counts = {c: 0 for c in range(10)}
    samples: list[tuple[np.ndarray, int]] = []
    for img_tensor, label in dataset:
        cls = int(label)
        if n_per_class is None or counts[cls] < n_per_class:
            arr = img_tensor.squeeze(0).numpy().astype(np.float64) * 255.0
            samples.append((arr, cls))
            counts[cls] += 1
        if n_per_class is not None and all(v >= n_per_class for v in counts.values()):
            break
    return samples


# ------------------------------------------------------------------ #
# Bulk evaluation
# ------------------------------------------------------------------ #
def evaluate(n_per_class: int | None) -> None:
    """Classify n_per_class images per digit class and print accuracy."""
    label_str = "all" if n_per_class is None else str(n_per_class)
    total_str = "~10 000" if n_per_class is None else str((n_per_class or 0) * 10)
    print(f"\nEvaluating on {label_str} examples per class ({total_str} total)...")

    samples = load_dataset(n_per_class)
    per_class_correct = {c: 0 for c in range(10)}
    per_class_total   = {c: 0 for c in range(10)}
    no_fire = 0

    for i, (image, label) in enumerate(samples):
        combined6 = run_pipeline(image)
        predicted, _ = classify(combined6)
        per_class_total[label] += 1
        if predicted == label:
            per_class_correct[label] += 1
        if predicted == -1:
            no_fire += 1
        if (i + 1) % 200 == 0:
            done = sum(per_class_correct.values())
            print(f"  {i+1}/{len(samples)} processed  (correct so far: {done})", flush=True)

    total   = len(samples)
    correct = sum(per_class_correct.values())
    print(f"\nPer-class accuracy:")
    print(f"{'Digit':<6}| {'Correct':<8}| {'Total':<7}| Accuracy")
    print(f"{'------'}+{'--------'}+{'-------'}+--------")
    for c in range(10):
        n = per_class_total[c]
        acc = per_class_correct[c] / n * 100 if n else 0.0
        print(f"{c:<6}| {per_class_correct[c]:<8}| {n:<7}| {acc:.1f}%")
    print(f"\nOverall: {correct}/{total} = {correct / total * 100:.1f}%")
    if no_fire:
        print(f"No-fire (predicted -1): {no_fire}/{total} = {no_fire/total*100:.1f}%")


# ------------------------------------------------------------------ #
# Visualization pipeline (10 examples, one per class)
# ------------------------------------------------------------------ #
def print_summary(stage_num: int, stage_name: str, all_values: list) -> None:
    print(f"\nStage {stage_num} — {stage_name}")
    print(f"{'Digit':<6}| {'Active signals':<16}| {'Min value':<11}| {'Mean value'}")
    print(f"{'------'}+{'----------------'}+{'-----------'}+-----------")
    for cls in range(10):
        vals = all_values[cls]
        finite_mask = np.isfinite(vals)
        active_count = int(finite_mask.sum())
        if active_count > 0:
            fv = vals[finite_mask]
            print(f"{cls:<6}| {active_count:<16}| {float(fv.min()):<11.1f}| {float(fv.mean()):.1f}")
        else:
            print(f"{cls:<6}| {active_count:<16}| {'N/A':<11}| N/A")


def main(n_eval) -> None:  # n_eval: False = skip, None = all, int = N/class
    # ------------------------------------------------------------------ #
    # Load one example per class for visualization
    # ------------------------------------------------------------------ #
    print("Loading MNIST test set...")
    images = load_one_per_class()
    print("Loaded one example per digit class (0-9).")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_base = os.path.join(script_dir, "output")

    # Original images
    from PIL import Image
    out_orig = os.path.join(out_base, "original")
    os.makedirs(out_orig, exist_ok=True)
    for cls in range(10):
        img = Image.fromarray(images[cls].astype(np.uint8), mode="L")
        img.save(os.path.join(out_orig, f"original_digit{cls}.png"))
    print(f"Original MNIST images saved to {out_orig}/")

    # ------------------------------------------------------------------ #
    # Stage 1 — Edge Detection
    # ------------------------------------------------------------------ #
    stage1_values, stage1_dir_ids = [], []
    for cls in range(10):
        vals, dids = edge_detection(images[cls])
        stage1_values.append(vals); stage1_dir_ids.append(dids)
    print_summary(1, "Edge Detection", stage1_values)
    out_dir1 = os.path.join(out_base, "stage1")
    visualize_all_digits("stage1", stage1_values, stage1_dir_ids, out_dir1)
    print(f"Stage 1 visualizations saved to {out_dir1}/")

    # ------------------------------------------------------------------ #
    # Stage 2 — Winner-Take-All
    # ------------------------------------------------------------------ #
    stage2_values, stage2_dir_ids = [], []
    for cls in range(10):
        vals, dids = winner_take_all(stage1_values[cls], stage1_dir_ids[cls])
        stage2_values.append(vals); stage2_dir_ids.append(dids)
    print_summary(2, "Winner-Take-All", stage2_values)
    out_dir2 = os.path.join(out_base, "stage2")
    visualize_all_digits("stage2", stage2_values, stage2_dir_ids, out_dir2)
    print(f"Stage 2 visualizations saved to {out_dir2}/")

    # ------------------------------------------------------------------ #
    # Stage 3 — Spatial Pooling
    # ------------------------------------------------------------------ #
    stage3_values, stage3_dir_ids = [], []
    for cls in range(10):
        vals, dids = spatial_pooling(stage2_values[cls], stage2_dir_ids[cls], 2, 2)
        stage3_values.append(vals); stage3_dir_ids.append(dids)
    print_summary(3, "Spatial Pooling", stage3_values)
    out_dir3 = os.path.join(out_base, "stage3")
    visualize_all_digits("stage3", stage3_values, stage3_dir_ids, out_dir3)
    print(f"Stage 3 visualizations saved to {out_dir3}/")

    # ------------------------------------------------------------------ #
    # Stage 4 — Sweep Detection
    # ------------------------------------------------------------------ #
    all_colors = [d.output_color for d in H_DETECTORS + V_DETECTORS]
    out_dir4 = os.path.join(out_base, "stage4")
    os.makedirs(out_dir4, exist_ok=True)
    stage4_maps = []

    print("\nStage 4 — Sweep Detection (H + V)")
    print(f"{'Digit':<6}| {'H fires':<10}| {'V fires':<10}| Shape")
    print(f"{'------'}+{'----------'}+{'----------'}+-----")
    for cls in range(10):
        s3 = stage3_values[cls]
        hm, _ = sweep_horizontal(s3, STAGE3_CHANNEL_LABELS, H_DETECTORS, view=2, scan_view=2)
        vm, _ = sweep_vertical(  s3, STAGE3_CHANNEL_LABELS, V_DETECTORS, view=2, scan_view=2)
        # hm is (H//2, W, Dh) and vm is (H, W//2, Dv) — pad both to the same spatial size
        Hh, Wh, Dh = hm.shape
        Hv, Wv, Dv = vm.shape
        H_max, W_max = max(Hh, Hv), max(Wh, Wv)
        hm_pad = np.full((H_max, W_max, Dh), np.inf); hm_pad[:Hh, :Wh, :] = hm
        vm_pad = np.full((H_max, W_max, Dv), np.inf); vm_pad[:Hv, :Wv, :] = vm
        combined = np.concatenate([hm_pad, vm_pad], axis=2)
        stage4_maps.append(combined)
        print(f"{cls:<6}| {int(np.isfinite(hm).sum()):<10}| {int(np.isfinite(vm).sum()):<10}| {combined.shape[0]}×{combined.shape[1]}")
        visualize_sweep(combined, all_colors).save(os.path.join(out_dir4, f"stage4_digit{cls}.png"))
    print(f"Stage 4 visualizations saved to {out_dir4}/")

    # ------------------------------------------------------------------ #
    # Stage 5 — Cassian Pooling
    # ------------------------------------------------------------------ #
    out_dir5 = os.path.join(out_base, "stage5")
    os.makedirs(out_dir5, exist_ok=True)
    stage5_maps = []

    print("\nStage 5 — Cassian Pooling (width=3, threshold=2, H + V)")
    print(f"{'Digit':<6}| {'H fires':<10}| {'V fires':<10}| {'H shape':<10}| V shape")
    print(f"{'------'}+{'----------'}+{'----------'}+{'----------'}+--------")
    for cls in range(10):
        h_pool, _ = cassian_pool_horizontal(stage4_maps[cls], width=3, threshold=2)
        v_pool, _ = cassian_pool_vertical(  stage4_maps[cls], width=3, threshold=2)
        Hh, Wh, Dh = h_pool.shape
        Hv, Wv, Dv = v_pool.shape
        print(f"{cls:<6}| {int(np.isfinite(h_pool).sum()):<10}| {int(np.isfinite(v_pool).sum()):<10}| {Hh}×{Wh}{'':4}| {Hv}×{Wv}")

        H_max, W_max = max(Hh, Hv), max(Wh, Wv)
        h_pad = np.full((H_max, W_max, Dh), np.inf); h_pad[:Hh, :Wh, :] = h_pool
        v_pad = np.full((H_max, W_max, Dv), np.inf); v_pad[:Hv, :Wv, :] = v_pool
        combined5 = np.concatenate([h_pad, v_pad], axis=2)
        colors5 = [(0, 0, 255)] * Dh + [(255, 0, 0)] * Dv
        visualize_sweep(combined5, colors5).save(os.path.join(out_dir5, f"stage5_digit{cls}.png"))
        stage5_maps.append((h_pool, v_pool))
    print(f"Stage 5 visualizations saved to {out_dir5}/")

    # ------------------------------------------------------------------ #
    # Stage 6 — Digit Classification (visualization batch)
    # ------------------------------------------------------------------ #
    print("\nStage 6 — Digit Classification (visualization batch)")
    print(f"{'True':<6}| {'Pred':<6}| {'Correct?':<10}| Score")
    print(f"{'------'}+{'------'}+{'----------'}+------")
    correct = 0
    for cls in range(10):
        h_pool, v_pool = stage5_maps[cls]
        Hh, Wh, Dh = h_pool.shape; Hv, Wv, Dv = v_pool.shape
        H_max, W_max = max(Hh, Hv), max(Wh, Wv)
        h_pad = np.full((H_max, W_max, Dh), np.inf); h_pad[:Hh, :Wh, :] = h_pool
        v_pad = np.full((H_max, W_max, Dv), np.inf); v_pad[:Hv, :Wv, :] = v_pool
        combined6 = np.concatenate([h_pad, v_pad], axis=2)

        predicted, scores = classify(combined6)
        ok = predicted == cls
        if ok:
            correct += 1
        score = scores.get(predicted, np.inf)
        score_str = f"{score:.1f}" if np.isfinite(score) else "inf"
        print(f"{cls:<6}| {predicted:<6}| {'YES' if ok else 'no':<10}| {score_str}")
    print(f"\nViz-batch accuracy: {correct}/10")

    # ------------------------------------------------------------------ #
    # Bulk evaluation (optional, controlled by --eval)
    # ------------------------------------------------------------------ #
    if n_eval is not False:
        evaluate(n_eval)

    print("\nPipeline complete. Visualizations saved to experiments/output/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval", metavar="N", nargs="?", const="all",
        help="Evaluate after visualization: --eval 100 (N/class) or --eval (full test set).",
    )
    args = parser.parse_args()

    if args.eval is None:
        # --eval not supplied → skip bulk evaluation
        main(n_eval=False)
    elif args.eval.lower() == "all":
        main(n_eval=None)      # None → load_dataset returns everything
    else:
        main(n_eval=int(args.eval))
