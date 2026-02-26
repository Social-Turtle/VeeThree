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
    CassianSweepDetector, cassian_sweep_horizontal, cassian_sweep_vertical,
    tighten_columns, tighten_rows,
    STAGE3_CHANNEL_LABELS,
)
from stages.digit_classifier import classify, DIGIT_TEMPLATES
from visualization.stage_viz import visualize_all_digits
from visualization.sweep_viz import visualize_sweep


# ------------------------------------------------------------------ #
# Detectors — defined at module level so both main() and evaluate()
# use the same configuration.
# ------------------------------------------------------------------ #
# Cassian sweep detectors: window_size / threshold are configurable.
# Start with 3/2 (fire if ≥2 of 3 consecutive pixels are active).
H_DETECTORS = [
    CassianSweepDetector("v", window_size=3, threshold=2, output_color=(0, 0, 255), name="h_line"),
]
V_DETECTORS = [
    CassianSweepDetector("h", window_size=3, threshold=2, output_color=(255, 0, 0), name="v_line"),
]
D_DETECTORS = [
    CassianSweepDetector("d", window_size=3, threshold=2, output_color=(0, 200, 0), name="d_line"),
]


# ------------------------------------------------------------------ #
# Stage-1 post-processing: derive "d" channel from h+v co-occurrence
# ------------------------------------------------------------------ #
# Stage-1 channel labels for the 4 active directions.
STAGE1_CHANNEL_LABELS: list[str] = ["v", "v", "h", "h"]


def _add_diagonal_channel(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Derive a "d" channel wherever both h and v are present at a pixel.

    Parameters
    ----------
    values : (H, W, 4+) float64 from stage 1 (channels 0-1 = v, 2-3 = h)

    Returns
    -------
    enriched : (H, W, 5) float64 — original 4 channels + derived "d"
    labels   : length-5 channel label list
    """
    H, W = values.shape[:2]
    v_channels = values[:, :, 0:2]   # v_down, v_up
    h_channels = values[:, :, 2:4]   # h_right, h_left

    # Pixel has "v" if ANY v-channel is finite, likewise for "h".
    has_v = np.any(np.isfinite(v_channels), axis=2)
    has_h = np.any(np.isfinite(h_channels), axis=2)
    has_both = has_v & has_h

    # Derive "d" value: use the min finite value across all 4 channels.
    d_channel = np.full((H, W, 1), np.inf)
    for ch in range(4):
        mask = has_both & np.isfinite(values[:, :, ch])
        d_channel[:, :, 0] = np.where(
            mask, np.minimum(d_channel[:, :, 0], values[:, :, ch]), d_channel[:, :, 0],
        )

    enriched = np.concatenate([values[:, :, 0:4], d_channel], axis=2)
    labels = STAGE1_CHANNEL_LABELS + ["d"]
    return enriched, labels


# ------------------------------------------------------------------ #
# Single-image pipeline (stages 1 + Cassian sweep)
# ------------------------------------------------------------------ #
def run_pipeline(image: np.ndarray) -> np.ndarray:
    """Run stage 1, derive diagonals, then Cassian sweep detection.

    Stages 2 (WTA) and 3 (pooling) are skipped so that interleaved
    h/v patterns at diagonal strokes are preserved.  A derived "d"
    channel is added wherever h and v co-occur at the same pixel.

    Cassian sweeps use a sliding window with a threshold (e.g. 2/3)
    instead of strict sequential matching, filtering noise at
    detection time.

    Returns the combined map: (H, W, 3) float64
      ch 0 = h_line, ch 1 = d_line, ch 2 = v_line
    """
    v1, _ = edge_detection(image)
    enriched, labels = _add_diagonal_channel(v1)

    hm, _ = cassian_sweep_horizontal(enriched, labels, H_DETECTORS)
    vm, _ = cassian_sweep_vertical(  enriched, labels, V_DETECTORS)

    # Diagonal: sweep both H and V, merge into one channel (take min).
    dh, _ = cassian_sweep_horizontal(enriched, labels, D_DETECTORS)
    dv, _ = cassian_sweep_vertical(  enriched, labels, D_DETECTORS)
    H_max = max(hm.shape[0], vm.shape[0], dh.shape[0], dv.shape[0])
    W_max = max(hm.shape[1], vm.shape[1], dh.shape[1], dv.shape[1])

    d_merged = np.full((H_max, W_max, 1), np.inf)
    for d_map in (dh, dv):
        h, w, _ = d_map.shape
        active = np.isfinite(d_map[:, :, 0])
        d_merged[:h, :w, 0] = np.where(
            active,
            np.minimum(d_merged[:h, :w, 0], d_map[:, :, 0]),
            d_merged[:h, :w, 0],
        )

    # Pad all to same spatial size, concatenate as [h_line, d_line, v_line].
    hm_pad = np.full((H_max, W_max, hm.shape[2]), np.inf)
    vm_pad = np.full((H_max, W_max, vm.shape[2]), np.inf)
    hm_pad[:hm.shape[0], :hm.shape[1], :] = hm
    vm_pad[:vm.shape[0], :vm.shape[1], :] = vm
    return np.concatenate([hm_pad, d_merged, vm_pad], axis=2)


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

    # confusion[true_label][predicted] = count
    confusion: dict[int, dict[int, int]] = {
        c: {p: 0 for p in range(-1, 10)} for c in range(10)
    }

    for i, (image, label) in enumerate(samples):
        combined6 = run_pipeline(image)
        predicted, _ = classify(combined6)
        per_class_total[label] += 1
        confusion[label][predicted] += 1
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

    # Save per-digit prediction distribution bar charts.
    _save_confusion_bars(confusion, per_class_total)


def _save_confusion_bars(
    confusion: dict[int, dict[int, int]],
    per_class_total: dict[int, int],
) -> None:
    """Save a bar chart per true digit showing prediction distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "output", "confusion")
    os.makedirs(out_dir, exist_ok=True)

    pred_labels = list(range(10)) + [-1]
    x_labels = [str(d) for d in range(10)] + ["none"]

    for true_digit in range(10):
        counts = [confusion[true_digit][p] for p in pred_labels]
        total = per_class_total[true_digit]

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["green" if pred_labels[i] == true_digit else "steelblue"
                  for i in range(len(pred_labels))]
        bars = ax.bar(x_labels, counts, color=colors)

        # Label each bar with its count.
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(count), ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("Predicted digit")
        ax.set_ylabel("Count")
        acc = counts[true_digit] / total * 100 if total else 0
        ax.set_title(f"True digit {true_digit}  (n={total}, acc={acc:.1f}%)")
        ax.set_ylim(0, max(counts) * 1.15 if max(counts) > 0 else 1)

        path = os.path.join(out_dir, f"digit{true_digit}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"\nPrediction distribution charts saved to {out_dir}/")


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

    # # ------------------------------------------------------------------ #
    # # Stage 2 — Winner-Take-All
    # # ------------------------------------------------------------------ #
    # stage2_values, stage2_dir_ids = [], []
    # for cls in range(10):
    #     vals, dids = winner_take_all(stage1_values[cls], stage1_dir_ids[cls])
    #     stage2_values.append(vals); stage2_dir_ids.append(dids)
    # print_summary(2, "Winner-Take-All", stage2_values)
    # out_dir2 = os.path.join(out_base, "stage2")
    # visualize_all_digits("stage2", stage2_values, stage2_dir_ids, out_dir2)
    # print(f"Stage 2 visualizations saved to {out_dir2}/")

    # # ------------------------------------------------------------------ #
    # # Stage 3 — Spatial Pooling
    # # ------------------------------------------------------------------ #
    # stage3_values, stage3_dir_ids = [], []
    # for cls in range(10):
    #     vals, dids = spatial_pooling(stage2_values[cls], stage2_dir_ids[cls], 2, 2)
    #     stage3_values.append(vals); stage3_dir_ids.append(dids)
    # print_summary(3, "Spatial Pooling", stage3_values)
    # out_dir3 = os.path.join(out_base, "stage3")
    # visualize_all_digits("stage3", stage3_values, stage3_dir_ids, out_dir3)
    # print(f"Stage 3 visualizations saved to {out_dir3}/")

    # ------------------------------------------------------------------ #
    # Stage 4 — Cassian Sweep Detection (H + D + V)
    # ------------------------------------------------------------------ #
    all_colors = [d.output_color for d in H_DETECTORS + D_DETECTORS + V_DETECTORS]
    out_dir4 = os.path.join(out_base, "stage4")
    os.makedirs(out_dir4, exist_ok=True)
    stage4_maps = []

    print("\nStage 4 — Cassian Sweep Detection (H + D + V)")
    print(f"{'Digit':<6}| {'H fires':<10}| {'D fires':<10}| {'V fires':<10}| Shape")
    print(f"{'------'}+{'----------'}+{'----------'}+{'----------'}+-----")
    for cls in range(10):
        combined = run_pipeline(images[cls])
        stage4_maps.append(combined)
        h_fires = int(np.isfinite(combined[:, :, 0]).sum())
        d_fires = int(np.isfinite(combined[:, :, 1]).sum())
        v_fires = int(np.isfinite(combined[:, :, 2]).sum())
        print(f"{cls:<6}| {h_fires:<10}| {d_fires:<10}| {v_fires:<10}| {combined.shape[0]}x{combined.shape[1]}")
        visualize_sweep(combined, all_colors).save(os.path.join(out_dir4, f"stage4_digit{cls}.png"))
    print(f"Stage 4 visualizations saved to {out_dir4}/")

    # ------------------------------------------------------------------ #
    # Stage 5 — Digit Classification (visualization batch)
    # ------------------------------------------------------------------ #
    print("\nStage 5 — Digit Classification (visualization batch)")
    print(f"{'True':<6}| {'Pred':<6}| {'Correct?':<10}| Score")
    print(f"{'------'}+{'------'}+{'----------'}+------")
    correct = 0
    for cls in range(10):
        predicted, scores = classify(stage4_maps[cls])
        ok = predicted == cls
        if ok:
            correct += 1
        score = scores.get(predicted, 0)
        score_str = str(score) if score > 0 else "none"
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
