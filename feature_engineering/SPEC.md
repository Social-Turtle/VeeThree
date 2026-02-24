# MNIST Ordering-Primitive Pipeline — Feature Spec

---

## Feature 1a: Core Primitive (`primitives/core.py`)

Implement the general-purpose ordering-sensitive primitive exactly as defined
in `sequence_functional_description.txt`. This module knows nothing about
pixels or images — it operates on arbitrary numeric sequences.

**Definition:** `N(s_0, s_1, ..., s_{n-1}) = { s_{n-1} if s_0 < s_1 < ... < s_{n-1}; ∞ otherwise }`

Output is the **last element** of the sequence when all elements are strictly
increasing left-to-right; `np.inf` otherwise.

**Vectorized API (batch over spatial dims):**
```python
def apply_primitive(sequences: np.ndarray) -> np.ndarray:
    """
    sequences: (..., N) float array — last axis is the sequence of length N.
    Returns: (...,) float array — last element if strictly increasing, np.inf otherwise.

    Works for any N >= 2.
    """
```

Convenience wrapper for the common N=3 case:
```python
def apply_primitive_3(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Stacks A, B, C along a new last axis and delegates to apply_primitive."""
```

No pixel-specific logic here. No value remapping. Pure ordering detection.

---

## Feature 1b: Pixel-to-Sequence Converter (`primitives/pixel_converter.py`)

Bridge layer between raw pixel windows and the core primitive. Handles all
MNIST-specific value semantics.

**Design:** We want to detect **strictly decreasing** pixel sequences
(bright→dark gradient along a direction). The core primitive detects strictly
**increasing** sequences. Bridge: invert pixel values (`p' = 255 - p`) so that
a decreasing pixel sequence becomes an increasing inverted sequence.

**Per-direction value formula:**
1. Given `(first, center, last)` pixel values for a direction window.
2. Invert: `(A, B, C) = (255 - first, 255 - center, 255 - last)`.
3. Use `apply_primitive_3(A, B, C)` to check strict increase of inverted values
   (equivalent to checking `first > center > last` in original pixel space).
4. If primitive fires (result ≠ ∞): `value = 255 - (first - last)`.
5. If primitive does not fire: `value = np.inf`.

**Vectorized API:**
```python
def pixels_to_values(
    first: np.ndarray,
    center: np.ndarray,
    last: np.ndarray,
) -> np.ndarray:
    """
    first, center, last: same-shape float arrays of raw pixel values [0, 255].
    Returns same-shape float array of primitive outputs;
    np.inf where strictly-decreasing condition is not met.
    Value formula: 255 - (first - last) when activated.
    """
```

**Rationale for value formula:**
- `first > last` always when activated → `first - last > 0` → value ∈ [0, 255).
- Larger drop (strong edge) → smaller value → brighter in visualization.
- `first == last + 1` (weakest valid edge) → value = 254 (near-black).
- `first = 255, last = 0` (maximum edge) → value = 0 (maximum brightness).

---

## Feature 2: Stage 1 — Directional Edge Detection (`stages/edge_detection.py`)

Apply 8 directional windows to every interior pixel of a 28×28 image
(interior = rows 1–26, cols 1–26, i.e. the 26×26 non-border region).
Uses `pixels_to_values` from `primitives/pixel_converter.py` for each direction.

**Eight directions** (each is a `[first, center, last]` triple):

| ID | Name               | Sequence                              |
|----|--------------------|---------------------------------------|
| 0  | vertical-down      | [above, center, below]                |
| 1  | vertical-up        | [below, center, above]                |
| 2  | horizontal-right   | [left, center, right]                 |
| 3  | horizontal-left    | [right, center, left]                 |
| 4  | diagonal-1-fwd     | [up-left, center, down-right]         |
| 5  | diagonal-1-bwd     | [down-right, center, up-left]         |
| 6  | diagonal-2-fwd     | [up-right, center, down-left]         |
| 7  | diagonal-2-bwd     | [down-left, center, up-right]         |

Border pixels (outer ring) are left as `np.inf` for all 8 directions.

**Direction type** groupings (for color encoding later):
- **Vertical** (red): directions 0, 1
- **Horizontal** (blue): directions 2, 3
- **Diagonal-1** (green): directions 4, 5
- **Diagonal-2** (yellow): directions 6, 7

**Output:**
```python
def edge_detection(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    image: (28, 28) float array, pixel values in [0, 255].
    Returns:
      values:    (28, 28, 8) float array — primitive output per direction.
      dir_ids:   (28, 28, 8) int array  — direction ID (0–7) per channel.
    """
```

---

## Feature 3: Stage 2 — Winner-Take-All (`stages/winner_take_all.py`)

Per-pixel: keep only the direction with the **minimum** value across the 8
directions. Set all other 7 to `np.inf`.

**Output:**
```python
def winner_take_all(values: np.ndarray, dir_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    values:  (28, 28, 8) float — stage 1 outputs.
    dir_ids: (28, 28, 8) int   — direction IDs.
    Returns same shapes; only the winning direction channel is non-inf
    per pixel.
    """
```

Pixels where ALL 8 directions are `np.inf` remain all-inf.

---

## Feature 4: Stage 3 — Spatial Pooling (`stages/spatial_pooling.py`)

Apply **non-overlapping 2×2 minimum pooling** while preserving 28×28 geometry.

- Tile the 28×28 image into 14×14 non-overlapping 2×2 blocks.
- Within each block, find the single pixel+direction with the minimum value
  across all 4 pixels × 8 directions = 32 candidates.
- In the output, that one winning pixel/direction retains its value; all
  other 31 are set to `np.inf`.
- Output is still `(28, 28, 8)` — geometry is preserved, ~75% of entries
  become `np.inf`.

**Output:**
```python
def spatial_pooling(values: np.ndarray, dir_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    values:  (28, 28, 8) float — stage 2 outputs.
    dir_ids: (28, 28, 8) int   — direction IDs.
    Returns same shapes with pooling applied.
    """
```

---

## Feature 5: Visualization (`visualization/stage_viz.py`)

Generate per-stage diagnostic PNG images for one example digit per class (0–9).

**Image structure:** Each MNIST pixel → 2×2 sub-pixel block, one sub-pixel
per direction type:
```
[vertical(red)  | horizontal(blue)  ]
[diagonal1(green)| diagonal2(yellow)]
```

**Color encoding:**
- Vertical (dirs 0,1): red channel
- Horizontal (dirs 2,3): blue channel
- Diagonal-1 (dirs 4,5): green channel
- Diagonal-2 (dirs 6,7): yellow (`R+G` channels, i.e. RGB = (255,255,0) scaled)

**Intensity mapping (local per image):**
- Collect all finite values in the current image.
- Normalize: `intensity = 1 - (v - v_min) / (v_max - v_min)` → maps to [0,1]
  (lower value = brighter).
- `np.inf` → intensity 0 (black).
- If all values are `np.inf`, the image is all black.

**Output API:**
```python
def visualize_stage(
    stage_name: str,
    digit_class: int,
    values: np.ndarray,   # (28, 28, 8)
    dir_ids: np.ndarray,  # (28, 28, 8)
    out_dir: str,
) -> None:
    """Saves a (56×56)-pixel PNG to out_dir/stage_name_digit{digit_class}.png"""
```

Also provide a convenience function:
```python
def visualize_all_digits(stage_name: str, all_values, all_dir_ids, out_dir: str) -> None:
    """Calls visualize_stage for each of the 10 digit classes."""
```

---

## Feature 6: Pipeline Orchestration (`experiments/mnist_pipeline.py`)

Top-level script that:

1. **Loads MNIST** via `torchvision` (or `tensorflow_datasets` as fallback),
   selects the **first example of each digit class (0–9)** from the test set.
2. Normalizes pixel values to `[0, 255]` float.
3. Runs **Stage 1** (edge detection) on all 10 images.
4. Saves Stage 1 visualizations to `experiments/output/stage1/`.
5. Runs **Stage 2** (winner-take-all) on Stage 1 outputs.
6. Saves Stage 2 visualizations to `experiments/output/stage2/`.
7. Runs **Stage 3** (spatial pooling) on Stage 2 outputs.
8. Saves Stage 3 visualizations to `experiments/output/stage3/`.

Print a summary table after each stage showing, per digit:
- Number of active (finite) signals
- Min / mean value of active signals

**CLI:** `python experiments/mnist_pipeline.py`

---

## Module Layout

```
feature_engineering/
├── primitives/
│   ├── __init__.py
│   ├── core.py            # Feature 1a — general N() primitive
│   └── pixel_converter.py # Feature 1b — pixel→value bridge
├── stages/
│   ├── __init__.py
│   ├── edge_detection.py   # Feature 2
│   ├── winner_take_all.py  # Feature 3
│   └── spatial_pooling.py  # Feature 4
├── visualization/
│   ├── __init__.py
│   └── stage_viz.py     # Feature 5
└── experiments/
    ├── __init__.py
    ├── mnist_pipeline.py  # Feature 6
    └── output/
        ├── stage1/
        ├── stage2/
        └── stage3/
```

---

## Data Contract Between Stages

Every stage passes `(values, dir_ids)`:
- `values`: `np.ndarray` of shape `(28, 28, 8)`, dtype `float64`, `np.inf` for inactive.
- `dir_ids`: `np.ndarray` of shape `(28, 28, 8)`, dtype `int8`, values `0–7`
  (constant — direction IDs don't change through the pipeline).

---

## Dependencies

- `numpy`
- `torch` + `torchvision` (MNIST loading)
- `Pillow` (PNG output)
