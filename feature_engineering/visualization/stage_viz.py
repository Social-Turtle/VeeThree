import numpy as np
from PIL import Image
import os


def visualize_stage(
    stage_name: str,
    digit_class: int,
    values: np.ndarray,   # (H, W, 8) float64
    dir_ids: np.ndarray,  # (H, W, 8) int8  (not actually used, dirs are fixed 0-7)
    out_dir: str,
) -> None:
    """Saves a (2H)x(2W) PNG to out_dir/{stage_name}_digit{digit_class}.png.

    Layout of the 2x2 sub-pixel block for each logical pixel (r, c):
        top-left     (2*r,   2*c  ): vertical   (dirs 0,1) -> RED
        top-right    (2*r,   2*c+1): horizontal (dirs 2,3) -> BLUE
        bottom-left  (2*r+1, 2*c  ): diagonal-1 (dirs 4,5) -> GREEN
        bottom-right (2*r+1, 2*c+1): diagonal-2 (dirs 6,7) -> YELLOW (R+G)

    Per direction type, the active value is np.nanmin of the two direction
    channels after replacing np.inf with np.nan.  If both channels are inf,
    the active value is np.inf.

    Intensity normalization is local to the entire 56x56 image:
        - Collect all finite values across all four direction types and all pixels.
        - v_min, v_max = min / max of those finite values.
        - intensity(v) = 1 - (v - v_min) / (v_max - v_min)
          (lower value -> brighter, maps [v_min, v_max] -> [1.0, 0.0])
        - If v_min == v_max, all finite values map to intensity 1.0.
        - np.inf -> intensity 0.0 (black).
    """
    os.makedirs(out_dir, exist_ok=True)
    H, W = values.shape[:2]

    # ------------------------------------------------------------------
    # Step 1: compute per-direction-type active values, shape (H, W)
    # ------------------------------------------------------------------
    # Replace inf with nan so np.nanmin ignores them cleanly.
    v = values.astype(np.float64)  # (H, W, 8)

    def _nanmin_pair(ch_a: int, ch_b: int) -> np.ndarray:
        """Return np.nanmin of two channels; result is np.inf where both are inf."""
        a = np.where(np.isinf(v[:, :, ch_a]), np.nan, v[:, :, ch_a])
        b = np.where(np.isinf(v[:, :, ch_b]), np.nan, v[:, :, ch_b])
        stacked = np.stack([a, b], axis=-1)          # (28, 28, 2)
        result = np.nanmin(stacked, axis=-1)          # (28, 28)
        # nanmin returns nan only when all inputs are nan (i.e., both were inf)
        result = np.where(np.isnan(result), np.inf, result)
        return result

    v_vertical   = _nanmin_pair(0, 1)   # (28, 28)
    v_horizontal = _nanmin_pair(2, 3)
    v_diagonal1  = _nanmin_pair(4, 5)
    v_diagonal2  = _nanmin_pair(6, 7)

    # ------------------------------------------------------------------
    # Step 2: local intensity normalization across the full 56x56 image
    # ------------------------------------------------------------------
    all_type_values = [v_vertical, v_horizontal, v_diagonal1, v_diagonal2]

    # Gather all finite values
    finite_vals = np.concatenate([
        tv[np.isfinite(tv)].ravel() for tv in all_type_values
    ])

    if finite_vals.size == 0:
        # All signals are inf -> all-black image
        v_min = v_max = 0.0
        has_finite = False
    else:
        v_min = float(finite_vals.min())
        v_max = float(finite_vals.max())
        has_finite = True

    def _to_intensity(type_values: np.ndarray) -> np.ndarray:
        """Map a (28, 28) array of active values to [0, 1] intensity floats."""
        intensity = np.zeros_like(type_values, dtype=np.float64)
        finite_mask = np.isfinite(type_values)
        if has_finite:
            if v_max == v_min:
                intensity[finite_mask] = 1.0
            else:
                intensity[finite_mask] = (
                    1.0 - (type_values[finite_mask] - v_min) / (v_max - v_min)
                )
        # np.inf positions remain 0.0 (black)
        return intensity

    int_vertical   = _to_intensity(v_vertical)    # (28, 28) in [0, 1]
    int_horizontal = _to_intensity(v_horizontal)
    int_diagonal1  = _to_intensity(v_diagonal1)
    int_diagonal2  = _to_intensity(v_diagonal2)

    # ------------------------------------------------------------------
    # Step 3: build the (2H, 2W, 3) uint8 image
    # ------------------------------------------------------------------
    img = np.zeros((2 * H, 2 * W, 3), dtype=np.uint8)

    for r in range(H):
        for c in range(W):
            # top-left: vertical -> RED channel
            iv = int_vertical[r, c]
            img[2 * r,     2 * c    ] = (round(255 * iv), 0, 0)

            # top-right: horizontal -> BLUE channel
            ih = int_horizontal[r, c]
            img[2 * r,     2 * c + 1] = (0, 0, round(255 * ih))

            # bottom-left: diagonal-1 -> GREEN channel
            id1 = int_diagonal1[r, c]
            img[2 * r + 1, 2 * c    ] = (0, round(255 * id1), 0)

            # bottom-right: diagonal-2 -> YELLOW (R + G channels)
            id2 = int_diagonal2[r, c]
            val = round(255 * id2)
            img[2 * r + 1, 2 * c + 1] = (val, val, 0)

    # ------------------------------------------------------------------
    # Step 4: save
    # ------------------------------------------------------------------
    filename = f"{stage_name}_digit{digit_class}.png"
    filepath = os.path.join(out_dir, filename)
    Image.fromarray(img, mode="RGB").save(filepath)


def visualize_all_digits(
    stage_name: str,
    all_values: list,    # list of 10 (28, 28, 8) arrays
    all_dir_ids: list,   # list of 10 (28, 28, 8) arrays
    out_dir: str,
) -> None:
    """Calls visualize_stage for digit classes 0-9."""
    for digit_class in range(10):
        visualize_stage(
            stage_name=stage_name,
            digit_class=digit_class,
            values=all_values[digit_class],
            dir_ids=all_dir_ids[digit_class],
            out_dir=out_dir,
        )
