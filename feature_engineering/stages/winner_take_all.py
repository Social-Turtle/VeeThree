import numpy as np


def winner_take_all(values: np.ndarray, dir_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    values:  (28, 28, 8) float64 — stage 1 outputs.
    dir_ids: (28, 28, 8) int8    — direction IDs (unchanged, pass through).
    Returns same shapes; only the winning direction channel (argmin along axis=2) is non-inf per pixel.
    Pixels where all 8 are np.inf remain all-inf.
    """
    # Find the index of the minimum value along the direction axis for each pixel.
    # np.argmin treats np.inf as a valid value, so for all-inf pixels it will return
    # some index (typically 0), but we guard against writing those below.
    winner_idx = np.argmin(values, axis=2)  # (28, 28)

    # Determine which pixels have all-inf (no active direction).
    # The minimum value of an all-inf pixel is still np.inf.
    min_values = values[
        np.arange(28)[:, np.newaxis],   # row indices (28, 1)
        np.arange(28)[np.newaxis, :],   # col indices (1, 28)
        winner_idx,                      # winning channel per pixel (28, 28)
    ]  # (28, 28)
    active_mask = np.isfinite(min_values)  # (28, 28) — True where at least one direction fired

    # Build output: start everything as inf, then copy the winning channel value
    # for each pixel that has at least one finite direction.
    out_values = np.full_like(values, np.inf)

    # Advanced indexing to write winner values in one vectorized step.
    # r, c are the row/col coordinates of active pixels.
    rows, cols = np.where(active_mask)          # 1-D arrays of active pixel coordinates
    winning_dirs = winner_idx[rows, cols]        # 1-D array of winning direction indices

    out_values[rows, cols, winning_dirs] = values[rows, cols, winning_dirs]

    return out_values, dir_ids.copy()
