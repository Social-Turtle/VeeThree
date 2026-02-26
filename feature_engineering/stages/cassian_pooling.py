"""Stage 5: Cassian pooling — width-2 AND gates applied along rows and columns.

For each window of `width` consecutive positions along a row (or column),
the gate fires (outputs the last value) only if ALL positions in the window
are active (finite).  Inactive positions propagate as np.inf.
"""
from __future__ import annotations

import numpy as np


def cassian_pool_horizontal(
    firing_map: np.ndarray,
    width: int = 3,
    threshold: int = 2,
) -> tuple[np.ndarray, list[str]]:
    """Cassian gate sliding along each row.

    For each window of `width` consecutive columns, output fires (= last value)
    if at least `threshold` positions in the window are active (finite).

    Parameters
    ----------
    firing_map : (H, W, D) float64 — np.inf = inactive
    width      : window size
    threshold  : minimum number of active positions required to fire

    Returns
    -------
    result : (H, W - width + 1, D) float64
    labels : length-D list of str
    """
    H, W, D = firing_map.shape
    W_out = W - width + 1
    # Stack width consecutive slices → (H, W_out, width, D)
    windows = np.stack([firing_map[:, i:i + W_out, :] for i in range(width)], axis=2)
    count_active = np.sum(np.isfinite(windows), axis=2)   # (H, W_out, D)
    last = firing_map[:, width - 1:width - 1 + W_out, :]
    result = np.where(count_active >= threshold, last, np.inf)
    labels = [f"h_cas_ch{d}" for d in range(D)]
    return result, labels


def cassian_pool_vertical(
    firing_map: np.ndarray,
    width: int = 3,
    threshold: int = 2,
) -> tuple[np.ndarray, list[str]]:
    """Cassian gate sliding along each column.

    Parameters
    ----------
    firing_map : (H, W, D) float64 — np.inf = inactive
    width      : window size
    threshold  : minimum number of active positions required to fire

    Returns
    -------
    result : (H - width + 1, W, D) float64
    labels : length-D list of str
    """
    H, W, D = firing_map.shape
    H_out = H - width + 1
    # Stack width consecutive slices → (H_out, W, width, D)
    windows = np.stack([firing_map[i:i + H_out, :, :] for i in range(width)], axis=2)
    count_active = np.sum(np.isfinite(windows), axis=2)   # (H_out, W, D)
    last = firing_map[width - 1:width - 1 + H_out, :, :]
    result = np.where(count_active >= threshold, last, np.inf)
    labels = [f"v_cas_ch{d}" for d in range(D)]
    return result, labels
