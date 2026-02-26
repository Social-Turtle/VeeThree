"""Stage 4: Sweep Detection — direction-sensitive presence collectors.

Each SweepDetector holds a pattern — an ordered list of direction labels
(e.g. ["h", "v", "h"]).  Scanning left→right (horizontal) or top→bottom
(vertical), the detector advances its state when it sees a pixel that has
any finite value in a channel matching the current pattern element.  Value
ordering is NOT checked; only presence in the right channel matters.

When all pattern slots are filled the detector fires, recording the value
at the last consumed pixel, clears its slots, and continues along the same
row / column.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Channel labels for Stage-3 output — maps each channel index to a direction.
#   channels 0,1 = vertical; 2,3 = horizontal; 4,5 = diagonal-1; 6,7 = diagonal-2
STAGE3_CHANNEL_LABELS: list[str] = ["v", "v", "h", "h", "d1", "d1", "d2", "d2"]


@dataclass
class SweepDetector:
    pattern: list[str]   # ordered sequence of direction labels, e.g. ["h", "v", "h"]
    output_color: tuple  # RGB base color for visualization, e.g. (0, 0, 255)
    name: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_active(
    input_values: np.ndarray,
    channel_labels: list[str],
    row: int,
    col: int,
    direction: str,
) -> bool:
    """Return True if (row, col) has any finite value in a channel labeled `direction`.

    This is pure presence detection — value ordering is not considered.
    """
    for ch_idx, label in enumerate(channel_labels):
        if label == direction and np.isfinite(input_values[row, col, ch_idx]):
            return True
    return False


def _value_at(input_values: np.ndarray, row: int, col: int) -> float:
    """Return the minimum finite value at (row, col) across all channels.

    After Stage-3 each pixel has at most one finite channel, so this equals
    that channel's value.  For downstream stages it returns the strongest
    (minimum) signal present at that pixel.
    """
    pixel = input_values[row, col, :]
    finite_mask = np.isfinite(pixel)
    if not finite_mask.any():
        return np.inf
    return float(pixel[finite_mask].min())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sweep_horizontal(
    input_values: np.ndarray,
    channel_labels: list[str],
    detectors: list[SweepDetector],
) -> tuple[np.ndarray, list[str]]:
    """Horizontal sweep (left→right per row).

    Parameters
    ----------
    input_values   : (H, W, C) float64 — np.inf = inactive
    channel_labels : length-C list mapping each channel index to a direction label
    detectors      : list of SweepDetector

    Returns
    -------
    firing_map : (H, W, D) float64
        firing_map[r, c, d] = value at the last consumed pixel when detector d
        fires at position (r, c); np.inf otherwise.
    output_channel_labels : list[str], length D
    """
    H, W, _ = input_values.shape
    D = len(detectors)
    firing_map = np.full((H, W, D), np.inf)

    for d_idx, detector in enumerate(detectors):
        pattern = detector.pattern
        P = len(pattern)
        for row in range(H):
            state = 0  # index into pattern; counts inputs collected so far
            for col in range(W):
                if _is_active(input_values, channel_labels, row, col, pattern[state]):
                    state += 1
                    if state == P:
                        firing_map[row, col, d_idx] = _value_at(input_values, row, col)
                        state = 0  # clear inputs and continue along the row

    output_labels = [
        d.name if d.name else f"detector_{i}" for i, d in enumerate(detectors)
    ]
    return firing_map, output_labels


def sweep_vertical(
    input_values: np.ndarray,
    channel_labels: list[str],
    detectors: list[SweepDetector],
) -> tuple[np.ndarray, list[str]]:
    """Vertical sweep (top→bottom per column).

    Parameters
    ----------
    input_values   : (H, W, C) float64 — np.inf = inactive
    channel_labels : length-C list mapping each channel index to a direction label
    detectors      : list of SweepDetector

    Returns
    -------
    firing_map : (H, W, D) float64
    output_channel_labels : list[str], length D
    """
    H, W, _ = input_values.shape
    D = len(detectors)
    firing_map = np.full((H, W, D), np.inf)

    for d_idx, detector in enumerate(detectors):
        pattern = detector.pattern
        P = len(pattern)
        for col in range(W):
            state = 0  # index into pattern; counts inputs collected so far
            for row in range(H):
                if _is_active(input_values, channel_labels, row, col, pattern[state]):
                    state += 1
                    if state == P:
                        firing_map[row, col, d_idx] = _value_at(input_values, row, col)
                        state = 0  # clear inputs and continue along the column

    output_labels = [
        d.name if d.name else f"detector_{i}" for i, d in enumerate(detectors)
    ]
    return firing_map, output_labels


def tighten_columns(firing_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop all-inf columns.

    Returns
    -------
    tightened          : (H, W', D)  where W' = number of active columns
    active_col_indices : (W',) int array of kept column indices
    """
    active_mask = np.any(np.isfinite(firing_map), axis=(0, 2))  # shape (W,)
    active_cols = np.where(active_mask)[0]
    return firing_map[:, active_cols, :], active_cols


def tighten_rows(firing_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop all-inf rows.

    Returns
    -------
    tightened          : (H', W, D)  where H' = number of active rows
    active_row_indices : (H',) int array of kept row indices
    """
    active_mask = np.any(np.isfinite(firing_map), axis=(1, 2))  # shape (H,)
    active_rows = np.where(active_mask)[0]
    return firing_map[active_rows, :, :], active_rows
