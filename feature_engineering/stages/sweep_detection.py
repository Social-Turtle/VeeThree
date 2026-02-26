"""Stage 4: Sweep Detection — direction-sensitive presence collectors.

Each SweepDetector holds a pattern — an ordered list of direction labels
(e.g. ["h", "v", "h"]).  Scanning left→right (horizontal) or top→bottom
(vertical), the detector advances its state when it sees a pixel that has
any finite value in a channel matching the current pattern element.  Value
ordering is NOT checked; only presence in the right channel matters.

When all pattern slots are filled the detector fires, recording the value
at the last consumed pixel, clears its slots, and continues along the same
row / column.

Both sweep functions accept a `view` parameter (default 1).  When view > 1,
consecutive rows (horizontal) or columns (vertical) are grouped into
non-overlapping bands of that width.  A band position is considered active
if ANY pixel within the band has the required channel active.  This lets
signals that span slightly different rows (or columns) still count as one.
Example with view=2 and two input rows:
    row 0:  1 0 1
    row 1:  0 1 0
    band:   1 1 1   <- OR across both rows, treated as one logical row
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from primitives.core import apply_cassian

# Channel labels for Stage-3 output — maps each channel index to a direction.
#   channels 0,1 = vertical; 2,3 = horizontal; 4,5 = diagonal-1; 6,7 = diagonal-2
STAGE3_CHANNEL_LABELS: list[str] = ["v", "v", "h", "h", "d1", "d1", "d2", "d2"]


@dataclass
class SweepDetector:
    pattern: list[str]   # ordered sequence of direction labels, e.g. ["h", "v", "h"]
    output_color: tuple  # RGB base color for visualization, e.g. (0, 0, 255)
    name: str = ""


@dataclass
class CassianSweepDetector:
    """Sliding-window Cassian sweep detector.

    Instead of matching an ordered sequence, slides a window of
    `window_size` pixels along each row (horizontal) or column
    (vertical).  At each position, counts how many pixels in the
    window have the target direction active.  Fires if the count
    meets `threshold`, using apply_cassian from core.py.

    The fire value is the minimum finite value across all active
    pixels in the window.
    """
    direction: str       # single direction label to look for, e.g. "h"
    window_size: int     # number of consecutive pixels in the window
    threshold: int       # minimum active pixels to fire
    output_color: tuple  # RGB base color for visualization
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
    view: int = 1,
    scan_view: int = 1,
) -> tuple[np.ndarray, list[str]]:
    """Horizontal sweep (left→right), with two independent compression parameters.

    `view` groups consecutive rows into bands (cross-sectional compression): a
    column position is active for a band if ANY row in that band has the required
    channel active.

    `scan_view` coarsens the output column coordinate: the state machine still
    scans every individual column, but a fire at column c is recorded at output
    column c // scan_view.  If two fires land in the same output bin the minimum
    value is kept (strongest signal wins).

    Parameters
    ----------
    input_values   : (H, W, C) float64 — np.inf = inactive
    channel_labels : length-C list mapping each channel index to a direction label
    detectors      : list of SweepDetector
    view           : row-band height for cross-sectional compression (default 1)
    scan_view      : output column bin size for scan-direction compression (default 1)

    Returns
    -------
    firing_map : (H // view, W // scan_view, D) float64
    output_channel_labels : list[str], length D
    """
    H, W, _ = input_values.shape
    D = len(detectors)
    n_row_bands = H // view
    n_col_bins  = W // scan_view
    firing_map = np.full((n_row_bands, n_col_bins, D), np.inf)

    for d_idx, detector in enumerate(detectors):
        pattern = detector.pattern
        P = len(pattern)
        for band_idx in range(n_row_bands):
            row_start = band_idx * view
            row_end   = row_start + view
            state = 0  # index into pattern; counts inputs collected so far
            for col in range(W):
                band_active = any(
                    _is_active(input_values, channel_labels, r, col, pattern[state])
                    for r in range(row_start, row_end)
                )
                if band_active:
                    state += 1
                    if state == P:
                        band_vals   = [_value_at(input_values, r, col) for r in range(row_start, row_end)]
                        finite_vals = [v for v in band_vals if np.isfinite(v)]
                        value       = min(finite_vals) if finite_vals else np.inf
                        out_col = col // scan_view
                        # Keep the minimum if multiple fires land in the same output bin
                        firing_map[band_idx, out_col, d_idx] = min(
                            firing_map[band_idx, out_col, d_idx], value
                        )
                        state = 0  # clear inputs and continue along the row

    output_labels = [
        d.name if d.name else f"detector_{i}" for i, d in enumerate(detectors)
    ]
    return firing_map, output_labels


def sweep_vertical(
    input_values: np.ndarray,
    channel_labels: list[str],
    detectors: list[SweepDetector],
    view: int = 1,
    scan_view: int = 1,
) -> tuple[np.ndarray, list[str]]:
    """Vertical sweep (top→bottom), with two independent compression parameters.

    `view` groups consecutive columns into bands (cross-sectional compression): a
    row position is active for a band if ANY column in that band has the required
    channel active.

    `scan_view` coarsens the output row coordinate: the state machine still scans
    every individual row, but a fire at row r is recorded at output row r // scan_view.
    If two fires land in the same output bin the minimum value is kept.

    Parameters
    ----------
    input_values   : (H, W, C) float64 — np.inf = inactive
    channel_labels : length-C list mapping each channel index to a direction label
    detectors      : list of SweepDetector
    view           : column-band width for cross-sectional compression (default 1)
    scan_view      : output row bin size for scan-direction compression (default 1)

    Returns
    -------
    firing_map : (H // scan_view, W // view, D) float64
    output_channel_labels : list[str], length D
    """
    H, W, _ = input_values.shape
    D = len(detectors)
    n_col_bands = W // view
    n_row_bins  = H // scan_view
    firing_map = np.full((n_row_bins, n_col_bands, D), np.inf)

    for d_idx, detector in enumerate(detectors):
        pattern = detector.pattern
        P = len(pattern)
        for band_idx in range(n_col_bands):
            col_start = band_idx * view
            col_end   = col_start + view
            state = 0  # index into pattern; counts inputs collected so far
            for row in range(H):
                band_active = any(
                    _is_active(input_values, channel_labels, row, c, pattern[state])
                    for c in range(col_start, col_end)
                )
                if band_active:
                    state += 1
                    if state == P:
                        band_vals   = [_value_at(input_values, row, c) for c in range(col_start, col_end)]
                        finite_vals = [v for v in band_vals if np.isfinite(v)]
                        value       = min(finite_vals) if finite_vals else np.inf
                        out_row = row // scan_view
                        # Keep the minimum if multiple fires land in the same output bin
                        firing_map[out_row, band_idx, d_idx] = min(
                            firing_map[out_row, band_idx, d_idx], value
                        )
                        state = 0  # clear inputs and continue along the column

    output_labels = [
        d.name if d.name else f"detector_{i}" for i, d in enumerate(detectors)
    ]
    return firing_map, output_labels


# ---------------------------------------------------------------------------
# Cassian sweep — sliding-window threshold detection
# ---------------------------------------------------------------------------

def cassian_sweep_horizontal(
    input_values: np.ndarray,
    channel_labels: list[str],
    detectors: list[CassianSweepDetector],
    view: int = 1,
    scan_view: int = 1,
) -> tuple[np.ndarray, list[str]]:
    """Horizontal Cassian sweep (sliding window left→right).

    For each detector, slides a window of `window_size` columns across
    each row-band.  At each window position, collects the values from
    pixels that have the target direction active, then feeds them to
    apply_cassian.  If at least `threshold` pixels in the window are
    active, the detector fires with the min finite value in the window.

    Parameters
    ----------
    input_values   : (H, W, C) float64
    channel_labels : length-C label list
    detectors      : list of CassianSweepDetector
    view           : row-band height (cross-sectional compression)
    scan_view      : output column bin size

    Returns
    -------
    firing_map : (H // view, W // scan_view, D) float64
    output_channel_labels : list[str], length D
    """
    H, W, _ = input_values.shape
    D = len(detectors)
    n_row_bands = H // view
    n_col_bins = W // scan_view
    firing_map = np.full((n_row_bands, n_col_bins, D), np.inf)

    for d_idx, det in enumerate(detectors):
        ws = det.window_size
        for band_idx in range(n_row_bands):
            row_start = band_idx * view
            row_end = row_start + view

            for col in range(W - ws + 1):
                # Collect values from the window for this band.
                window_vals = []
                for wc in range(col, col + ws):
                    for r in range(row_start, row_end):
                        if _is_active(input_values, channel_labels, r, wc, det.direction):
                            window_vals.append(_value_at(input_values, r, wc))
                            break  # one active row in band is enough for this column
                    else:
                        window_vals.append(np.inf)

                # Apply Cassian: fire if >= threshold values are finite.
                vals_arr = np.array(window_vals)
                result = apply_cassian(vals_arr.reshape(1, -1), det.threshold)
                if np.isfinite(result.squeeze()):
                    finite_in_window = vals_arr[np.isfinite(vals_arr)]
                    value = float(finite_in_window.min())
                    out_col = (col + ws // 2) // scan_view  # center of window
                    out_col = min(out_col, n_col_bins - 1)
                    firing_map[band_idx, out_col, d_idx] = min(
                        firing_map[band_idx, out_col, d_idx], value
                    )

    output_labels = [
        d.name if d.name else f"cassian_{i}" for i, d in enumerate(detectors)
    ]
    return firing_map, output_labels


def cassian_sweep_vertical(
    input_values: np.ndarray,
    channel_labels: list[str],
    detectors: list[CassianSweepDetector],
    view: int = 1,
    scan_view: int = 1,
) -> tuple[np.ndarray, list[str]]:
    """Vertical Cassian sweep (sliding window top→bottom).

    Mirror of cassian_sweep_horizontal along the column axis.

    Parameters
    ----------
    input_values   : (H, W, C) float64
    channel_labels : length-C label list
    detectors      : list of CassianSweepDetector
    view           : column-band width (cross-sectional compression)
    scan_view      : output row bin size

    Returns
    -------
    firing_map : (H // scan_view, W // view, D) float64
    output_channel_labels : list[str], length D
    """
    H, W, _ = input_values.shape
    D = len(detectors)
    n_col_bands = W // view
    n_row_bins = H // scan_view
    firing_map = np.full((n_row_bins, n_col_bands, D), np.inf)

    for d_idx, det in enumerate(detectors):
        ws = det.window_size
        for band_idx in range(n_col_bands):
            col_start = band_idx * view
            col_end = col_start + view

            for row in range(H - ws + 1):
                # Collect values from the window for this band.
                window_vals = []
                for wr in range(row, row + ws):
                    for c in range(col_start, col_end):
                        if _is_active(input_values, channel_labels, wr, c, det.direction):
                            window_vals.append(_value_at(input_values, wr, c))
                            break  # one active col in band is enough for this row
                    else:
                        window_vals.append(np.inf)

                # Apply Cassian: fire if >= threshold values are finite.
                vals_arr = np.array(window_vals)
                result = apply_cassian(vals_arr.reshape(1, -1), det.threshold)
                if np.isfinite(result.squeeze()):
                    finite_in_window = vals_arr[np.isfinite(vals_arr)]
                    value = float(finite_in_window.min())
                    out_row = (row + ws // 2) // scan_view  # center of window
                    out_row = min(out_row, n_row_bins - 1)
                    firing_map[out_row, band_idx, d_idx] = min(
                        firing_map[out_row, band_idx, d_idx], value
                    )

    output_labels = [
        d.name if d.name else f"cassian_{i}" for i, d in enumerate(detectors)
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
