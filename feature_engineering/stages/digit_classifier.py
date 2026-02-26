"""Stage 6: Digit classification via recursive presence-based templates.

Uses apply_primitive and apply_cassian from primitives/core.py as the
evaluation primitives.

Tree building blocks:

    HPrimitive(*pattern)
        Scans each row of the stage-5 map left→right.  For each row, the
        state machine advances when it finds a pixel whose channel label
        matches the current pattern element.  When all slots fill, the
        collected x-positions are fed into apply_primitive, which checks
        they are strictly increasing — always true for a left→right scan,
        so the primitive fires whenever the full pattern completes.
        Fire value = x-position of the last matched pixel.

        evaluate() → (H,) float64: fire x-position per row, or inf if the
                     pattern never completed for that row.
        score()    → int: number of rows that fired (count of finite values).

    VPrimitive(*pattern)
        Same as HPrimitive but scans each column top→bottom.
        evaluate() → (W,) float64: fire y-position per column, or inf.
        score()    → int: number of columns that fired.

    Cassian(*children, threshold)
        Soft AND gate.  Reduces each child's output to a single "fired"
        scalar: the last finite value in the child's output array, or inf
        if none.  Then calls apply_cassian on those scalars with the given
        threshold, which fires if at least `threshold` scalars are finite.

        evaluate() → float64 scalar: non-inf if ≥ threshold children fired.
        score()    → int: sum of child scores if this node fires, else 0.
                     More primitive rows/cols firing → higher score, which
                     is used to rank digits when multiple templates fire.

Templates are built by composing these classes.  Each digit gets one root
node; the digit with the highest score wins.

STAGE5_CHANNEL_LABELS must match the detector names used in the pipeline
(H_DETECTORS names followed by V_DETECTORS names).
"""
from __future__ import annotations

import numpy as np

from primitives.core import apply_primitive, apply_cassian as _apply_cassian


# Channel labels for the combined Stage-5 map fed into classify().
# Order must match: [h_pool channels..., v_pool channels...]
# These are the .name fields of H_DETECTORS + V_DETECTORS in mnist_pipeline.py.
STAGE5_CHANNEL_LABELS: list[str] = ["h_line", "v_line"]


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _is_present(
    stage5_map: np.ndarray,
    channel_labels: list[str],
    row: int,
    col: int,
    label: str,
) -> bool:
    """Return True if (row, col) has any finite value in a channel named `label`."""
    for ch_idx, ch_label in enumerate(channel_labels):
        if ch_label == label and np.isfinite(stage5_map[row, col, ch_idx]):
            return True
    return False


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class HPrimitive:
    """Horizontal sequence detector: scans each row left→right.

    For each row, the state machine looks for the pattern elements in order.
    Each matched pixel contributes its x-coordinate to the position buffer.
    When all slots fill, apply_primitive checks strict increase (guaranteed
    left→right), and the fire value (last x-position) is recorded for that row.

    Parameters
    ----------
    *pattern : channel labels to match in order, e.g. "h_line", "v_line"
    """

    def __init__(self, *pattern: str):
        self.pattern = pattern

    def evaluate(self, stage5_map: np.ndarray, channel_labels: list[str]) -> np.ndarray:
        """Return (H,) array: fire x-position per row, or inf if pattern did not complete."""
        H, W, _ = stage5_map.shape
        P = len(self.pattern)
        results = np.full(H, np.inf)
        for row in range(H):
            state = 0
            pos_buffer = np.full(P, np.inf)
            for col in range(W):
                if _is_present(stage5_map, channel_labels, row, col, self.pattern[state]):
                    pos_buffer[state] = float(col)
                    state += 1
                    if state == P:
                        # Positions are always increasing left→right, so this always fires.
                        fire_val = apply_primitive(pos_buffer.reshape(1, -1))[0]
                        results[row] = fire_val
                        pos_buffer = np.full(P, np.inf)
                        state = 0
        return results

    def score(self, stage5_map: np.ndarray, channel_labels: list[str]) -> int:
        """Return count of rows where the pattern completed."""
        return int(np.sum(np.isfinite(self.evaluate(stage5_map, channel_labels))))


class VPrimitive:
    """Vertical sequence detector: scans each column top→bottom.

    Mirror of HPrimitive along the column axis.  Fire value = y-position
    of the last matched pixel in that column.

    Parameters
    ----------
    *pattern : channel labels to match in order, e.g. "v_line", "h_line"
    """

    def __init__(self, *pattern: str):
        self.pattern = pattern

    def evaluate(self, stage5_map: np.ndarray, channel_labels: list[str]) -> np.ndarray:
        """Return (W,) array: fire y-position per column, or inf if pattern did not complete."""
        H, W, _ = stage5_map.shape
        P = len(self.pattern)
        results = np.full(W, np.inf)
        for col in range(W):
            state = 0
            pos_buffer = np.full(P, np.inf)
            for row in range(H):
                if _is_present(stage5_map, channel_labels, row, col, self.pattern[state]):
                    pos_buffer[state] = float(row)
                    state += 1
                    if state == P:
                        fire_val = apply_primitive(pos_buffer.reshape(1, -1))[0]
                        results[col] = fire_val
                        pos_buffer = np.full(P, np.inf)
                        state = 0
        return results

    def score(self, stage5_map: np.ndarray, channel_labels: list[str]) -> int:
        """Return count of columns where the pattern completed."""
        return int(np.sum(np.isfinite(self.evaluate(stage5_map, channel_labels))))


class Cassian:
    """Soft AND gate: fires if at least `threshold` children fired.

    Each child's output (an array or scalar) is first reduced to a single
    scalar — the last finite value if any exist, else inf.  apply_cassian
    then counts how many of those scalars are finite, and fires only if
    the count meets the threshold.

    score() returns the sum of all children's scores when this node fires,
    giving a richer signal than just "fired/not fired" for digit ranking.

    Parameters
    ----------
    *children : HPrimitive, VPrimitive, or Cassian nodes
    threshold : minimum number of children that must fire
    """

    def __init__(self, *children: HPrimitive | VPrimitive | Cassian, threshold: int):
        self.children = children
        self.threshold = threshold

    def _reduce_to_scalar(self, child_output) -> float:
        """Collapse a child's array to one scalar: last finite value, or inf."""
        arr = np.atleast_1d(np.asarray(child_output, dtype=float))
        finite_vals = arr[np.isfinite(arr)]
        return float(finite_vals[-1]) if len(finite_vals) > 0 else np.inf

    def evaluate(self, stage5_map: np.ndarray, channel_labels: list[str]) -> float:
        """Return a scalar: non-inf if ≥ threshold children fired, else inf."""
        child_outputs = [c.evaluate(stage5_map, channel_labels) for c in self.children]
        scalars = np.array([self._reduce_to_scalar(o) for o in child_outputs])
        result = _apply_cassian(scalars.reshape(1, -1), self.threshold)
        return float(result.squeeze())

    def score(self, stage5_map: np.ndarray, channel_labels: list[str]) -> int:
        """Sum of child scores when this node fires, else 0."""
        if np.isfinite(self.evaluate(stage5_map, channel_labels)):
            return sum(c.score(stage5_map, channel_labels) for c in self.children)
        return 0


# ---------------------------------------------------------------------------
# Templates  (placeholders — re-derive after inspecting stage-5 output)
# ---------------------------------------------------------------------------

DIGIT_TEMPLATES: dict[int, HPrimitive | VPrimitive | Cassian] = {
    0: Cassian(HPrimitive("h_line", "h_line"), VPrimitive("v_line", "v_line"), threshold=2),
    1: VPrimitive("v_line", "v_line", "v_line"),
    2: Cassian(HPrimitive("h_line"), VPrimitive("v_line"), threshold=2),
    3: HPrimitive("v_line", "v_line"),
    4: Cassian(VPrimitive("v_line"), HPrimitive("h_line"), threshold=2),
    5: Cassian(HPrimitive("h_line"), HPrimitive("h_line"), threshold=2),
    6: Cassian(HPrimitive("h_line"), VPrimitive("v_line"), threshold=2),
    7: Cassian(HPrimitive("h_line"), VPrimitive("v_line"), threshold=2),
    8: VPrimitive("v_line", "v_line"),
    9: Cassian(HPrimitive("h_line"), VPrimitive("v_line"), threshold=2),
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify(
    stage5_map: np.ndarray,
    channel_labels: list[str] | None = None,
    templates: dict[int, HPrimitive | VPrimitive | Cassian] | None = None,
) -> tuple[int, dict[int, int]]:
    """Classify one image from its Stage-5 feature map.

    Parameters
    ----------
    stage5_map     : (H, W, D) float64 from Stage 5
    channel_labels : length-D list of label strings; defaults to STAGE5_CHANNEL_LABELS
    templates      : override for DIGIT_TEMPLATES

    Returns
    -------
    predicted : int 0–9, or -1 if no template fires
    scores    : dict digit → score (0 = did not fire)
    """
    if channel_labels is None:
        channel_labels = STAGE5_CHANNEL_LABELS
    if templates is None:
        templates = DIGIT_TEMPLATES

    scores = {
        digit: node.score(stage5_map, channel_labels)
        for digit, node in templates.items()
    }

    firing = {d: s for d, s in scores.items() if s > 0}
    predicted = max(firing, key=firing.__getitem__) if firing else -1
    return predicted, scores
