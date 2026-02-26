"""Stage 6: Digit classification via priority-ordered presence-based rules.

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

Classification uses a priority-ordered rule list.  Each rule pairs a
template (tree of primitives/cassians) with a digit label.  The first
rule whose template fires (score > 0) wins.  More specific / complex
rules appear earlier so they are checked before simpler, broader ones.
This sidesteps the "simplest template always wins" problem: a broad
pattern can serve as a fallback without stealing matches from the
specific patterns above it.

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
# Priority-ordered rules
# ---------------------------------------------------------------------------
# Each rule is (digit, template).  First template that fires wins.
# Ordered from most specific / rare to most general / common.
#
# Primitive firing profile on the 10 reference digits (single example each):
#
#       H(h)  H(v)  H(v,h)  H(v,v)  H(h,h)  V(h)  V(v)  V(h,v)  V(v,h)  V(h,h)  V(h,v,h)
# 0:     5     3      2       3       0       4     6      2       0       1        0
# 1:     2     0      0       0       0       2     0      0       0       0        0
# 2:     6     3      0       1       1       4     4      0       1       2        0
# 3:     9     7      6       3       1       6     8      4       2       3        2
# 4:     5     4      1       0       0       3     4      1       0       1        0
# 5:     5     4      1       1       1       6     6      4       0       0        0
# 6:     7     4      1       3       2       5     8      4       1       4        1
# 7:     5     2      0       0       0       4     2      1       0       1        0
# 8:     3     4      0       3       0       3     8      1       1       0        0
# 9:     6     5      2       1       0       3     6      2       0       1        0

PRIORITY_RULES: list[tuple[int, HPrimitive | VPrimitive | Cassian]] = [
    # --- Digit 3: uniquely high H(v,h) AND V(h,v,h) ---
    # Only digit 3 has both of these firing together (score 8).
    (3, Cassian(
        HPrimitive("v_line", "h_line"),            # 6 rows — uniquely high
        VPrimitive("h_line", "v_line", "h_line"),  # 2 cols — only 3 and 6 have this
        threshold=2,
    )),

    # --- Digit 6: H(h,h) AND V(h,h) AND V(h,v,h) — all three required ---
    # After 3 removed, V(h,v,h) fires only for 6 (=1). Combined with
    # H(h,h) and V(h,h) this is very selective for 6.
    (6, Cassian(
        HPrimitive("h_line", "h_line"),            # rows with two h-bars
        VPrimitive("h_line", "h_line"),            # cols with two h-bars
        VPrimitive("h_line", "v_line", "h_line"),  # col with h→v→h (6's structure)
        threshold=3,
    )),

    # --- Digit 8: H(v,v) AND V(v,h) — two v-sides + v-then-h columns ---
    # After 3 and 6 removed, only 8 has both H(v,v)≥3 and V(v,h)≥1.
    (8, Cassian(
        HPrimitive("v_line", "v_line"),    # two v-sides per row
        VPrimitive("v_line", "h_line"),    # v then h going down
        threshold=2,
    )),

    # --- Digit 0: H(v,v) AND V(h,h) AND V(v) — oval shape ---
    # 3-child AND: two v-sides + h-bars top/bottom + columns with v.
    # After 3, 6, 8 removed, fires for 0, 2, 9. Still imperfect —
    # 0 vs 9 is the hardest pair (see note below).
    (0, Cassian(
        HPrimitive("v_line", "v_line"),    # two v-sides
        VPrimitive("h_line", "h_line"),    # top and bottom h-bars
        VPrimitive("v_line"),              # columns with v presence
        threshold=3,
    )),

    # --- Digit 5: V(h,v) AND H(h,h) — h-then-v columns + two h-bars ---
    # After 3 and 6 removed, fires mainly for 5.
    (5, Cassian(
        VPrimitive("h_line", "v_line"),    # cols with h→v vertically
        HPrimitive("h_line", "h_line"),    # rows with two h-bars
        threshold=2,
    )),

    # --- Digit 2: V(v,v) AND V(h,h) — v_line pairs + two h-bars in columns ---
    # After 3 and 6 removed, V(v,v) fires only for 2 (=1).
    # Combined with V(h,h) which 2 also has (=2), this is selective.
    (2, Cassian(
        VPrimitive("v_line", "v_line"),    # two v_lines in a column
        VPrimitive("h_line", "h_line"),    # two h-bars in a column
        threshold=2,
    )),

    # --- Digit 4: H(v,h) AND H(v) — v-then-h rows + rows with v ---
    # 4 has H(v,h)=1 and H(v)=4. Checked before 9 so that 4 is caught
    # first; 9 falls through to its own broader rule below.
    (4, Cassian(
        HPrimitive("v_line", "h_line"),    # at least 1 row with v then h
        HPrimitive("v_line"),              # rows with v_line
        threshold=2,
    )),

    # --- Digit 9: H(v,h) AND V(h,v) — v-then-h in rows + h-then-v in cols ---
    # After 3, 6, 0, 5, 4 removed. 9 has H(v,h)=2 and V(h,v)=2.
    (9, Cassian(
        HPrimitive("v_line", "h_line"),    # v then h in a row
        VPrimitive("h_line", "v_line"),    # h then v in a column
        threshold=2,
    )),

    # --- Digit 7: V(h,v) — at least one column has h then v going down ---
    # Digit 7 has V(h,v)=1. Digit 1 has V(h,v)=0. This separates them.
    (7, VPrimitive("h_line", "v_line")),

    # --- Digit 1: fallback ---
    # 1 is the only digit with zero v_line. All rules above require
    # v_line in some form, so 1 falls through everything.
    # We use H("h_line") as a catch-all: fires if any h_line exists.
    (1, HPrimitive("h_line")),
]

# Keep DIGIT_TEMPLATES around for backward compatibility / inspection.
DIGIT_TEMPLATES: dict[int, HPrimitive | VPrimitive | Cassian] = {
    digit: template for digit, template in PRIORITY_RULES
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify(
    stage5_map: np.ndarray,
    channel_labels: list[str] | None = None,
    rules: list[tuple[int, HPrimitive | VPrimitive | Cassian]] | None = None,
) -> tuple[int, dict[int, int]]:
    """Classify one image from its Stage-5 feature map.

    Uses priority-ordered rules: the first rule whose template fires
    (score > 0) determines the predicted digit.

    Parameters
    ----------
    stage5_map     : (H, W, D) float64 from Stage 4
    channel_labels : length-D list of label strings; defaults to STAGE5_CHANNEL_LABELS
    rules          : override for PRIORITY_RULES

    Returns
    -------
    predicted : int 0–9, or -1 if no template fires
    scores    : dict digit → score (0 = did not fire)
    """
    if channel_labels is None:
        channel_labels = STAGE5_CHANNEL_LABELS
    if rules is None:
        rules = PRIORITY_RULES

    # Compute all scores for reporting.
    scores: dict[int, int] = {}
    for digit, template in rules:
        scores[digit] = template.score(stage5_map, channel_labels)

    # First rule that fires wins.
    predicted = -1
    for digit, _ in rules:
        if scores[digit] > 0:
            predicted = digit
            break

    return predicted, scores
