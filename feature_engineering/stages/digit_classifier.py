"""Stage 6: Digit classification using per-digit Cassian AND gates.

Each digit is characterized by a template — a list of (row, col, channel)
positions that must ALL be active in the Stage-5 combined map for that digit
to fire.  The digit whose Cassian output value is highest among those that
fire is the predicted class ("highest lowest value" selection rule).

Combined map channels (passed from pipeline):
  ch 0 = h_pool ch 0  (h_line detector, horizontal Cassian)
  ch 1 = h_pool ch 1  (v_line detector, horizontal Cassian)
  ch 2 = v_pool ch 0  (h_line detector, vertical Cassian)
  ch 3 = v_pool ch 1  (v_line detector, vertical Cassian)

Templates were derived by inspecting actual Stage-5 firing positions for
each digit class and selecting pairs that are unique to that class.
"""
from __future__ import annotations

import numpy as np

# ── Templates derived from Stage-5 firing analysis ───────────────────────────
DIGIT_TEMPLATES: dict[int, list[tuple[int, int, int]]] = {
    # h_line Cassian fires at col 9, rows 5-6 (right side of oval)
    0: [(5,  9, 0), (6,  9, 0)],
    # v_pool h_line Cassian fires at col 7, rows 5-6 (vertical stroke)
    1: [(5,  7, 2), (6,  7, 2)],
    # v_line Cassian fires at row 10, cols 9-10 (bottom horizontal bar)
    2: [(10, 9, 1), (10, 10, 1)],
    # v_line Cassian fires at row 7, cols 7-8 (right bumps)
    3: [(7,  7, 1), (7,  8, 1)],
    # single h_pool v_line fire + v_pool top-right vertical stroke
    4: [(9,  4, 1), (2, 10, 2)],
    # v_line Cassian fires at row 3, cols 9-10 (top-right arc)
    5: [(3,  9, 1), (3, 10, 1)],
    # h_line Cassian fires at col 4, rows 7-8 (lower-left vertical)
    6: [(7,  4, 0), (8,  4, 0)],
    # v_line Cassian fires at row 4, cols 2-3 (top horizontal bar)
    7: [(4,  2, 1), (4,  3, 1)],
    # v_line Cassian fires at row 10, cols 1-2 (bottom-left of lower loop)
    8: [(10, 1, 1), (10, 2, 1)],
    # v_pool h_line Cassians: left stem + right curve
    9: [(4,  5, 2), (5,  8, 2)],
}


def _collect(stage5_map: np.ndarray, template: list[tuple[int, int, int]]) -> np.ndarray:
    """Extract values at template positions; out-of-bounds → np.inf."""
    H, W, D = stage5_map.shape
    vals = []
    for row, col, ch in template:
        if 0 <= row < H and 0 <= col < W and 0 <= ch < D:
            vals.append(float(stage5_map[row, col, ch]))
        else:
            vals.append(np.inf)
    return np.array(vals, dtype=np.float64)


def classify(
    stage5_map: np.ndarray,
    templates: dict[int, list[tuple[int, int, int]]] | None = None,
) -> tuple[int, dict[int, float]]:
    """Classify one image from its Stage-5 feature map.

    For each digit template, applies a Cassian AND gate (all K positions must
    be active).  The digit whose Cassian output value is highest (finite) wins.

    Parameters
    ----------
    stage5_map : (H, W, D) float64 from Stage 5
    templates  : override for DIGIT_TEMPLATES

    Returns
    -------
    predicted : int 0–9, or -1 if no digit fires
    scores    : dict digit → Cassian output value (np.inf = did not fire)
    """
    if templates is None:
        templates = DIGIT_TEMPLATES

    scores: dict[int, float] = {}
    for digit, template in templates.items():
        vals = _collect(stage5_map, template)
        # Cassian AND: all K must be finite; output = last value
        scores[digit] = float(vals[-1]) if np.all(np.isfinite(vals)) else np.inf

    finite = {d: s for d, s in scores.items() if np.isfinite(s)}
    predicted = max(finite, key=finite.__getitem__) if finite else -1
    return predicted, scores
