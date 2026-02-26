"""
Tests for sweep_detection.py — detector-independence invariant.

Detectors run on the unmodified input: one detector "consuming" a pixel
(advancing its internal state through it) must not prevent any other detector
from matching that same pixel.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from stages.sweep_detection import SweepDetector, sweep_horizontal, sweep_vertical

# Minimal two-label scheme: channel 0 = "v", channel 1 = "h"
LABELS = ["v", "h"]


def _row(width: int, *active: tuple) -> np.ndarray:
    """Return a (1, width, 2) float64 array (1 row, `width` cols, 2 channels).

    active: (col, channel_idx, value) triples for non-inf positions.
    """
    arr = np.full((1, width, 2), np.inf)
    for col, ch, val in active:
        arr[0, col, ch] = val
    return arr


def _col(height: int, *active: tuple) -> np.ndarray:
    """Return a (height, 1, 2) float64 array (1 column, `height` rows).

    active: (row, channel_idx, value) triples for non-inf positions.
    """
    arr = np.full((height, 1, 2), np.inf)
    for row, ch, val in active:
        arr[row, 0, ch] = val
    return arr


# ---------------------------------------------------------------------------
# Independence: a second detector sees every pixel that a first detector
# "consumed" during its own scan.
# ---------------------------------------------------------------------------

class TestDetectorIndependence:

    def test_second_detector_sees_pixel_consumed_by_first_horizontal(self):
        """
        Row layout: v@col0, h@col1
          det_vh = ["v","h"]  fires at col1 (consumed both pixels on the way)
          det_h  = ["h"]      must ALSO fire at col1

        If detectors shared state or modified the input, det_h would see the
        "h" already gone and produce no firing.  Correct behaviour: it fires.
        """
        arr = _row(4, (0, 0, 10.0), (1, 1, 20.0))

        det_vh = SweepDetector(["v", "h"], (255, 0, 0), name="vh")
        det_h  = SweepDetector(["h"],      (0, 255, 0), name="h")

        fm, _ = sweep_horizontal(arr, LABELS, [det_vh, det_h])

        assert np.isfinite(fm[0, 1, 0]), "det_vh must fire at col1"
        assert np.isfinite(fm[0, 1, 1]), (
            "det_h must also fire at col1 — the 'h' pixel was not removed by det_vh"
        )

    def test_detector_order_does_not_affect_independence_horizontal(self):
        """Swapping the list order must not change whether each detector fires."""
        arr = _row(4, (0, 0, 10.0), (1, 1, 20.0))

        det_vh = SweepDetector(["v", "h"], (255, 0, 0), name="vh")
        det_h  = SweepDetector(["h"],      (0, 255, 0), name="h")

        # det_h is listed first this time
        fm, _ = sweep_horizontal(arr, LABELS, [det_h, det_vh])

        assert np.isfinite(fm[0, 1, 0]), "det_h (first) fires at col1"
        assert np.isfinite(fm[0, 1, 1]), (
            "det_vh (second) fires at col1 — det_h did not consume the 'v' pixel"
        )

    def test_second_detector_sees_pixel_consumed_by_first_vertical(self):
        """Same independence property for sweep_vertical (top→bottom)."""
        arr = _col(4, (0, 0, 5.0), (1, 1, 15.0))  # v@row0, h@row1

        det_vh = SweepDetector(["v", "h"], (255, 0, 0), name="vh")
        det_h  = SweepDetector(["h"],      (0, 255, 0), name="h")

        fm, _ = sweep_vertical(arr, LABELS, [det_vh, det_h])

        assert np.isfinite(fm[1, 0, 0]), "det_vh must fire at row1"
        assert np.isfinite(fm[1, 0, 1]), (
            "det_h must also fire at row1 — the 'h' pixel was not removed by det_vh"
        )

    def test_three_detectors_all_see_shared_pixel(self):
        """
        A single 'h' pixel at col2 — three detectors all want it.
        All three must fire independently.
        """
        arr = _row(4, (2, 1, 7.0))  # h@col2

        d1 = SweepDetector(["h"], (255, 0,   0), name="d1")
        d2 = SweepDetector(["h"], (0,   255, 0), name="d2")
        d3 = SweepDetector(["h"], (0,   0, 255), name="d3")

        fm, _ = sweep_horizontal(arr, LABELS, [d1, d2, d3])

        assert np.isfinite(fm[0, 2, 0]), "d1 fires"
        assert np.isfinite(fm[0, 2, 1]), "d2 fires (d1 did not consume the pixel)"
        assert np.isfinite(fm[0, 2, 2]), "d3 fires (neither d1 nor d2 consumed it)"

    def test_input_array_is_not_mutated(self):
        """sweep_horizontal must leave the input array unchanged."""
        arr = _row(4, (0, 0, 10.0), (1, 1, 20.0))
        original = arr.copy()

        det = SweepDetector(["v", "h"], (255, 0, 0), name="vh")
        sweep_horizontal(arr, LABELS, [det])

        np.testing.assert_array_equal(arr, original,
            err_msg="sweep_horizontal must not mutate its input array")


# ---------------------------------------------------------------------------
# Sanity: a single pixel cannot satisfy a multi-element pattern.
# ---------------------------------------------------------------------------

class TestSinglePixelCannotCompleteMultiStepPattern:

    def test_one_pixel_cannot_satisfy_two_step_pattern_horizontal(self):
        arr = _row(4, (2, 0, 7.0))  # single 'v' at col2
        det = SweepDetector(["v", "v"], (255, 0, 0), name="vv")
        fm, _ = sweep_horizontal(arr, LABELS, [det])
        assert not np.any(np.isfinite(fm)), (
            "one 'v' pixel cannot satisfy ['v','v']"
        )

    def test_one_pixel_cannot_satisfy_three_step_pattern_horizontal(self):
        arr = _row(4, (1, 1, 3.0))  # single 'h' at col1
        det = SweepDetector(["h", "h", "h"], (0, 0, 255), name="hhh")
        fm, _ = sweep_horizontal(arr, LABELS, [det])
        assert not np.any(np.isfinite(fm)), (
            "one 'h' pixel cannot satisfy ['h','h','h']"
        )

    def test_one_pixel_cannot_satisfy_two_step_pattern_vertical(self):
        arr = _col(4, (2, 0, 9.0))  # single 'v' at row2
        det = SweepDetector(["v", "v"], (255, 0, 0), name="vv")
        fm, _ = sweep_vertical(arr, LABELS, [det])
        assert not np.any(np.isfinite(fm)), (
            "one 'v' pixel cannot satisfy ['v','v'] vertically"
        )
