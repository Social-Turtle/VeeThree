"""Stage 4 visualization — render a sweep firing map to a PIL Image."""
from __future__ import annotations

import numpy as np
from PIL import Image


def visualize_sweep(
    firing_map: np.ndarray,
    detector_colors: list[tuple],
    scale: int = 8,
) -> Image.Image:
    """Render a sweep firing map.

    Parameters
    ----------
    firing_map : (H, W, D) float64 — np.inf = no firing
    detector_colors : length-D list of (R, G, B) tuples (base color per channel)
    scale : integer upscale factor (each logical pixel → scale×scale block)

    Returns
    -------
    PIL Image of size (W*scale, H*scale) in RGB mode.

    Rendering rules
    ---------------
    - Black background.
    - Global normalization: v_min / v_max over ALL finite values in the map.
      intensity = 1 - (v - v_min) / (v_max - v_min)   (lower value → brighter)
      If v_min == v_max, all finite values render at full brightness.
    - At each (row, col) the channel with the minimum (strongest) finite value
      wins; its base_color is scaled by intensity.
    - Pixels with no finite value in any channel remain black.
    """
    H, W, D = firing_map.shape

    # Global normalization
    finite_mask = np.isfinite(firing_map)
    finite_vals = firing_map[finite_mask]
    if finite_vals.size == 0:
        v_min = v_max = 0.0
        has_finite = False
    else:
        v_min = float(finite_vals.min())
        v_max = float(finite_vals.max())
        has_finite = True

    img = np.zeros((H * scale, W * scale, 3), dtype=np.uint8)

    for row in range(H):
        for col in range(W):
            pixel = firing_map[row, col, :]       # (D,)
            pf_mask = np.isfinite(pixel)
            if not pf_mask.any():
                continue

            # Channel with minimum value wins
            winner_d = int(np.argmin(np.where(pf_mask, pixel, np.inf)))
            v = float(pixel[winner_d])

            if has_finite:
                if v_max == v_min:
                    intensity = 1.0
                else:
                    intensity = 1.0 - (v - v_min) / (v_max - v_min)
            else:
                intensity = 0.0

            base = detector_colors[winner_d]
            rendered = tuple(int(c * intensity) for c in base)

            r0, r1 = row * scale, (row + 1) * scale
            c0, c1 = col * scale, (col + 1) * scale
            img[r0:r1, c0:c1] = rendered

    return Image.fromarray(img, mode="RGB")
