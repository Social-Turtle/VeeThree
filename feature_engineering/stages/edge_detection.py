import numpy as np
from primitives.pixel_converter import pixels_to_values


def edge_detection(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    image: (28, 28) float array, pixel values in [0, 255].
    Returns:
      values:  (28, 28, 8) float64 array — primitive output per direction, np.inf for border/inactive.
      dir_ids: (28, 28, 8) int8 array   — direction ID 0–7 per channel (constant, just np.arange broadcast).
    """
    num_filters = 4
    values = np.full((28, 28, 8), np.inf, dtype=np.float64)

    # Interior pixel slices (rows 1–26, cols 1–26 — the 26×26 non-border region).
    # For each direction, define (first, center, last) as 26×26 vectorized slices.

    # Direction 0 — vertical-down: first=img[r-1,c], center=img[r,c], last=img[r+1,c]
    values[1:27, 1:27, 0] = pixels_to_values(
        image[0:26, 1:27],   # first:  one row above
        image[1:27, 1:27],   # center: current row
        image[2:28, 1:27],   # last:   one row below
    )

    # # Direction 1 — vertical-up: first=img[r+1,c], center=img[r,c], last=img[r-1,c]
    # values[1:27, 1:27, 1] = pixels_to_values(
    #     image[2:28, 1:27],   # first:  one row below
    #     image[1:27, 1:27],   # center: current row
    #     image[0:26, 1:27],   # last:   one row above
    # )

    # Direction 2 — horizontal-right: first=img[r,c-1], center=img[r,c], last=img[r,c+1]
    values[1:27, 1:27, 2] = pixels_to_values(
        image[1:27, 0:26],   # first:  one col to the left
        image[1:27, 1:27],   # center: current col
        image[1:27, 2:28],   # last:   one col to the right
    )

    # # Direction 3 — horizontal-left: first=img[r,c+1], center=img[r,c], last=img[r,c-1]
    # values[1:27, 1:27, 3] = pixels_to_values(
    #     image[1:27, 2:28],   # first:  one col to the right
    #     image[1:27, 1:27],   # center: current col
    #     image[1:27, 0:26],   # last:   one col to the left
    # )

    # # Direction 4 — diagonal-1-fwd: first=img[r-1,c-1], center=img[r,c], last=img[r+1,c+1]
    # values[1:27, 1:27, 4] = pixels_to_values(
    #     image[0:26, 0:26],   # first:  up-left
    #     image[1:27, 1:27],   # center: current
    #     image[2:28, 2:28],   # last:   down-right
    # )

    # # Direction 5 — diagonal-1-bwd: first=img[r+1,c+1], center=img[r,c], last=img[r-1,c-1]
    # values[1:27, 1:27, 5] = pixels_to_values(
    #     image[2:28, 2:28],   # first:  down-right
    #     image[1:27, 1:27],   # center: current
    #     image[0:26, 0:26],   # last:   up-left
    # )

    # # Direction 6 — diagonal-2-fwd: first=img[r-1,c+1], center=img[r,c], last=img[r+1,c-1]
    # values[1:27, 1:27, 6] = pixels_to_values(
    #     image[0:26, 2:28],   # first:  up-right
    #     image[1:27, 1:27],   # center: current
    #     image[2:28, 0:26],   # last:   down-left
    # )

    # # Direction 7 — diagonal-2-bwd: first=img[r+1,c-1], center=img[r,c], last=img[r-1,c+1]
    # values[1:27, 1:27, 7] = pixels_to_values(
    #     image[2:28, 0:26],   # first:  down-left
    #     image[1:27, 1:27],   # center: current
    #     image[0:26, 2:28],   # last:   up-right
    # )

    # dir_ids: (28, 28, 8) int8 array where dir_ids[:, :, d] == d for all d in 0..7.
    # Construct once via broadcast: shape (8,) -> (1, 1, 8) -> broadcast to (28, 28, 8).
    dir_ids = np.broadcast_to(
        np.arange(num_filters, dtype=np.int8)[np.newaxis, np.newaxis, :],
        (28, 28, num_filters),
    ).copy()

    return values, dir_ids
