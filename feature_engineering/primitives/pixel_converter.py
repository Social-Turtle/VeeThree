import numpy as np
from .core import apply_primitive_3


def pixels_to_values(first: np.ndarray, center: np.ndarray, last: np.ndarray) -> np.ndarray:
    """
    first, center, last: same-shape float arrays, pixel values in [0, 255].
    Detects strictly-decreasing pixel sequences (first > center > last).
    Bridge: invert pixels (255 - p) so decreasing becomes increasing, then use core primitive.
    Value formula when activated: 255 - (first - last).
    Returns same-shape float array; np.inf where condition not met.
    """
    A = 255.0 - first
    B = 255.0 - center
    C = 255.0 - last

    result = apply_primitive_3(A, B, C)
    # adding a little gaussian noise for tiebreaking.
    base_output = 255.0 - (first - last)
    noise = np.random.normal(loc=3.0, scale=1.0, size=base_output.shape)
    fired = result != np.inf
    output = np.where(fired, base_output+noise, np.inf)
    return output
