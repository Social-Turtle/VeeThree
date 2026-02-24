import numpy as np


def apply_primitive(sequences: np.ndarray) -> np.ndarray:
    """
    sequences: (..., N) float array — last axis is the sequence of length N >= 2.
    Returns: (...,) float array — last element if strictly increasing left-to-right, np.inf otherwise.
    """
    diffs = np.diff(sequences, axis=-1)
    strictly_increasing = np.all(diffs > 0, axis=-1)
    result = np.where(strictly_increasing, sequences[..., -1], np.inf)
    return result


def apply_primitive_3(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Convenience wrapper: stacks A, B, C along new last axis and calls apply_primitive."""
    sequences = np.stack([A, B, C], axis=-1)
    return apply_primitive(sequences)
