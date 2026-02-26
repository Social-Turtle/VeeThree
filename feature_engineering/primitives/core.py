import numpy as np

def apply_primitive(sequences: np.ndarray) -> np.ndarray:
    """
    sequences: (..., N) float array — last axis is the sequence of length N >= 2.
    Returns: (...,) float array — last element if strictly increasing left-to-right, np.inf otherwise.
    """
    diffs = np.diff(sequences, axis=-1)
    strictly_increasing = np.all(diffs >= 0, axis=-1)
    result = np.where(strictly_increasing, sequences[..., -1], np.inf)
    return result

def apply_primitive_3(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Convenience wrapper: stacks A, B, C along new last axis and calls apply_primitive."""
    sequences = np.stack([A, B, C], axis=-1)
    return apply_primitive(sequences)

def apply_cassian(values: np.ndarray, threshold: int) -> np.ndarray:
    """
    values: (..., K) float array — sequence of K input values.
    threshold: int — activation threshold for input count.
    Returns: (...,) float array — last value if count of non-inf values >= threshold, np.inf otherwise.

    Gates activation based on the number of non-infinite inputs present.
    """
    count = np.sum(~np.isinf(values), axis=-1)
    result = np.where(count >= threshold, values[..., -1], np.inf)
    return result

def apply_or(values: np.ndarray) -> np.ndarray:
    """Convenience wrapper: applies cassian with threshold of 1 (OR gate)."""
    return apply_cassian(values, 1)

def apply_and(values: np.ndarray) -> np.ndarray:
    """Convenience wrapper: applies cassian with threshold equal to K, the number of inputs (AND gate)."""
    num_inputs = values.shape[-1]
    return apply_cassian(values, num_inputs)
