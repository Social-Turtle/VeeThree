"""Active-bit cost formulas for all three model families.

Core model: transmitting a value costs proportional to the number of 1-bits
in its encoding.

  Binary/spike signal  → 1 bit per fired signal
  Float32 signal       → 16 expected-active-bits per non-zero value
                         (bit_width/2 = 32/2 = 16; zero is free)

These functions compute total active bits flowing across each layer boundary
during a single forward pass (inference).
"""

import numpy as np


BIT_WIDTH = 32          # float32


# --------------------------------------------------------------------------- #
# CNN active bits
# --------------------------------------------------------------------------- #

def cnn_active_bits(model) -> int:
    """Return active bits accumulated in model's forward hooks.

    Call AFTER a forward pass with cost hooks registered (see
    ConventionalCNN.register_cost_hooks / get_active_bits).

    Hooks fire on every nn.ReLU output and count:
        count_nonzero(activation) × 16  (float32)

    The final output layer is excluded — it is a classification logit,
    not transmitted further.
    """
    return model.get_active_bits()


# --------------------------------------------------------------------------- #
# LUT active bits
# --------------------------------------------------------------------------- #

def lut_active_bits(spike_counts: dict, y_outputs: list) -> int:
    """Compute active bits for one LUT model forward pass.

    Parameters
    ----------
    spike_counts : dict returned by LUTModel.forward()
        spike_counts['total'] = popcount of all binary comparison bits (exact).
    y_outputs    : list of float32 numpy arrays
        Intermediate output vectors from each LUT layer (local + global),
        stored in all_caches['y_outputs']. Final output logits are excluded.

    Returns
    -------
    Total active spikes: binary comparisons + LUT output spikes.

    Note on float32 y outputs: each non-zero entry represents one spike whose
    *timing* is encoded as an FP32 value.  In hardware this is still a single
    spike on a wire, so it costs 1 (not bit_width/2).
    """
    # Binary comparison bits — exact popcount
    total = int(spike_counts['total'])

    # Table-lookup outputs: one spike per non-zero entry (timing encoded as FP32)
    for y in y_outputs:
        total += int(np.count_nonzero(y))

    return total


# --------------------------------------------------------------------------- #
# Feature-engineering active bits
# --------------------------------------------------------------------------- #

def fe_active_bits(stage_signal_counts: list[int]) -> int:
    """Sum active (non-∞) signals across all pipeline stages.

    Each fired signal in the FE pipeline is binary (fired or ∞), so its
    cost is exactly 1 bit.

    Parameters
    ----------
    stage_signal_counts : list of ints
        Count of finite (active) signals at each stage boundary.
        Returned by run_pipeline_with_config() in mnist_pipeline.py.
    """
    return sum(stage_signal_counts)
