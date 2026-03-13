"""Hybrid FE→LUT classifier.

Preprocessing: FE pipeline stages 1–4 (fixed), producing 392-dim features.
Classifier:    small LUT stack trained with SGD (replaces hand-crafted Stage 6).
"""

import numpy as np
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FE_DIR     = os.path.join(_SCRIPT_DIR, 'feature_engineering')
sys.path.insert(0, _FE_DIR)      # feature_engineering/ for stages, experiments, primitives
sys.path.insert(0, _SCRIPT_DIR)  # VeeThree/ first so lut_model and main are found here

from lut_model import LUT, learning_rate, _softmax, N_T, N_C, EMBED_DIM, WARMUP
from experiments.mnist_pipeline import run_pipeline

N_CLASSES = 10
N_GLOBAL  = 2      # global LUT layers  (mirrors lut_model.N_GLOBAL)
FE_DIM    = 392    # 14 × 14 × 2 after FE pipeline


def normalize_inf(arr):
    """Replace ∞ with 5% above the max finite value."""
    finite_vals = arr[np.isfinite(arr)]
    cap = float(finite_vals.max()) * 1.05 if len(finite_vals) > 0 else 1.0
    return np.where(np.isfinite(arr), arr, cap).astype(np.float32)


class FELUTModel:
    """Feature-engineering preprocessing + LUT stack classifier.

    Architecture:
        image (28×28, [0,1])
        → FE pipeline stages 1–4  → (14, 14, 2) float64 with ∞
        → normalize_inf            → (14, 14, 2) float32
        → flatten                  → 392-dim vector
        → N_GLOBAL LUT layers      → EMBED_DIM
        → output LUT               → N_CLASSES logits
    """

    def __init__(self):
        self.global_layers = []
        in_dim = FE_DIM
        for _ in range(N_GLOBAL):
            self.global_layers.append(LUT(in_dim, EMBED_DIM))
            in_dim = EMBED_DIM
        self.output_lut = LUT(EMBED_DIM, N_CLASSES)

    def _preprocess(self, image):
        """Run FE pipeline on image ∈ [0, 1]. Returns 392-dim float32 vector."""
        image_255 = (image * 255.0).astype(np.float64)
        fe_out = run_pipeline(image_255)   # (14, 14, 2)
        return normalize_inf(fe_out).flatten()

    def forward(self, image):
        """Full forward pass.

        Returns (logits, all_caches, spike_counts, seq2s).
        """
        x = self._preprocess(image)

        global_inputs  = []
        global_caches  = []
        global_spikes  = []
        global_comps   = []
        global_outputs = []

        current = x
        for layer in self.global_layers:
            global_inputs.append(current)
            y, cache, spikes, comparisons = layer.forward(current)
            global_caches.append(cache)
            global_spikes.append(spikes)
            global_comps.append(comparisons)
            global_outputs.append(y)
            current = y

        output_input = current
        logits, output_cache, output_spikes, output_comps = (
            self.output_lut.forward(output_input)
        )

        spike_counts = {
            'local':  [],
            'global': global_spikes,
            'output': output_spikes,
            'total':  sum(global_spikes) + output_spikes,
        }
        seq2s = {
            'local':  [],
            'global': global_comps,
            'output': output_comps,
            'total':  sum(global_comps) + output_comps,
        }
        all_caches = {
            'x':             x,
            'global_inputs': global_inputs,
            'global_caches': global_caches,
            'output_input':  output_input,
            'output_cache':  output_cache,
            'y_outputs':     global_outputs,   # for active-signal cost accounting
        }
        return logits, all_caches, spike_counts, seq2s

    def step(self, image, label, t):
        """Forward + backward for one training sample.

        Returns (loss, correct, spike_counts, seq2s).
        """
        lr = learning_rate(t)

        logits, caches, spike_counts, seq2s = self.forward(image)

        probs   = _softmax(logits.astype(np.float64)).astype(np.float32)
        loss    = -float(np.log(probs[label] + 1e-9))
        correct = int(np.argmax(probs) == label)

        grad = probs.copy()
        grad[label] -= 1.0

        # Backward through output LUT
        grad = self.output_lut.backward(
            caches['output_input'], caches['output_cache'], grad, lr
        )

        # Backward through global layers (reversed)
        for i in range(len(self.global_layers) - 1, -1, -1):
            grad = self.global_layers[i].backward(
                caches['global_inputs'][i], caches['global_caches'][i], grad, lr
            )
        # grad now has shape (FE_DIM,); FE pipeline is fixed, discard it.

        return loss, correct, spike_counts, seq2s
