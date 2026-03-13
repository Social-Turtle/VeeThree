"""Hybrid Edge-LUT classifier.

Preprocessing: Stage 1 directional edge detection (2 active channels),
               applied to the 26×26 interior of each 28×28 image.
Classifier:    same spatial-grid LUT stack as LUTModel, adapted for 26×26 input.
"""

import numpy as np
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FE_DIR     = os.path.join(_SCRIPT_DIR, 'feature_engineering')
sys.path.insert(0, _FE_DIR)      # feature_engineering/ for stages and primitives
sys.path.insert(0, _SCRIPT_DIR)  # VeeThree/ first so lut_model and main are found here

from lut_model import (
    LUT, learning_rate, _Up_vec, _softmax,
    N_T, N_C, EMBED_DIM, WARMUP, N_GRID, N_LOCAL, N_GLOBAL,
)
from stages.edge_detection import edge_detection

N_CLASSES   = 10
EDGE_STRIDE = 1          # subsampling stride within each cell patch
_ACTIVE_DIRS = [0, 2]   # dir 0 = vertical-down, dir 2 = horizontal-right


def normalize_inf(arr):
    """Replace ∞ with 5% above the max finite value."""
    finite_vals = arr[np.isfinite(arr)]
    cap = float(finite_vals.max()) * 1.05 if len(finite_vals) > 0 else 1.0
    return np.where(np.isfinite(arr), arr, cap).astype(np.float32)


class EdgeLUTModel:
    """Spatial-grid LUT classifier with edge-detection preprocessing.

    Architecture:
        image (28×28, [0,1])
        → edge_detection          → (28, 28, 8)  (dirs 0 & 2 active, rest ∞)
        → trim border [1:27,1:27] → (26, 26, 8)
        → take dirs [0, 2]        → (26, 26, 2)
        → normalize_inf           → (26, 26, 2) float32
        → split into N_GRID×N_GRID cells (cell = 26 // N_GRID)
        → N_LOCAL LUT layers per region (weight-shared)
        → concat all region embeddings
        → N_GLOBAL LUT layers
        → output LUT              → N_CLASSES logits

    seq2s gains an 'edge' key counting comparisons used from edge detection.
    """

    def __init__(self, stride=EDGE_STRIDE):
        self.stride   = stride
        self.cell     = 26 // N_GRID                          # e.g. N_GRID=4 → cell=6
        self.sampled  = len(range(0, self.cell, stride))      # pixels per side after striding
        region_size   = self.sampled * self.sampled * 2       # ×2 for two active channels
        n_regions     = N_GRID * N_GRID

        # Local LUT layers (weight-shared across all regions)
        self.local_layers = []
        in_dim = region_size
        for _ in range(N_LOCAL):
            self.local_layers.append(LUT(in_dim, EMBED_DIM))
            in_dim = EMBED_DIM

        # Global LUT layers on concatenated region embeddings
        merge_dim = n_regions * EMBED_DIM
        self.global_layers = []
        in_dim = merge_dim
        for _ in range(N_GLOBAL):
            self.global_layers.append(LUT(in_dim, EMBED_DIM))
            in_dim = EMBED_DIM

        self.output_lut = LUT(EMBED_DIM, N_CLASSES)

    def _preprocess(self, image):
        """Run edge detection and return normalized (26, 26, 2) float32 array."""
        image_255 = (image * 255.0).astype(np.float64)
        values, _ = edge_detection(image_255)     # (28, 28, 8)
        trimmed   = values[1:27, 1:27, :]         # (26, 26, 8)
        active    = trimmed[:, :, _ACTIVE_DIRS]   # (26, 26, 2)
        return normalize_inf(active)              # (26, 26, 2) float32

    def _split_regions(self, edge_arr):
        """Split (26, 26, 2) edge map into N_GRID² flattened region vectors."""
        cell    = self.cell
        stride  = self.stride
        regions = []
        for row in range(N_GRID):
            for col in range(N_GRID):
                patch = edge_arr[
                    row * cell:(row + 1) * cell,
                    col * cell:(col + 1) * cell,
                    :
                ]                                               # (cell, cell, 2)
                regions.append(
                    patch[::stride, ::stride, :].flatten().astype(np.float32)
                )
        return regions   # list of N_GRID² arrays, each length sampled² × 2

    def forward(self, image):
        """Full forward pass.

        Returns (logits, all_caches, spike_counts, seq2s).
        """
        edge_arr = self._preprocess(image)
        regions  = self._split_regions(edge_arr)

        # --- Local layers (per region, weight-shared) ---
        local_inputs  = []
        local_caches  = []
        local_outputs = []
        local_spikes  = []
        local_comps   = []
        global_spikes = []
        global_comps  = []

        current_regions = regions
        for layer in self.local_layers:
            layer_inputs    = list(current_regions)
            layer_caches    = []
            layer_outputs   = []
            layer_spike_sum = 0
            layer_comp_sum  = 0
            for x in current_regions:
                y, cache, spikes, comparisons = layer.forward(x)
                layer_caches.append(cache)
                layer_outputs.append(y)
                layer_spike_sum += spikes
                layer_comp_sum  += comparisons
            local_inputs.append(layer_inputs)
            local_caches.append(layer_caches)
            local_outputs.append(layer_outputs)
            local_spikes.append(layer_spike_sum)
            local_comps.append(layer_comp_sum)
            current_regions = layer_outputs

        # --- Merge ---
        merged = np.concatenate(current_regions).astype(np.float32)

        # --- Global layers ---
        global_inputs  = []
        global_caches  = []
        global_outputs = []

        current = merged
        for layer in self.global_layers:
            global_inputs.append(current)
            y, cache, spikes, comparisons = layer.forward(current)
            global_caches.append(cache)
            global_spikes.append(spikes)
            global_comps.append(comparisons)
            global_outputs.append(y)
            current = y

        # --- Output LUT ---
        output_input = current
        logits, output_cache, output_spikes, output_comps = (
            self.output_lut.forward(output_input)
        )

        # Edge-detection comparison count:
        # sampled² pixels per region × N_GRID² regions × 2 active directions.
        # Each (pixel, direction) invokes one N(A,B,C) primitive = 2 comparisons,
        # so the factor of 2 in the formula encodes the two active directions.
        edge_comps = self.sampled * self.sampled * (N_GRID * N_GRID) * 2

        lut_total = (
            sum(local_comps) + sum(global_comps) + output_comps
        )

        spike_counts = {
            'local':  local_spikes,
            'global': global_spikes,
            'output': output_spikes,
            'total':  sum(local_spikes) + sum(global_spikes) + output_spikes,
        }
        seq2s = {
            'edge':   edge_comps,
            'local':  local_comps,
            'global': global_comps,
            'output': output_comps,
            'total':  edge_comps + lut_total,
        }

        y_outputs = (
            [y for lyr_outs in local_outputs for y in lyr_outs] + global_outputs
        )
        all_caches = {
            'local_inputs':  local_inputs,
            'local_caches':  local_caches,
            'global_inputs': global_inputs,
            'global_caches': global_caches,
            'output_input':  output_input,
            'output_cache':  output_cache,
            'y_outputs':     y_outputs,
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

        # grad shape now: (merge_dim,) = (n_regions × EMBED_DIM,)
        n_regions    = N_GRID * N_GRID
        region_grads = list(np.split(grad, n_regions))

        # Backward through local layers (reversed, weight-shared)
        for layer_idx in range(len(self.local_layers) - 1, -1, -1):
            layer = self.local_layers[layer_idx]
            new_region_grads = []
            for r_idx in range(n_regions):
                x_grad = layer.backward(
                    caches['local_inputs'][layer_idx][r_idx],
                    caches['local_caches'][layer_idx][r_idx],
                    region_grads[r_idx],
                    lr,
                )
                new_region_grads.append(x_grad)
            region_grads = new_region_grads
        # Edge preprocessing is fixed; input gradients are discarded.

        return loss, correct, spike_counts, seq2s
