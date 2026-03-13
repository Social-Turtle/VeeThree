"""LUT-based MNIST classifier, ported from SNN/main.c (Izhikevich 2025).

Matrix multiplies replaced by rank-order comparison primitives.
No attention (MNIST has no sequence); spatial grid hierarchy substitutes.
All constants at top for easy experimentation.
"""

import numpy as np
import math

# ===========================================================================
# CONSTANTS (edit here for experiments)
# ===========================================================================

N_GRID    = 4    # regions per side → N_GRID×N_GRID non-overlapping regions
N_T       = 16   # sub-tables per LUT                  (SNN default, main.c:25)
N_C       = 6    # comparisons per sub-table            (SNN default, main.c:26)
EMBED_DIM = 32   # LUT output dimension                 (SNN default, main.c:22)
N_LOCAL      = 1    # local LUT layers per region
N_GLOBAL     = 2    # global LUT layers after merge
WARMUP       = 400  # LR warmup steps (SNN default 4000, scaled ~10x for MNIST)
INPUT_STRIDE = 1    # pixel stride for input sampling (1=every pixel, 2=every other, …)
N_CLASSES = 10


# ===========================================================================
# DIRECT C → PYTHON TRANSLATIONS (main.c)
# ===========================================================================

def _sign(u):
    """sign macro from main.c: zero → -1 (matching C 'x > 0 ? 1 : -1')."""
    return 1 if u > 0 else -1


def Up(u):
    """Marginal gradient function (main.c:33).

    #define Up(x) (-0.5*sign(x)/(1+fabs(x))/(1+fabs(x)))
    """
    return -0.5 * _sign(u) / (1 + abs(u)) / (1 + abs(u))


def _Up_vec(u):
    """Vectorized Up for NumPy arrays (same formula, array-safe sign)."""
    signs = np.where(u > 0, 1.0, -1.0)
    return -0.5 * signs / (1.0 + np.abs(u)) ** 2


def learning_rate(t):
    """LR schedule (main.c:35).

    #define LEARNING_RATE (MIN(1/sqrt(1+t), t/(4000)/sqrt(4000)))
    """
    return min(1.0 / math.sqrt(1 + t), t / WARMUP / math.sqrt(WARMUP))


def _softmax(x):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ===========================================================================
# LUT CLASS
# ===========================================================================

class LUT:
    """One Look-Up Table layer.

    Mirrors main.c structs:
        typedef struct { int a[N_C]; int b[N_C]; } Anchors;
        typedef struct { int y_dim; float* S[N_T]; Anchors anchors[N_T]; } LUT;

    anchors_a, anchors_b: shape (N_T, N_C) — fixed random indices, never updated
    S:                    shape (N_T, 2**N_C, y_dim) — learned, zero-initialized
    """

    def __init__(self, input_dim, y_dim):
        self.input_dim = input_dim
        self.y_dim = y_dim

        # anchors_a: random in [0, input_dim)  (main.c:151-160)
        anchors_a = np.random.randint(0, input_dim, size=(N_T, N_C))

        # anchors_b: random in [0, input_dim), different from a
        anchors_b = np.array([
            [np.random.choice([x for x in range(input_dim) if x != anchors_a[i, r]])
             for r in range(N_C)]
            for i in range(N_T)
        ])

        self.anchors_a = anchors_a  # (N_T, N_C)
        self.anchors_b = anchors_b  # (N_T, N_C)

        # S: zero-initialized (main.c calloc)
        self.S = np.zeros((N_T, 2 ** N_C, y_dim), dtype=np.float32)

    def cache_index(self, x):
        """Forward index computation (main.c:169-186).

        Vectorized: all N_T × N_C comparisons computed in two NumPy index ops.
        Returns j (table row indices), r_min, u_min — each length N_T — plus
        total spike count (popcount of all fired bits).
        """
        # u[i, r] = x[anchors_a[i,r]] - x[anchors_b[i,r]]   shape (N_T, N_C)
        u = x[self.anchors_a] - x[self.anchors_b]

        fired = u > 0                                           # (N_T, N_C) bool

        # j[i] = bitmask of which comparisons fired (u > 0)
        powers = (1 << np.arange(N_C, dtype=np.int32))         # [1, 2, 4, …, 32]
        j = (fired.astype(np.int32) * powers).sum(axis=1)      # (N_T,)

        # r_min[i] = index of the comparison with smallest |u|
        r_min = np.abs(u).argmin(axis=1)                       # (N_T,)
        u_min = u[np.arange(N_T), r_min]                       # (N_T,)

        spikes = int(fired.sum())                               # total popcount
        return j, r_min, u_min, spikes

    def forward(self, x):
        """LUT_forward (main.c:206-214).

        y[k] += S[i][j[i] * y_dim + k]  for i in 0..N_T
        Returns (y, cache, spikes, comparisons) where:
          cache       = (j, r_min, u_min) for backward
          spikes      = total active comparisons (popcount of all j[i])
          comparisons = N_T * N_C (total element comparisons made)
        """
        j, r_min, u_min, spikes = self.cache_index(x)
        # Gather the N_T selected rows and sum: S[arange, j] → (N_T, y_dim)
        y = self.S[np.arange(N_T), j].sum(axis=0)
        comparisons = N_T * N_C
        return y, (j, r_min, u_min), spikes, comparisons

    def backward(self, x, cache, y_grad, lr):
        """LUT_backward (main.c:228-241).

        Updates S in-place. Returns x_grad.
        """
        j, r_min, u_min = cache
        idx = np.arange(N_T)

        # Flip the minimum-margin bit to get the counterfactual row index
        jbar = j ^ (1 << r_min)                                    # (N_T,)

        # gi[i] = y_grad · (S[i,jbar[i]] - S[i,j[i]])
        diff = self.S[idx, jbar] - self.S[idx, j]                  # (N_T, y_dim)
        gi   = diff @ y_grad                                        # (N_T,)

        # v[i] = gi[i] * Up(u_min[i])
        v = gi * _Up_vec(u_min)                                     # (N_T,)

        # Scatter v into x_grad; anchors may repeat, so use np.add.at
        a_idx = self.anchors_a[idx, r_min]                         # (N_T,)
        b_idx = self.anchors_b[idx, r_min]                         # (N_T,)
        x_grad = np.zeros(self.input_dim, dtype=np.float32)
        np.add.at(x_grad, a_idx,  v)
        np.add.at(x_grad, b_idx, -v)

        # S update: each (i, j[i]) pair is unique (i is always distinct),
        # so plain fancy-index assignment is safe
        self.S[idx, j] -= lr * y_grad                              # (N_T, y_dim) broadcast

        return x_grad


# ===========================================================================
# LUT MODEL
# ===========================================================================

class LUTModel:
    """Spatial-grid LUT classifier for MNIST.

    Architecture:
        28×28 image
        → split into N_GRID×N_GRID non-overlapping regions (each (28//N_GRID)² px)
        → N_LOCAL LUT layers per region (weight-shared across regions)
        → concatenate all region embeddings  (N_GRID² × EMBED_DIM)
        → N_GLOBAL LUT layers on merged vector
        → output LUT → N_CLASSES logits

    Mirrors SNN's unembedder at the output (main.c:300).
    """

    def __init__(self, stride=INPUT_STRIDE):
        self.stride = stride
        cell        = 28 // N_GRID
        sampled     = len(range(0, cell, stride))   # pixels per side after striding
        region_size = sampled * sampled              # e.g. 49→stride=1, 16→stride=2
        n_regions   = N_GRID * N_GRID               # 16

        # Local LUT layers: weight-shared across all regions
        #   layer 0  : region_size → EMBED_DIM
        #   layer 1+ : EMBED_DIM  → EMBED_DIM
        self.local_layers = []
        in_dim = region_size
        for _ in range(N_LOCAL):
            self.local_layers.append(LUT(in_dim, EMBED_DIM))
            in_dim = EMBED_DIM

        # Global LUT layers on the concatenated region embeddings
        merge_dim = n_regions * EMBED_DIM
        self.global_layers = []
        in_dim = merge_dim
        for _ in range(N_GLOBAL):
            self.global_layers.append(LUT(in_dim, EMBED_DIM))
            in_dim = EMBED_DIM

        # Output LUT: EMBED_DIM → N_CLASSES
        self.output_lut = LUT(EMBED_DIM, N_CLASSES)

    def _split_regions(self, image):
        """Split 28×28 image into N_GRID² flattened region vectors.

        Pixels within each region are sampled at self.stride in both axes,
        so stride=2 keeps only every other pixel (rows and cols independently).
        """
        cell = 28 // N_GRID
        regions = []
        for row in range(N_GRID):
            for col in range(N_GRID):
                patch = image[row*cell:(row+1)*cell, col*cell:(col+1)*cell]
                regions.append(patch[::self.stride, ::self.stride].flatten().astype(np.float32))
        return regions  # list of N_GRID² arrays, each length sampled²

    def forward(self, image):
        """Full forward pass.

        Returns (logits, all_caches, spike_counts, seq2s).
        all_caches is a dict holding every intermediate needed for backward.
        """
        regions = self._split_regions(image)

        # --- Local layers (per region, weight-shared) ---
        local_inputs  = []   # local_inputs[layer_idx]  = list of n_regions input vectors
        local_caches  = []   # local_caches[layer_idx]  = list of n_regions caches
        local_outputs = []   # local_outputs[layer_idx] = list of n_regions output vectors

        local_spikes  = []   # spikes per local layer (summed across all regions)
        local_comps   = []   # comparisons per local layer (summed across all regions)
        global_spikes = []   # spikes per global layer
        global_comps  = []   # comparisons per global layer
        output_spikes = 0
        output_comps  = 0

        current_regions = regions
        for layer in self.local_layers:
            layer_inputs  = list(current_regions)
            layer_caches  = []
            layer_outputs = []
            layer_spike_sum = 0
            layer_comp_sum = 0
            for x in current_regions:
                y, cache, spikes, comparisons = layer.forward(x)
                layer_caches.append(cache)
                layer_outputs.append(y)
                layer_spike_sum += spikes
                layer_comp_sum += comparisons
            local_inputs.append(layer_inputs)
            local_caches.append(layer_caches)
            local_outputs.append(layer_outputs)
            local_spikes.append(layer_spike_sum)
            local_comps.append(layer_comp_sum)
            current_regions = layer_outputs

        # --- Merge: concatenate final region embeddings ---
        merged = np.concatenate(current_regions).astype(np.float32)

        # --- Global layers ---
        global_inputs   = []
        global_caches   = []
        global_outputs  = []   # y vector after each global layer

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
        logits, output_cache, output_spikes, output_comps = self.output_lut.forward(output_input)

        spike_counts = {
            'local':  local_spikes,   # list of ints, one per local layer
            'global': global_spikes,  # list of ints, one per global layer
            'output': output_spikes,  # int
            'total':  sum(local_spikes) + sum(global_spikes) + output_spikes,
        }

        seq2s = {
            'local':  local_comps,   # list of ints, one per local layer
            'global': global_comps,  # list of ints, one per global layer
            'output': output_comps,  # int
            'total':  sum(local_comps) + sum(global_comps) + output_comps,
        }

        # Flat list of all intermediate float32 y outputs (excluding final logits)
        y_outputs = [y for lyr_outs in local_outputs for y in lyr_outs] + global_outputs

        all_caches = {
            'local_inputs':  local_inputs,
            'local_caches':  local_caches,
            'global_inputs': global_inputs,
            'global_caches': global_caches,
            'output_input':  output_input,
            'output_cache':  output_cache,
            'y_outputs':     y_outputs,   # for active-bit cost accounting
        }
        return logits, all_caches, spike_counts, seq2s

    def step(self, image, label, t):
        """Forward + backward for one training sample.

        Returns (loss, correct, spike_counts, seq2s).
        """
        lr = learning_rate(t)

        # Forward
        logits, caches, spike_counts, seq2s = self.forward(image)

        # Softmax + cross-entropy loss (main.c:412-415 pattern)
        probs   = _softmax(logits.astype(np.float64)).astype(np.float32)
        loss    = -float(np.log(probs[label] + 1e-9))
        correct = int(np.argmax(probs) == label)

        # Gradient: softmax - one_hot (same pattern as C)
        grad = probs.copy()
        grad[label] -= 1.0

        # --- Backward through output LUT ---
        grad = self.output_lut.backward(
            caches['output_input'], caches['output_cache'], grad, lr
        )

        # --- Backward through global layers (reversed) ---
        for i in range(len(self.global_layers) - 1, -1, -1):
            grad = self.global_layers[i].backward(
                caches['global_inputs'][i], caches['global_caches'][i], grad, lr
            )

        # grad shape now: (merge_dim,) = (n_regions * EMBED_DIM,)
        # Split into per-region gradients
        n_regions    = N_GRID * N_GRID
        region_grads = list(np.split(grad, n_regions))

        # --- Backward through local layers (reversed, weight-shared) ---
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

        return loss, correct, spike_counts, seq2s
