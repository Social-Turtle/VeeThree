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
N_LOCAL   = 2    # local LUT layers per region
N_GLOBAL  = 2    # global LUT layers after merge
WARMUP    = 400  # LR warmup steps (SNN default 4000, scaled ~10x for MNIST)
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

        Written as explicit loops matching C's "strives for simplicity, not efficiency".
        Returns j (table row indices), r_min, u_min — each length N_T.
        """
        j     = np.zeros(N_T, dtype=np.int32)
        u_min = np.full(N_T, np.inf)
        r_min = np.zeros(N_T, dtype=np.int32)
        for i in range(N_T):
            for r in range(N_C):
                u = float(x[self.anchors_a[i, r]]) - float(x[self.anchors_b[i, r]])
                if u > 0:
                    j[i] |= (1 << r)
                if abs(u) < abs(u_min[i]):
                    r_min[i] = r
                    u_min[i] = u
        return j, r_min, u_min

    def forward(self, x):
        """LUT_forward (main.c:206-214).

        y[k] += S[i][j[i] * y_dim + k]  for i in 0..N_T
        Returns (y, cache) where cache = (j, r_min, u_min) for backward.
        """
        j, r_min, u_min = self.cache_index(x)
        y = np.zeros(self.y_dim, dtype=np.float32)
        for i in range(N_T):
            y += self.S[i, j[i]]
        return y, (j, r_min, u_min)

    def backward(self, x, cache, y_grad, lr):
        """LUT_backward (main.c:228-241).

        Updates S in-place. Returns x_grad.
        """
        j, r_min, u_min = cache
        x_grad = np.zeros(self.input_dim, dtype=np.float32)
        for i in range(N_T):
            jbar = j[i] ^ (1 << r_min[i])
            gi = float(y_grad @ (self.S[i, jbar] - self.S[i, j[i]]))
            v  = gi * Up(u_min[i])
            x_grad[self.anchors_a[i, r_min[i]]] += v
            x_grad[self.anchors_b[i, r_min[i]]] -= v
            self.S[i, j[i]] -= lr * y_grad
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

    def __init__(self):
        region_size = (28 // N_GRID) ** 2   # 49 for N_GRID=4
        n_regions   = N_GRID * N_GRID        # 16

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
        """Split 28×28 image into N_GRID² flattened region vectors."""
        cell = 28 // N_GRID
        regions = []
        for row in range(N_GRID):
            for col in range(N_GRID):
                patch = image[row*cell:(row+1)*cell, col*cell:(col+1)*cell]
                regions.append(patch.flatten().astype(np.float32))
        return regions  # list of N_GRID² arrays, each length cell²

    def forward(self, image):
        """Full forward pass.

        Returns (logits, all_caches).
        all_caches is a dict holding every intermediate needed for backward.
        """
        regions = self._split_regions(image)
        n_regions = len(regions)

        # --- Local layers (per region, weight-shared) ---
        local_inputs  = []   # local_inputs[layer_idx]  = list of n_regions input vectors
        local_caches  = []   # local_caches[layer_idx]  = list of n_regions caches
        local_outputs = []   # local_outputs[layer_idx] = list of n_regions output vectors

        current_regions = regions
        for layer in self.local_layers:
            layer_inputs  = list(current_regions)
            layer_caches  = []
            layer_outputs = []
            for x in current_regions:
                y, cache = layer.forward(x)
                layer_caches.append(cache)
                layer_outputs.append(y)
            local_inputs.append(layer_inputs)
            local_caches.append(layer_caches)
            local_outputs.append(layer_outputs)
            current_regions = layer_outputs

        # --- Merge: concatenate final region embeddings ---
        merged = np.concatenate(current_regions).astype(np.float32)

        # --- Global layers ---
        global_inputs  = []
        global_caches  = []

        current = merged
        for layer in self.global_layers:
            global_inputs.append(current)
            y, cache = layer.forward(current)
            global_caches.append(cache)
            current = y

        # --- Output LUT ---
        output_input = current
        logits, output_cache = self.output_lut.forward(output_input)

        all_caches = {
            'local_inputs':  local_inputs,
            'local_caches':  local_caches,
            'global_inputs': global_inputs,
            'global_caches': global_caches,
            'output_input':  output_input,
            'output_cache':  output_cache,
        }
        return logits, all_caches

    def step(self, image, label, t):
        """Forward + backward for one training sample.

        Returns (loss, correct).
        """
        lr = learning_rate(t)

        # Forward
        logits, caches = self.forward(image)

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

        return loss, correct
