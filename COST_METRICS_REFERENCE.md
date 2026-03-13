# VeeThree MNIST Cost Metrics Reference

## Overview

Five metrics quantify the computational and communication cost of LUT-based models (FE-LUT, Edge-LUT) versus conventional CNN on MNIST benchmarks. All metrics are summed across the entire forward pass.

---

## Metric Definitions

### **seq2s** — Ordering-Primitive Comparisons
Count of binary comparison operations in the rank-order primitive.

- **LUT models**: `N_T × N_C` per LUT layer. Each layer selects N_T sub-table rows via N_C binary comparisons (`u > 0`), where `u = x[a] - x[b]`. Each N(A,B,C) call counts as 2 comparisons.
- **Edge-LUT**: Includes additional comparisons from edge detection preprocessing.
- **CNN**: 0 (no ordering primitives).

*Intuition*: Direct hardware count of comparison instructions.

---

### **spikes** — Fired Comparison Bits
Count of comparison results that evaluate to true (1) across all binary operations.

- **LUT models**: popcount of all `(u > 0)` results. Measures active/firing comparisons during inference.
- **CNN**: 0 (no comparisons executed).

*Intuition*: Adopted from SNN terminology; tracks which "decision bits" are active in the LUT computation.

---

### **active_signals** — Transmitted Non-Zero Values (Bit-Weighted)
Count of non-zero activations transmitted through layers, weighted by bit cost.

- **LUT models**:
  - Input to first layer: 1 bit each (single voltage spike per non-zero pixel/feature value)
  - Binary spikes: 1 bit each
  - Non-zero float32 outputs from LUT layers: 1 bit each (rank-order timing values have fixed encoding width)
- **CNN**: `count_nonzero(layer_input) × 16` for each Conv/Linear layer input (including the raw image into the first layer). Assumes float32 activations with ~50% bit density on average (BIT_WIDTH / 2 = 32 / 2 = 16).

*Intuition*: Communication cost if activations are transmitted over a bus or between processor units.

---

### **adds** — Scalar Additions (Bit-Scaled)
Count of addition operations, scaled by operand bit-width (32 bits).

- **LUT models**: Each LUT layer sums N_T selected rows → `(N_T - 1) adds` per output element, multiplied by 32.
- **CNN**: All dot-product additions in Conv/Linear layers, multiplied by 32.

*Intuition*: Accumulation cost in systolic arrays or adder trees; scaling reflects energy cost of 32-bit arithmetic.

---

### **multiplies** — Scalar Multiplications (Bit-Scaled, O(n²))
Count of multiplication operations, scaled by operand bit-width **squared** (32² = 1024).

- **LUT models**: 0 (table lookups replace all multiplications).
- **CNN**: All dot-product multiplications in Conv/Linear layers, multiplied by 1024.

*Intuition*: Multiplying two n-bit numbers requires O(n²) bit operations (naive grade-school algorithm), making multiplication disproportionately more expensive than addition (O(n)). Squaring the bit-width scaling reflects this; the absence of multiplications in LUT models is therefore an even larger efficiency advantage than linear scaling would suggest.

---

## Metric Applicability

| Metric | LUT Models | CNN |
|--------|-----------|-----|
| **seq2s** | Count of comparisons per layer | 0 |
| **spikes** | Popcount of true comparisons | 0 |
| **active_signals** | Input spikes (1b) + binary spikes + nonzero outputs | count_nonzero × 16 per layer (incl. first) |
| **adds** | (N_T − 1) × 32 per layer | dot-product adds × 32 |
| **multiplies** | 0 | dot-product mults × 1024 (32²) |

---

## Key Insights for Comparison

- **LUT advantage**: Zero multiplications, small spike counts if comparisons are sparse.
- **CNN advantage**: Fixed architecture and well-understood complexity; no preprocessing overhead.
- **Trade-off**: LUT models exchange multiplication cost for comparison cost and table lookup latency; use `seq2s + spikes + active_signals` to estimate total LUT overhead vs. CNN's `adds + multiplies`.
